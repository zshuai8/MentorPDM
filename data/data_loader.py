"""
Data loading utilities for the MentorPDM project.

This module contains functions for loading and preprocessing bearing fault data
from MATLAB files, specifically designed for the Paderborn University bearing dataset.
"""

import os
import sys
import errno
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from typing import List, Dict, Union, Tuple, Any


def get_df_all(data_path: Union[str, Path], data_cat: str, segment_length: int = 512, 
               normalize: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load, preprocess and return a DataFrame which contains all signals data and
    labels and is ready to be used for model training.
    
    Parameters:
        data_path: Path of the folder which contains matlab files of bearings
        data_cat: Data category of interest, i.e. 'force', 'phase_current_1', 'phase_current_2',
                  'speed', 'temp_2_bearing_module', 'torque', 'vibration_1'
        segment_length: Number of points per segment. See divide_signal() function
        normalize: Boolean to perform normalization to the signal data
        
    Returns:
        Tuple containing:
        - df_processed: DataFrame which is ready to be used for model training
        - df_raw: Raw DataFrame before processing
    """
    from .preprocessing import normalize_signal, divide_signal
    
    df = matfile_to_df(data_path, data_cat)

    if normalize:
        normalize_signal(df, data_cat)
    df_processed = divide_signal(df, segment_length, data_cat)

    map_label = {'NORMAL': 0, 'IR': 1, 'OR': 2, 'OR + IR': 3}
    df_processed['label'] = df_processed['label'].map(map_label)
    return df_processed, df


def matfile_to_df(data_path: Union[str, Path], data_cat: Union[str, List[str]]) -> pd.DataFrame:
    """
    Convert MATLAB files to DataFrame format.
    
    This function loads all .mat files from the specified directory and converts
    them to a pandas DataFrame with the specified data categories.
    
    Parameters:
        data_path: Path to directory containing .mat files
        data_cat: Data category or list of categories to extract
        
    Returns:
        DataFrame containing the loaded data with labels
    """
    data_path = Path(data_path)
    if isinstance(data_cat, str):
        data_cat = [data_cat]
    
    all_data = []
    
    # Process all .mat files in the directory
    for mat_file in data_path.glob("*.mat"):
        try:
            mat_dict = loadmat(str(mat_file))
            # Get the main data (usually the last key that's not metadata)
            data_key = [k for k in mat_dict.keys() if not k.startswith('__')][-1]
            file_data = mat_dict[data_key]
            
            # Extract filename without extension for labeling
            filename = mat_file.stem
            
            # Create row data
            row_data = {'filename': filename}
            
            # Add data categories
            for cat in data_cat:
                if isinstance(file_data, np.ndarray) and file_data.size > 0:
                    # Handle different data structures
                    if file_data.ndim > 1 and file_data.shape[0] == 1:
                        row_data[cat] = file_data[0].tolist()
                    else:
                        row_data[cat] = file_data.tolist()
                else:
                    row_data[cat] = []
            
            # Add label based on filename
            row_data['label'] = label(filename)
            all_data.append(row_data)
            
        except Exception as e:
            print(f"Error processing {mat_file}: {e}")
            continue
    
    return pd.DataFrame(all_data)


def matfile_to_dic(data_path: Union[str, Path]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Convert all MATLAB files in a directory to a dictionary format.
    
    Parameters:
        data_path: Path to directory containing .mat files
        
    Returns:
        Dictionary with filename as key and data dictionary as value
    """
    data_path = Path(data_path)
    result_dict = {}
    
    for mat_file in data_path.glob("*.mat"):
        try:
            mat_dict = loadmat(str(mat_file))
            filename = mat_file.stem
            
            # Extract the main data structure
            data_key = [k for k in mat_dict.keys() if not k.startswith('__')][-1]
            file_data = mat_dict[data_key]
            
            # Parse the structured data
            if isinstance(file_data, np.ndarray) and file_data.dtype.names:
                # Structured array - extract each field
                file_dict = {}
                for field_name in file_data.dtype.names:
                    field_data = file_data[field_name][0, 0]
                    if isinstance(field_data, np.ndarray):
                        file_dict[field_name] = field_data.flatten()
                    else:
                        file_dict[field_name] = field_data
                result_dict[filename] = file_dict
            else:
                # Regular array
                result_dict[filename] = {'data': file_data}
                
        except Exception as e:
            print(f"Error processing {mat_file}: {e}")
            continue
    
    return result_dict


def label(filename: str) -> str:
    """
    Extract bearing condition label from filename.
    
    Based on Paderborn University bearing dataset naming convention:
    - Files without alphabet suffix are healthy bearings (NORMAL)
    - KA* files indicate outer race damage (OR)
    - KI* files indicate inner race damage (IR)
    - Files with both damages are labeled as OR + IR
    
    Parameters:
        filename: Name of the file (without extension)
        
    Returns:
        Label string: 'NORMAL', 'OR', 'IR', or 'OR + IR'
    """
    filename = filename.upper()
    
    if filename.startswith('KA'):
        return 'OR'  # Outer Race damage
    elif filename.startswith('KI'):
        return 'IR'  # Inner Race damage
    elif 'KA' in filename and 'KI' in filename:
        return 'OR + IR'  # Both damages
    else:
        return 'NORMAL'  # Healthy bearing


def download(url: str, filepath: Union[str, Path], create_dirs: bool = True) -> None:
    """
    Download a file from URL to the specified filepath.
    
    Parameters:
        url: URL to download from
        filepath: Local path to save the file
        create_dirs: Whether to create directories if they don't exist
    """
    filepath = Path(filepath)
    
    if create_dirs:
        _mkdir(filepath.parent)
    
    print(f"Downloading to: '{filepath}'")
    urllib.request.urlretrieve(url, str(filepath))


def _mkdir(path: Union[str, Path]) -> None:
    """
    Create directory if it doesn't exist.
    
    Parameters:
        path: Directory path to create
    """
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if exc.errno == errno.EEXIST and path.is_dir():
            pass
        else:
            print(f"Can't create directory '{path}'")
            sys.exit(1)


class BearingDataLoader:
    """
    Class for loading and processing bearing fault data from various sources.
    
    This class handles data loading from different bearing fault datasets,
    with support for segmentation and threshold-based selection.
    """
    
    def __init__(self, experiment: str, seq_len: int, *bearing_element):
        """
        Initialize the bearing data loader.
        
        Parameters:
            experiment: Type of experiment ('Artificial', 'Healthy', 'Real')
            seq_len: Sequence length for data segmentation
            bearing_element: Bearing elements to load ('OR', 'IR', 'Normal')
        """
        if experiment not in ('Artificial', 'Healthy', 'Real'):
            print(f"Wrong experiment name: {experiment}")
            sys.exit(1)
        
        for element in bearing_element:
            if element not in ('OR', 'IR', 'Normal'):
                print(f"Wrong bearing element value: {bearing_element}")
                sys.exit(1)
        
        # Root directory of all data
        self.rdir = os.path.join(os.path.expanduser('~'), 'Datasets')
        self.experiment = experiment
        self.bearing_elements = bearing_element
        self.seq_len = seq_len
        self.empty_list = []
        self.y_list = []
        
        self.read_matfiles(self.rdir, experiment, bearing_element)
        self.threshold_selector()
        self.data_divider()
    
    def read_matfiles(self, directory: str, experiment: str, bearing_element: tuple):
        """
        Read .mat files based on bearing element damage and experiment name.
        
        Parameters:
            directory: Root directory containing data
            experiment: Experiment type
            bearing_element: Tuple of bearing elements to process
        """
        y_divider = 0
        self.y_list = []
        directory = os.path.join(directory, experiment)
        self.empty_list = []
        file_names = []
        
        # Read .mat files based on the bearing element damage and the experiment name
        for paths, dirs, files in os.walk(directory):
            print(paths, dirs, files)
            if paths.endswith(bearing_element):
                for file in files:
                    y_divider += 1
                    print(y_divider)
                    self.y_list.append(1)
                    
                    if '.mat' in file:
                        mat_dict = loadmat(os.path.join(paths, file))
                        file_data = mat_dict[list(mat_dict.keys())[-1]]
                        file_names.append(file)
                        
                        # Extract sensor data
                        for index, data in enumerate(file_data[0][0]):
                            if len(data) > 200000:  # Ensure sufficient data points
                                self.empty_list.append([data])
    
    def threshold_selector(self):
        """
        Select data based on threshold criteria.
        
        Iterate over every file to receive sensor values sampled at 64 KHz
        for approximately 4 seconds per file (>200k datapoints).
        """
        print("Applying threshold selection...")
        threshold_list = []
        labels = []
        
        for index, data_list in enumerate(self.empty_list, self.seq_len):
            for data_group in data_list:
                for data_array in data_group:
                    if len(data_array) > 200000:
                        threshold_list.append(data_array)
                        # Assign labels based on data characteristics
                        if index < len(self.y_list):
                            labels.append(self.y_list[index])
        
        self.threshold_list = threshold_list
        self.labels = labels
    
    def data_divider(self):
        """Divide the data into training segments."""
        print("Dividing data into segments...")
        self.segmented_data = []
        
        for data_array in self.threshold_list:
            segments = self.slicer(data_array, self.seq_len)
            self.segmented_data.extend(segments)
    
    def slicer(self, time_series: np.ndarray, seq_len: int) -> List[np.ndarray]:
        """
        Slice time series into overlapping segments.
        
        Parameters:
            time_series: Input time series data
            seq_len: Length of each segment
            
        Returns:
            List of time series segments
        """
        segments = []
        step_size = seq_len // 2  # 50% overlap
        
        for i in range(0, len(time_series) - seq_len + 1, step_size):
            segment = time_series[i:i + seq_len]
            segments.append(segment)
        
        return segments
    
    def most_frequent(self, list_values: List) -> Any:
        """
        Find the most frequent value in a list.
        
        Parameters:
            list_values: List of values
            
        Returns:
            Most frequent value
        """
        return max(set(list_values), key=list_values.count)