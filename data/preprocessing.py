"""
Signal processing and preprocessing utilities for the MentorPDM project.

This module contains functions for signal normalization, segmentation, 
feature extraction, and frequency domain transformations.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import rfft
from scipy.stats import kurtosis, skew, shapiro
from sklearn.neighbors import NearestNeighbors
from typing import List, Union, Tuple, Dict, Any
import warnings


def normalize_signal(df: pd.DataFrame, data_cat: Union[str, List[str]]) -> pd.DataFrame:
    """
    Normalize the signals in the DataFrame by subtracting
    the mean and dividing by the standard deviation.
    
    Parameters:
        df: DataFrame containing signal data
        data_cat: Data category or list of categories to normalize
        
    Returns:
        DataFrame with normalized signals
    """
    if type(data_cat) is not list:
        data_cat = [data_cat]
    
    for category in data_cat:
        if category == 'label':
            continue
        df[category] = df[category].apply(
            lambda x: [(i - np.mean(x)) / np.std(x) for i in x]
        )
    
    return df


def divide_signal(df: pd.DataFrame, segment_length: int, data_cat: Union[str, List[str]]) -> pd.DataFrame:
    """
    Divide signals into segments of specified length.
    
    This function takes long signals and divides them into smaller segments
    for training purposes. Each segment becomes a separate row in the output DataFrame.
    
    Parameters:
        df: DataFrame containing signal data
        segment_length: Length of each segment
        data_cat: Data category or list of categories to segment
        
    Returns:
        DataFrame with segmented signals
    """
    if isinstance(data_cat, str):
        data_cat = [data_cat]
    
    segmented_data = []
    
    for _, row in df.iterrows():
        filename = row.get('filename', '')
        label = row.get('label', '')
        
        # Find the longest signal to determine number of segments
        max_length = 0
        for cat in data_cat:
            if cat in row and len(row[cat]) > max_length:
                max_length = len(row[cat])
        
        # Calculate number of segments
        num_segments = max_length // segment_length
        
        # Create segments
        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = start_idx + segment_length
            
            segment_row = {
                'filename': f"{filename}_seg_{i}",
                'label': label
            }
            
            # Add segmented data for each category
            for cat in data_cat:
                if cat in row and len(row[cat]) >= end_idx:
                    segment_row[cat] = row[cat][start_idx:end_idx]
                else:
                    # Pad with zeros if signal is shorter
                    if cat in row:
                        signal_segment = row[cat][start_idx:min(end_idx, len(row[cat]))]
                        padding_length = segment_length - len(signal_segment)
                        if padding_length > 0:
                            signal_segment.extend([0] * padding_length)
                        segment_row[cat] = signal_segment
                    else:
                        segment_row[cat] = [0] * segment_length
            
            segmented_data.append(segment_row)
    
    return pd.DataFrame(segmented_data)


def FE(lst: List[float]) -> List[float]:
    """
    Extract statistical and signal processing features from a time series.
    
    This function computes various features commonly used in bearing fault diagnosis:
    - Zero crossings
    - Statistical moments (mean, std, skewness, kurtosis)
    - RMS value
    - Peak analysis
    - Energy measures
    
    Parameters:
        lst: Input time series data
        
    Returns:
        List of extracted features
    """
    try:
        from scipy.signal import find_peaks
    except ImportError:
        # Fallback peak detection
        def find_peaks(data):
            peaks = []
            for i in range(1, len(data) - 1):
                if data[i] > data[i-1] and data[i] > data[i+1]:
                    peaks.append(i)
            return peaks, {}
    
    feature_array = []
    lst = np.array(lst)
    
    # Zero crossing
    zero_cross = ((lst[:-1] * lst[1:]) < 0).sum() + (lst == 0).sum()
    feature_array.append(zero_cross)
    
    # Kurtosis
    kurt = kurtosis(lst)
    feature_array.append(kurt)
    
    # RMS (Root Mean Square)
    rms = np.sqrt(np.mean(lst**2))
    feature_array.append(rms)
    
    # Number of peaks (normalized by signal length)
    peaks, _ = find_peaks(lst)
    feature_array.append(len(peaks) / len(lst))
    
    # Mean
    feature_array.append(np.mean(lst))
    
    # Median of absolute values
    feature_array.append(np.median(np.abs(lst)))
    
    # Standard deviation
    feature_array.append(np.std(lst))
    
    # Skewness
    feature_array.append(skew(lst))
    
    # Energy (sum of squared values)
    energy = np.sum(lst**2)
    feature_array.append(energy)
    
    # Crest factor (peak to RMS ratio)
    if rms > 0:
        crest_factor = np.max(np.abs(lst)) / rms
    else:
        crest_factor = 0
    feature_array.append(crest_factor)
    
    # Shape factor (RMS to mean of absolute values)
    mean_abs = np.mean(np.abs(lst))
    if mean_abs > 0:
        shape_factor = rms / mean_abs
    else:
        shape_factor = 0
    feature_array.append(shape_factor)
    
    # Impulse factor (peak to mean of absolute values)
    if mean_abs > 0:
        impulse_factor = np.max(np.abs(lst)) / mean_abs
    else:
        impulse_factor = 0
    feature_array.append(impulse_factor)
    
    return feature_array


def FE_comprehensive(df: pd.DataFrame, signal_columns: List[str], 
                     size_const: int = 1000) -> pd.DataFrame:
    """
    Comprehensive feature extraction for multiple signals.
    
    This function extracts features from horizontal and vertical vibration signals
    using a sliding window approach.
    
    Parameters:
        df: DataFrame containing signal data
        signal_columns: List of column names containing signal data
        size_const: Window size for feature extraction
        
    Returns:
        DataFrame with extracted features
    """
    features_dict = {}
    
    # Initialize feature lists for each signal
    for col in signal_columns:
        features_dict[f'{col}_zerocross'] = []
        features_dict[f'{col}_kurtosis'] = []
        features_dict[f'{col}_rms'] = []
        features_dict[f'{col}_peaks'] = []
        features_dict[f'{col}_mean'] = []
        features_dict[f'{col}_std'] = []
        features_dict[f'{col}_median'] = []
        features_dict[f'{col}_skewness'] = []
        features_dict[f'{col}_energy'] = []
        features_dict[f'{col}_shapiro'] = []
        features_dict[f'{col}_crest'] = []
    
    # Extract features using sliding window
    for i in range(0, len(df), size_const):
        for col in signal_columns:
            if col in df.columns:
                # Get signal segment
                signal_data = signal.decimate(
                    df[col][i : i + size_const], 2
                )
                
                # Zero crossing
                zero_cross = ((signal_data[:-1] * signal_data[1:]) < 0).sum() + (signal_data == 0).sum()
                features_dict[f'{col}_zerocross'].append(zero_cross)
                
                # Kurtosis
                kurt = kurtosis(signal_data)
                features_dict[f'{col}_kurtosis'].append(kurt)
                
                # RMS
                rms = np.sqrt(np.mean(signal_data**2))
                features_dict[f'{col}_rms'].append(rms)
                
                # Number of peaks
                try:
                    from scipy.signal import find_peaks
                    peaks, _ = find_peaks(signal_data)
                    peak_count = len(peaks) / len(signal_data)
                except ImportError:
                    peak_count = 0
                features_dict[f'{col}_peaks'].append(peak_count)
                
                # Statistical features
                features_dict[f'{col}_mean'].append(np.mean(signal_data))
                features_dict[f'{col}_std'].append(np.std(signal_data))
                features_dict[f'{col}_median'].append(np.median(np.abs(signal_data)))
                features_dict[f'{col}_skewness'].append(skew(signal_data))
                
                # Energy
                energy = np.sum(signal_data**2)
                features_dict[f'{col}_energy'].append(energy)
                
                # Shapiro-Wilk test (p-value)
                try:
                    _, p_value = shapiro(signal_data[:5000] if len(signal_data) > 5000 else signal_data)
                    features_dict[f'{col}_shapiro'].append(p_value)
                except:
                    features_dict[f'{col}_shapiro'].append(0)
                
                # Crest factor
                if rms > 0:
                    crest = np.max(np.abs(signal_data)) / rms
                else:
                    crest = 0
                features_dict[f'{col}_crest'].append(crest)
    
    return pd.DataFrame(features_dict)


def ToFrequency(data: Union[List, np.ndarray], n_features: int = 128, 
                downsample_factor: int = 5) -> np.ndarray:
    """
    Convert time series data to frequency domain features.
    
    This function performs downsampling followed by FFT to extract
    frequency domain features from time series data.
    
    Parameters:
        data: Input time series data
        n_features: Number of frequency features to extract
        downsample_factor: Factor by which to downsample the signal
        
    Returns:
        Array of frequency domain features
    """
    data = np.array(data)
    
    # Ensure the signal is long enough for downsampling
    if len(data) > downsample_factor * n_features:
        # Downsample the time series data if possible
        downsampled = signal.decimate(data, downsample_factor, ftype='fir', zero_phase=True)
    else:
        # If not, just use the original data
        downsampled = data
    
    # Compute the one-dimensional n-point discrete Fourier Transform
    fft_result = np.abs(rfft(downsampled))
    
    # Keep only the first n_features components
    return fft_result[:n_features]


def normalize_column(arr_list: List[np.ndarray]) -> List[np.ndarray]:
    """
    Normalize a list of arrays column-wise using min-max normalization.
    
    Parameters:
        arr_list: List of arrays to normalize
        
    Returns:
        List of normalized arrays
    """
    # Stack the list of arrays to find global min and max
    all_data = np.vstack(arr_list)
    min_val = np.min(all_data, axis=0)
    max_val = np.max(all_data, axis=0)
    
    # Avoid division by zero
    range_val = max_val - min_val
    range_val[range_val == 0] = 1
    
    # Normalize each array column-wise
    normalized_list = [(arr - min_val) / range_val for arr in arr_list]
    
    return normalized_list


def check_data_shapes(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check and report the shapes of data in DataFrame columns.
    
    Parameters:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with shape information for each column
    """
    shape_info = {}
    
    for col in df.columns:
        if col in ['filename', 'label']:
            shape_info[col] = f"Text column with {len(df[col].unique())} unique values"
        else:
            try:
                # Check if column contains lists/arrays
                sample_data = df[col].iloc[0]
                if isinstance(sample_data, (list, np.ndarray)):
                    lengths = [len(x) for x in df[col] if isinstance(x, (list, np.ndarray))]
                    if lengths:
                        shape_info[col] = {
                            'type': 'Array/List column',
                            'min_length': min(lengths),
                            'max_length': max(lengths),
                            'mean_length': np.mean(lengths),
                            'unique_lengths': len(set(lengths))
                        }
                    else:
                        shape_info[col] = 'Empty or invalid array column'
                else:
                    shape_info[col] = f"Scalar column with type {type(sample_data)}"
            except Exception as e:
                shape_info[col] = f"Error analyzing column: {e}"
    
    return shape_info


def create_graph_data(embeddings: np.ndarray, labels: np.ndarray, k: int = 8) -> Tuple:
    """
    Create graph data for PyTorch Geometric from embeddings using k-nearest neighbors.
    
    This function creates a k-NN graph from feature embeddings and prepares it
    for use with PyTorch Geometric.
    
    Parameters:
        embeddings: Feature embeddings array
        labels: Labels array
        k: Number of nearest neighbors
        
    Returns:
        Tuple containing (edge_index, edge_attr, node_features, labels)
    """
    try:
        import torch
        from scipy.sparse import coo_matrix
    except ImportError:
        raise ImportError("PyTorch and/or SciPy not available for graph data creation")
    
    # Create k-NN graph
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(embeddings)
    distances, indices = knn.kneighbors(embeddings)
    
    # Convert the indices matrix to a COO sparse matrix format for PyTorch Geometric
    edge_indices = np.stack([
        np.repeat(np.arange(embeddings.shape[0]), k), 
        indices.flatten()
    ])
    edge_weights = distances.flatten()
    
    # Create sparse matrix
    sparse_matrix = coo_matrix(
        (np.ones(embeddings.shape[0] * k), edge_indices), 
        shape=(embeddings.shape[0], embeddings.shape[0])
    )
    
    # Convert to PyTorch tensors
    edge_index = torch.LongTensor(np.vstack((sparse_matrix.row, sparse_matrix.col)))
    edge_attr = torch.FloatTensor(edge_weights)
    node_features = torch.FloatTensor(embeddings)
    node_labels = torch.LongTensor(labels)
    
    return edge_index, edge_attr, node_features, node_labels


def create_pytorch_geometric_data(embeddings: np.ndarray, labels: np.ndarray, k: int = 8):
    """
    Create PyTorch Geometric Data object from embeddings.
    
    Parameters:
        embeddings: Feature embeddings
        labels: Labels for each sample
        k: Number of nearest neighbors for graph construction
        
    Returns:
        PyTorch Geometric Data object
    """
    try:
        from torch_geometric.data import Data
        import torch
    except ImportError:
        raise ImportError("PyTorch Geometric not available. Install with: pip install torch-geometric")
    
    edge_index, edge_attr, node_features, node_labels = create_graph_data(embeddings, labels, k)
    
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=node_labels
    )
    
    return data


def extract_time_domain_features(signal_data: np.ndarray) -> Dict[str, float]:
    """
    Extract comprehensive time domain features from a signal.
    
    Parameters:
        signal_data: Input signal array
        
    Returns:
        Dictionary of extracted features
    """
    features = {}
    
    # Basic statistical features
    features['mean'] = np.mean(signal_data)
    features['std'] = np.std(signal_data)
    features['var'] = np.var(signal_data)
    features['rms'] = np.sqrt(np.mean(signal_data**2))
    features['peak'] = np.max(np.abs(signal_data))
    features['min'] = np.min(signal_data)
    features['max'] = np.max(signal_data)
    features['range'] = features['max'] - features['min']
    
    # Higher order moments
    features['skewness'] = skew(signal_data)
    features['kurtosis'] = kurtosis(signal_data)
    
    # Shape factors
    if features['rms'] > 0:
        features['crest_factor'] = features['peak'] / features['rms']
        features['clearance_factor'] = features['peak'] / (np.mean(np.sqrt(np.abs(signal_data)))**2)
    else:
        features['crest_factor'] = 0
        features['clearance_factor'] = 0
    
    if np.mean(np.abs(signal_data)) > 0:
        features['shape_factor'] = features['rms'] / np.mean(np.abs(signal_data))
        features['impulse_factor'] = features['peak'] / np.mean(np.abs(signal_data))
    else:
        features['shape_factor'] = 0
        features['impulse_factor'] = 0
    
    # Zero crossings
    features['zero_crossings'] = ((signal_data[:-1] * signal_data[1:]) < 0).sum()
    
    # Energy
    features['energy'] = np.sum(signal_data**2)
    
    return features


def extract_frequency_domain_features(signal_data: np.ndarray, 
                                      sampling_rate: float = 1.0) -> Dict[str, float]:
    """
    Extract frequency domain features from a signal.
    
    Parameters:
        signal_data: Input signal array
        sampling_rate: Sampling rate of the signal
        
    Returns:
        Dictionary of extracted frequency features
    """
    features = {}
    
    # Compute FFT
    fft_result = np.fft.fft(signal_data)
    fft_magnitude = np.abs(fft_result[:len(fft_result)//2])
    fft_frequencies = np.fft.fftfreq(len(signal_data), 1/sampling_rate)[:len(fft_result)//2]
    
    # Power spectral density
    psd = fft_magnitude**2
    
    # Frequency domain features
    features['spectral_centroid'] = np.sum(fft_frequencies * fft_magnitude) / np.sum(fft_magnitude)
    features['spectral_spread'] = np.sqrt(np.sum(((fft_frequencies - features['spectral_centroid'])**2) * fft_magnitude) / np.sum(fft_magnitude))
    features['spectral_rolloff'] = fft_frequencies[np.where(np.cumsum(psd) >= 0.85 * np.sum(psd))[0][0]]
    features['spectral_flatness'] = np.exp(np.mean(np.log(psd + 1e-10))) / (np.mean(psd) + 1e-10)
    
    # Peak frequency
    peak_idx = np.argmax(fft_magnitude)
    features['peak_frequency'] = fft_frequencies[peak_idx]
    features['peak_magnitude'] = fft_magnitude[peak_idx]
    
    # Band power features
    total_power = np.sum(psd)
    features['total_power'] = total_power
    
    # Divide frequency bands
    n_bands = 5
    band_edges = np.linspace(0, len(fft_magnitude), n_bands + 1, dtype=int)
    for i in range(n_bands):
        band_power = np.sum(psd[band_edges[i]:band_edges[i+1]])
        features[f'band_{i+1}_power'] = band_power
        features[f'band_{i+1}_power_ratio'] = band_power / (total_power + 1e-10)
    
    return features