import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def normalize(data: np.ndarray) -> np.ndarray:
    """Normalize a numpy array to zero mean and unit variance."""
    scaler = StandardScaler()
    orig_shape = data.shape
    data = data.reshape(len(data), -1)
    data = scaler.fit_transform(data)
    return data.reshape(orig_shape)


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    """Split data into train and validation sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
