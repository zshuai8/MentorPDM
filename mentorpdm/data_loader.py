import os
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.io import loadmat


class PaderbornDataset(Dataset):
    """Simple loader for the Paderborn bearing dataset."""

    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.samples = []
        self.labels = []
        self.transform = transform

        for root, dirs, files in os.walk(root_dir):
            for f in files:
                if f.endswith('.mat'):
                    label = os.path.basename(root)
                    self.samples.append(os.path.join(root, f))
                    self.labels.append(label)

        self.label2idx = {l: i for i, l in enumerate(sorted(set(self.labels)))}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path = self.samples[idx]
        data = loadmat(path)
        # take first non-meta key
        key = next(k for k in data.keys() if not k.startswith('__'))
        array = data[key]
        array = np.asarray(array)
        if self.transform:
            array = self.transform(array)
        array = torch.tensor(array, dtype=torch.float32)
        if array.ndim == 1:
            array = array.unsqueeze(0)
        label = self.label2idx[self.labels[idx]]
        return array, label


def create_loaders(dataset: Dataset, batch_size: int = 32, val_ratio: float = 0.2):
    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    return train_loader, val_loader
