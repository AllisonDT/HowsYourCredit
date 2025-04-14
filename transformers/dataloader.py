import torch
from torch.utils.data import Dataset

import numpy as np

class CreditDataset(Dataset):
    def __init__(self, data_path_X, data_path_y):
        self.X = torch.tensor(np.load(data_path_X), dtype=torch.float32)
        self.y = torch.tensor(np.load(data_path_y), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
