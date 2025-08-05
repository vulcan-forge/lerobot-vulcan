import torch
from torch.utils.data import Dataset

class SyntheticDataset(Dataset):
    def __init__(self, num_samples=1000):
        # Generate synthetic data
        self.data = torch.randn(num_samples, 10)
        self.targets = torch.randn(num_samples, 1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx] 