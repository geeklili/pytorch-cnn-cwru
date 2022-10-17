import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.data = np.load('./data/data.npy')

    def __getitem__(self, index):
        return self.data[index][:2048][None, :], self.data[index][2048:]

    def __len__(self):
        return len(self.data)
