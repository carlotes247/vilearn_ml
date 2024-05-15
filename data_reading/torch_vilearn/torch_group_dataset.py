import numpy as np
import torch
from torch.utils.data import Dataset

class TorchGroupDataset(Dataset):
    def __init__(self, group_file_path: str, transform=None):
        self.transform = transform
        xy = np.loadtxt(group_file_path, delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) # n_samples, 1
        self.n_samples = xy.shape[0]        

    def __len__(self, index):
        return self.n_samples
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    

# testing 
if __name__ == '__main__':
    dataset = TorchGroupDataset('C:/Users/gonzacar/Documents/Unity_Projects/ViLearn/Recordings/2024_Group_Features/TRIAD_2023_10_30_Seminar_Munich_No_VAD_edit.csv')
    print('hola')