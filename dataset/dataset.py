import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

all = [
    'CustomDataset'
]

class CustomDataset(Dataset):
    # def __init__(self, data):
    #     self.data = data
    #     
    # def __len__(self):
    #     return len(self.data)
    
    def __init__(self, input_list, gt_list):
        self.input_list = input_list
        self.gt_list = gt_list
    
    def __len__(self):
        return len(self.input_list)
    
    def __getitem__(self, index):
        input = self.input_list[index]
        gt = self.gt_list[index]

        input = input.permute(2, 0, 1)
        gt = gt.unsqueeze(0)
        
        return input, gt
    
    
    # def __getitem__(self, index):
    #     return torch.from_numpy(self.data[index]).float()
