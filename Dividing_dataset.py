"""
Created on Mon Oct 12 08:46:04 2020

@author: Sharjeel Masood
"""

import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import torch

data_path = 'C:\\Users\\Dutchfoundation\\Desktop\\FYP\\nano-degree simulator\\new_test_data_rgb.npy'
training_data = np.load(data_path, allow_pickle = True)
np.random.shuffle(training_data)


'''            Data preperation using Pytorch custom class            '''

class SimulationData(Dataset):
    def __init__(self, data_path):
        self.training_data = np.load(data_path, allow_pickle=True)
        self.images = training_data[:, 0]
        self.targets = training_data[:, 1]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        
        image = image/255
        
        return image, target
        
seg_dataset = SimulationData(data_path)
loader = DataLoader(seg_dataset, batch_size = 100, shuffle = True, num_workers = 0)


