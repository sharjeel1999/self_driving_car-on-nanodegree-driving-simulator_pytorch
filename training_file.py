"""
Created on Sun Oct 11 09:02:03 2020

@author: Sharjeel Masood
"""

from Model import Main_NN
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms

class SimulationData(Dataset):
    def __init__(self, data_path):
        self.training_data = np.load(data_path, allow_pickle=True)
        self.images = self.training_data[:, 0]
        self.targets = self.training_data[:, 1]
        self.tranform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.6473, 0.5195, 0.4208],
                             std=[0.1576, 0.0982, 0.0856])
        ])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        
        image = self.tranform(image)
        
        return image, target
        

data_path = 'C:\\Users\\Dutchfoundation\\Desktop\\FYP\\nano-degree simulator\\new_test_data_rgb.npy'
dataset = SimulationData(data_path)

train_set, val_set = random_split(dataset, [2500, 475])
train_loader = DataLoader(train_set, batch_size = 32, shuffle = True, num_workers = 0, pin_memory=True)
val_loader = DataLoader(val_set, batch_size = 32, shuffle = True, num_workers = 0, pin_memory=True)


model = Main_NN()
model.train()

epochs = 10

optimizer = optim.Adam(model.parameters(), lr=0.001, momentum=0.9)
loss_function = nn.MSELoss()

for epoch in tqdm(range(epochs)):
    t_average_loss = 0
    t_count = 0
    t_total_loss = 0
    v_average_loss = 0
    v_count = 0
    v_total_loss = 0
    
    for sample in train_loader:
        images, targets = sample
        
        optimizer.zero_grad()
        
        outputs = model(images)
        outputs = outputs.squeeze()

        loss = loss_function(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        t_total_loss += loss.item()
        t_count += 1
        t_average_loss = t_total_loss / t_count
        
    for v_sample in val_loader:
        v_images, v_targets = v_sample
        
        with torch.no_grad():
            outputs = model(v_images)
        
        outputs = outputs.squeeze()
        loss = loss_function(outputs, v_targets)
        
        v_total_loss += loss.item()
        v_count += 1
        v_average_loss = v_total_loss / v_count
    
    print("\n", f"Epoch: {epoch+1}", "\n",f"Training Loss: {t_average_loss}", "\n", f"Val Loss: {v_average_loss}", "\n")

torch.save(model.state_dict(),"Trained_model.pth")

