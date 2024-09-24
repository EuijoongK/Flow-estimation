import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim

import model
from dataset import image_preprocess, CustomDataset
from loss_func import CustomFlowLoss

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Available device :", device)

input_dir = '/mnt/c/Users/MICS/Desktop/sample/opencv/vtest.avi'

input_data = image_preprocess.preprocess(input_dir)
ds = CustomDataset(input_data)
dataloader = DataLoader(dataset = ds, batch_size = 16, shuffle = True)

unet = model.UNet().to(device)

learning_rate = 1e-7
optimizer = optim.Adam(unet.parameters(), lr = learning_rate)
num_epochs = 50

epsilon = 2e-7
loss_fn = CustomFlowLoss(epsilon = epsilon)

for epoch in range(num_epochs):
    for data in dataloader:
        data = data.to(device)
        output = unet(data)
    
        prev_frame = data[:, 0, :, :].unsqueeze(1)
        current_frame = data[:, 1, :, :].unsqueeze(1)
        
        u = output[:, 0, :, :].unsqueeze(1)
        v = output[:, 1, :, :].unsqueeze(1)
        
        loss = loss_fn(prev_frame, current_frame, u, v)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss / len(dataloader)}")
        

model_name = './output/' + 'epoch_' + str(num_epochs) + '_' + 'lr_' + str(learning_rate) + '_' + 'ep_' + str(epsilon) + '.pth'

torch.save({
    'model_state_dict' : unet.state_dict(),
    'optimizer_state_dict' : optimizer.state_dict()
}, model_name)