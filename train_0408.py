import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import cv2

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim

import model
from dataset import CustomDataset, image_preprocess

import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# '0408_3_0006'

train_dir = ['0408_0000', '0408_0001', '0408_3_0004']
base_dir = '/mnt/c/Users/MICS/Desktop/test/needle_guidance/img/'

input_list = []
gt_list = []

for folder in train_dir:
    input_dir = base_dir + folder + '_hsv/'
    gt_dir = base_dir + folder + '_gt/'
    
    input_files = os.listdir(input_dir)
    gt_files = os.listdir(gt_dir)
    
    for input_file, gt_file in zip(input_files, gt_files):
        input_img = cv2.imread(input_dir + input_file)
        gt_img = cv2.imread(gt_dir + gt_file)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
        
        input_img = cv2.resize(input_img, (256, 256))
        gt_img = cv2.resize(gt_img, (256, 256))
        
        input_t = torch.from_numpy(input_img).float() / 255.0
        gt_t = torch.from_numpy(gt_img).float() / 255.0
        
        input_list.append(input_t)
        gt_list.append(gt_t)

ds = CustomDataset(input_list, gt_list)
dataloader = DataLoader(dataset = ds, batch_size = 16, shuffle = True)
unet = model.UNet().to(device)
unet.train()

learning_rate = 1e-7
optimizer = optim.Adam(unet.parameters(), lr = learning_rate)
num_epochs = 35
epsilon = 2e-7
loss_fn = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for input_img, gt_img in dataloader:

        input_img.to(device)                        # Input image not sent to cuda device!!!
        gt_img.to(device)
        
        output = unet(input_img)
        loss = loss_fn(output, gt_img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss / len(dataloader)}")
        

model_name = './output/needle_' + 'epoch_' + str(num_epochs) + '_' + 'lr_' + str(learning_rate) + '_' + 'ep_' + str(epsilon) + '.pth'

torch.save({
    'model_state_dict' : unet.state_dict(),
    'optimizer_state_dict' : optimizer.state_dict()
}, model_name)