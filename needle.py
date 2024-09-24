import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim

import cv2

import model
from dataset import image_preprocess, CustomDataset
from loss_func import CustomFlowLoss

import matplotlib.pyplot as plt

input_dir = '/mnt/c/Users/MICS/Desktop/needle_enhancement.avi'
input_data = image_preprocess.preprocess(input_dir)
ds = CustomDataset(input_data)
dataloader = DataLoader(dataset = ds, batch_size = 16, shuffle = True)

unet = model.UNet()

checkpoint = torch.load('./output/epoch_35_lr_1e-07_ep_2e-07.pth')
# checkpoint = torch.load('./output/trained_model_35epoch.pth')
# checkpoint = torch.load('./output/trained_model_35epoch_1e-7lr_1e-7epsilon.pth')

unet.load_state_dict(checkpoint['model_state_dict'])

unet.eval()

contrast_factor = 1.0


with torch.no_grad():
    for data in dataloader:
        
        # print(type(data))
        
        batch = unet(data)
        
        prev_frame = data[0][0].numpy()
        current_frame = data[0][1].numpy()
        
        # plt.imshow(prev_frame, cmap = 'gray')
        # plt.savefig('./frame1.png')
        plt.imsave("./needle_frame1.png", prev_frame, cmap = 'gray')
        
        # plt.imshow(current_frame, cmap = 'gray')
        # plt.savefig('./frame2.png')
        plt.imsave('./needle_frame2.png', current_frame, cmap = 'gray')
        
        output = batch[0]
        u = output[0] * 50
        v = output[1] * 50
        
        print(u)
        
        height = u.shape[0]
        width = u.shape[1]
        
        result = np.zeros((height, width), dtype = np.uint8)
        
        for i in range(height):
            for j in range(width):
                result = cv2.arrowedLine(result, (j, i), (j + int(v[i][j]),
                                                          i + int(u[i][j])), 255, 2)
        
        # result = torch.pow(u, 2) + torch.pow(v, 2)
        # plt.imshow(result, cmap = 'gray')
        # plt.savefig('./flow.png')
        plt.imsave('./needle_flow.png', result, cmap = 'gray')
        
        break