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

import matplotlib.pyplot as plt

input_dir = '/mnt/c/Users/MICS/Desktop/sample/opencv/vtest.avi'
# input_dir = '/mnt/c/Users/MICS/Desktop/needle_enhancement.avi'
input_data = image_preprocess.preprocess(input_dir)
ds = CustomDataset(input_data)
dataloader = DataLoader(dataset = ds, batch_size = 1, shuffle = False)

unet = model.UNet()

# checkpoint = torch.load('./output/epoch_35_lr_1e-07_ep_2e-07.pth')
# checkpoint = torch.load('./output/epoch_40_lr_1e-08_ep_2e-07.pth')
# checkpoint = torch.load('./output/trained_model_35epoch.pth')
checkpoint = torch.load('./output/epoch_35_lr_1e-07_ep_2e-07.pth')

unet.load_state_dict(checkpoint['model_state_dict'])

unet.eval()

contrast_factor = 1.0
index = 0


with torch.no_grad():
    for data in dataloader:
        batch = unet(data)
        
        prev_frame = data[0][0].numpy()
        current_frame = data[0][1].numpy()
        
        # plt.imshow(prev_frame, cmap = 'gray')
        # plt.savefig('./frame1.png')
        # plt.imsave("./frame1.png", prev_frame, cmap = 'gray')
        
        # plt.imshow(current_frame, cmap = 'gray')
        # plt.savefig('./frame2.png')
        # plt.imsave('./frame2.png', current_frame, cmap = 'gray')
        
        output = batch[0]
        # u = output[0] * 255.0
        # v = output[1] * 255.0
        
        u = output[0]
        v = output[1]
        
        result = torch.pow(u, 2) + torch.pow(v, 2)
        result = torch.pow(result, 1.5)
        # plt.imshow(result, cmap = 'gray')
        # plt.savefig('./flow.png')
        plt.imsave('./video/flow_' + str(index) + '.png', result, cmap = 'gray')
        
        index += 1