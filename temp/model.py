import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.util import *

class CustomModel(nn.Module):
    
    def __init__(self, batchNorm):
        super(CustomModel, self).__init__()
        self.batchNorm = batchNorm
        
        self.conv1 = conv(batchNorm, 2, 256, kernel_size = 7, stride = 1)
        self.conv2 = conv(batchNorm, 256, 256, kernel_size = 5, stride = 1)
        self.conv3 = conv(batchNorm, 256, 512, kernel_size = 5, stride = 1)
        self.conv4 = conv(batchNorm, 512, 512, kernel_size = 3, stride = 1)
        self.conv5 = conv(batchNorm, 512, 512, kernel_size = 3, stride = 1)
        self.conv6 = conv(batchNorm, 512, 512, kernel_size = 3, stride = 1)
        self.conv7 = conv(batchNorm, 512, 1024, kernel_size = 3, stride = 1)
        self.conv8 = conv(batchNorm, 1024, 512, kernel_size = 1, stride = 1)
        self.conv9 = conv(batchNorm, 512, 256, kernel_size = 5, stride = 1)
        self.conv10 = conv(batchNorm, 256, 128, kernel_size = 5, stride = 1)
        self.conv11 = conv(batchNorm, 128, 64, kernel_size = 5, stride = 1)
        self.conv12 = conv(batchNorm, 64, 2, kernel_size = 5, stride = 1)
        
        