import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(batchNorm, in_planes, out_planes, kernel_size = 3, stride = 1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, padding = (kernel_size - 1) // 2, bias = False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace = True)
        )
    else:
        return nn.Sequential([
            nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, padding = (kernel_size - 1) // 2, bias = True),
            nn.LeakyReLU(0.1, inplace = True)
        ])
        
