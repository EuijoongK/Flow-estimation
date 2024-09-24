import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import *

__all__ = [
    'UNet'
]

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.sz = 64
        
        #Contracting Path
        # self.enc1_1 = CBR2d(in_channels = 2, out_channels = self.sz)
        
        self.enc1_1 = CBR2d(in_channels = 3, out_channels = self.sz)
        self.enc1_2 = CBR2d(in_channels = self.sz, out_channels = self.sz)
        
        self.pool1 = nn.MaxPool2d(kernel_size = 2)
        
        self.enc2_1 = CBR2d(in_channels = self.sz, out_channels = self.sz * 2)
        self.enc2_2 = CBR2d(in_channels = self.sz * 2, out_channels = self.sz * 2)
        
        self.pool2 = nn.MaxPool2d(kernel_size = 2)
        
        self.enc3_1 = CBR2d(in_channels = self.sz * 2, out_channels = self.sz * 4)
        self.enc3_2 = CBR2d(in_channels = self.sz * 4, out_channels = self.sz * 4)
        
        self.pool3 = nn.MaxPool2d(kernel_size = 2)
        
        self.enc4_1 = CBR2d(in_channels = self.sz * 4, out_channels = self.sz * 8)
        self.enc4_2 = CBR2d(in_channels = self.sz * 8, out_channels = self.sz * 8)
        
        self.pool4 = nn.MaxPool2d(kernel_size = 2)
        
        self.enc5_1 = CBR2d(in_channels = self.sz * 8, out_channels = self.sz * 16)

        
        #Expansive Path
        self.dec5_1 = CBR2d(in_channels = self.sz * 16, out_channels = self.sz * 8)
        self.unpool4 = nn.ConvTranspose2d(in_channels = self.sz * 8, out_channels = self.sz * 8,
                                          kernel_size = 4, stride = 2, padding = 1, bias = True)
        
        self.dec4_2 = CBR2d(in_channels = self.sz * 16, out_channels = self.sz * 8)
        self.dec4_1 = CBR2d(in_channels = self.sz * 8, out_channels = self.sz * 4)
        
        self.unpool3 = nn.ConvTranspose2d(in_channels = self.sz * 4, out_channels = self.sz * 4,
                                          kernel_size = 4, stride = 2, padding = 1, bias = True)
        
        self.dec3_2 = CBR2d(in_channels = self.sz * 8, out_channels = self.sz * 4)
        self.dec3_1 = CBR2d(in_channels = self.sz * 4, out_channels = self.sz * 2)
        
        self.unpool2 = nn.ConvTranspose2d(in_channels = self.sz * 2, out_channels = self.sz * 2,
                                          kernel_size = 4, stride = 2, padding = 1, bias = True)
        
        self.dec2_2 = CBR2d(in_channels = self.sz * 4, out_channels = self.sz * 2)
        self.dec2_1 = CBR2d(in_channels = self.sz * 2, out_channels = self.sz)
        
        self.unpool1 = nn.ConvTranspose2d(in_channels = self.sz, out_channels = self.sz,
                                          kernel_size = 4, stride = 2, padding = 1, bias = True)
        
        self.dec1_2 = CBR2d(in_channels = self.sz * 2, out_channels = self.sz)
        self.dec1_1 = CBR2d(in_channels = self.sz, out_channels = self.sz)
        
        # self.flow_cal = CBR2d(in_channels = self.sz, out_channels = 2)
        self.flow_cal = CBR2d(in_channels = self.sz, out_channels = 3)
        
        
    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)
        
        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)
        
        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)
        
        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)
        
        enc5_1 = self.enc5_1(pool4)
        
        
        dec5_1 = self.dec5_1(enc5_1)
        unpool4 = self.unpool4(dec5_1)
        
        cat4 = torch.cat((unpool4, enc4_2), dim = 1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)
        unpool3 = self.unpool3(dec4_1)
        
        cat3 = torch.cat((unpool3, enc3_2), dim = 1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        unpool2 = self.unpool2(dec3_1)
        
        cat2 = torch.cat((unpool2, enc2_2), dim = 1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        unpool1 = self.unpool1(dec2_1)
        
        cat1 = torch.cat((unpool1, enc1_2), dim = 1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)
        
        return self.flow_cal(dec1_1)