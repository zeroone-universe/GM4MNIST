import os

import numpy as np
import torch
import torch.nn as nn
from torch.nn import BatchNorm2d
import torch.nn.functional as F


class MaskedConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.mask = torch.ones_like(self.conv.weight)
        
        self.mask[:, :, kernel_size//2, kernel_size//2+1:] = 0.
        
        for i in range(kernel_size//2+1, kernel_size):
            self.mask[:, :, i, :] = 0.
        self.conv.weight = nn.Parameter(self.conv.weight * self.mask)
        
    def forward(self, x):
        x = self.conv(x)
        return x
    
class PixelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            MaskedConvolution(1, 64, 7, 1, 3),
            BatchNorm2d(64),
            nn.ReLU(),
            
            MaskedConvolution(64, 64, 7, 1, 3),
            BatchNorm2d(64),
            nn.ReLU(),
            
            MaskedConvolution(64, 64, 3, 1, 1),
            BatchNorm2d(64),
            nn.ReLU(),
            
            MaskedConvolution(64, 64, 3, 1, 1),
            BatchNorm2d(64),
            nn.ReLU(),
            
            MaskedConvolution(64, 64, 1, 1, 0),
            BatchNorm2d(64),
            nn.ReLU(),
            
            MaskedConvolution(64, 64, 1, 1, 0),
            BatchNorm2d(64),
            nn.ReLU(),
        
            MaskedConvolution(64, 256, 1, 1, 0),
        )
        
    def forward(self, x):
        x = self.model(x)
        return x
    
    def loss_function(self, x_hat, x):
        loss = F.cross_entropy(x_hat, x)
        return loss