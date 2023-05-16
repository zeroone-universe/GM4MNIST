import os

import numpy as np
import torch
import torch.nn as nn

import os

import numpy as np
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        
        self.label_embed = nn.Embedding(10, 10)
        
        self.CTin_z = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 100, out_channels = 256, kernel_size = 4, stride = 1, padding = 0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        self.CTin_c = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 10, out_channels = 256, kernel_size = 4, stride = 1, padding = 0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        self.CTout = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, 4, 2, 1),
            nn.Tanh(),
        )
        

    def forward(self, z, labels):
        c = self.label_embed(labels)
        c = c.unsqueeze(2).unsqueeze(3)
        
        z = z.unsqueeze(2).unsqueeze(3)
        
        x = self.CTin_z(z)
        c = self.CTin_c(c)
        
        x = torch.cat([x, c], dim=1)
        
        img = self.CTout(x)
        
        return img
    


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        self.label_embed = nn.Embedding(10, 10)

        self.convin_img = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        
        self.convin_c = nn.Sequential(
            nn.Conv2d(10, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )

        self.convout = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid(),
        )
            
            
    def forward(self, img, labels):
        c = self.label_embed(labels)
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.shape[0], c.shape[1], 32, 32)
        
        x = self.convin_img(img)
        c = self.convin_c(c)
        
        x = torch.cat([x, c], dim=1)
        validity = self.convout(x)
        validity = validity.view(validity.shape[0], -1)
        
        return validity