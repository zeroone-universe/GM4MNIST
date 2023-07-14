import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class VAEEncoder(nn.Module):
    def __init__(self, latent_dim = 2):
        super().__init__()
        self.convin = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
        )
        
        self.flat = nn.Flatten()
        self.convmid = nn.Linear(64*14*14, 32)
        
        self.convout_mean = nn.Linear(32, latent_dim)
        self.convout_logvar = nn.Linear(32, latent_dim)
        
    def forward(self, x):
        x = self.convin(x)
        x = self.flat(x)
        x = self.convmid(x)
        
        z_mean = self.convout_mean(x)
        z_logvar = self.convout_logvar(x)
        
        return z_mean, z_logvar

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim = 2):
        super().__init__()
        
        self.linearin = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64*14*14),
            nn.ReLU(),
        )
        
        self.convout = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 32, out_channels = 1, kernel_size = 3, stride = 1, padding = 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        print(x.shape)
        x = self.linearin(x)
        x = x.view(-1, 64, 14, 14)
        x = self.convout(x)
        
        return x
        
        
class VAE(nn.Module):
    def __init__(self, latent_dim = 2):
        super().__init__()
        
        self.vae_encoder = VAEEncoder(latent_dim = latent_dim)
        self.vae_decoder = VAEDecoder(latent_dim = latent_dim)
        
        self.bce = nn.BCELoss(reduction = 'sum')
        
    def forward(self, x):
        z_mean, z_logvar = self.vae_encoder(x)
        z = self.sampling(z_mean, z_logvar)
        x_hat = self.vae_decoder(z)
        return x_hat, z_mean, z_logvar
        
    def sampling(self, z_mean, z_logvar):
        std = torch.exp(z_logvar/2)
        eps = torch.randn_like(std)
        return z_mean + eps * std
    
    def loss_function(self, x, x_hat, z_mean, z_logvar):
        loss_bce = self.bce(x_hat, x) / x.shape[0]
        loss_kld = ( -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())) / x.shape[0]
        return loss_bce, loss_kld
