import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class cVAEEncoder(nn.Module):
    def __init__(self, latent_dim = 2):
        super().__init__()
        
        self.label_emb = nn.Embedding(10, 10)
        
        self.convinx = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU()
        )
            
        self.convinc = nn.Sequential(
            nn.Conv2d(in_channels = 10, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU()
        )
            
        self.convin2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding = 1),
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
        
    def forward(self, x, labels):
        c = self.label_emb(labels)
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.shape[0], c.shape[1], 28, 28)
        
        x = self.convinx(x)
        c = self.convinc(c)
        x = torch.cat([x, c], dim=1)
        x = self.convin2(x)
        
        x = self.flat(x)
        x = self.convmid(x)
        
        z_mean = self.convout_mean(x)
        z_logvar = self.convout_logvar(x)
        
        return z_mean, z_logvar

class cVAEDecoder(nn.Module):
    def __init__(self, latent_dim = 2):
        super().__init__()
        
        self.label_emb = nn.Embedding(10, 10)
        
        self.linearinx = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
        )
        
        self.linearinc = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
        )
        
        self.linearin2 = nn.Sequential(
            nn.Linear(64, 64*14*14),
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
    
    def forward(self, x, labels):
        c = self.label_emb(labels)
        
        x = self.linearinx(x)
        c = self.linearinc(c)
        x = torch.cat([x, c], dim=1)
        
        x = self.linearin2(x)
        
        x = x.view(-1, 64, 14, 14)
        x = self.convout(x)
        
        return x
        
        
class cVAE(nn.Module):
    def __init__(self, latent_dim = 2):
        super().__init__()
        
        self.cvae_encoder = cVAEEncoder(latent_dim = latent_dim)
        self.cvae_decoder = cVAEDecoder(latent_dim = latent_dim)
        
        self.bce = nn.BCELoss(reduction = 'sum')
        
    def forward(self, x, labels):
        z_mean, z_logvar = self.cvae_encoder(x, labels)
        z = self.sampling(z_mean, z_logvar)
        x_hat = self.cvae_decoder(z, labels)
        return x_hat, z_mean, z_logvar
        
    def sampling(self, z_mean, z_logvar):
        std = torch.exp(z_logvar/2)
        eps = torch.randn_like(std)
        return z_mean + eps * std
    
    def loss_function(self, x, x_hat, z_mean, z_logvar):
        loss_bce = self.bce(x_hat, x) / x.shape[0]
        loss_kld = (-0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())) / x.shape[0]
        return loss_bce, loss_kld 
