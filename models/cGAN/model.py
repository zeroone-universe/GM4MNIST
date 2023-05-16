import os

import numpy as np
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape
        self.label_embed = nn.Embedding(10, 10)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim+10, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        c = self.label_embed(labels)
        z = torch.cat([z, c], dim=1)
        
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img
    


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        self.label_embed = nn.Embedding(10, 10)

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape))+10, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        c = self.label_embed(labels)
        x = torch.cat([img_flat, c], dim=1)
        validity = self.model(x)
        return validity