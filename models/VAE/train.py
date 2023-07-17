import os
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision

from models.VAE.model import VAE

import matplotlib.pyplot as plt
from scipy.stats import norm

class Train(pl.LightningModule):
    def __init__(
        self,
        hparams,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.automatic_optimization = False

        self.latent_dim = self.hparams.model.latent_dim
        
        # networks
        self.model = VAE(latent_dim=self.latent_dim)
        
        self.lr = self.hparams.optim.lr
        self.loss=  self.model.loss_function
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
    

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        optimizer = self.optimizers()

        imgs_hat, mean, logvar = self.forward(imgs)
        loss_bce, loss_kld = self.loss(x = imgs, x_hat = imgs_hat, z_mean = mean, z_logvar = logvar)
        
        loss = loss_bce + loss_kld
        
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/loss_bce", loss_bce, prog_bar=True)
        self.log("train/loss_kld", loss_kld, prog_bar=True)
        
        self.manual_backward(loss)
        optimizer.step()
        optimizer.zero_grad()
         
    def validation_step(self, batch, batch_idx):
        imgs, _ = batch

        imgs_hat, mean, logvar = self.forward(imgs)
        
        loss_bce, loss_kld = self.loss(x = imgs, x_hat = imgs_hat, z_mean = mean, z_logvar = logvar)
  
        
        loss = loss_bce + loss_kld
        
        self.log("val/loss", loss)
        self.log("val/loss_bce", loss_bce)
        self.log("val/loss_kld", loss_kld)
        
    def on_validation_end(self):
        #sample 이미지 로깅
        z = torch.randn(10, self.latent_dim)
        z = z.to(self.device)
        sample_imgs = self.model.vae_decoder(z)
        
        grid = torchvision.utils.make_grid(sample_imgs, nrow=5)
        self.logger.experiment.add_image("val/sample_images", grid, self.current_epoch)
        
        #latent space 로깅
        if self.current_epoch % 10 == 0:
            self.plot_latent_space()
        
    def plot_latent_space(self):
        #from https://taeu.github.io/paper/deeplearning-paper-vae/
    
        n = 20
        figure = torch.zeros((28 * n, 28 * n))
        grid_x = torch.tensor(norm.ppf(torch.linspace(0.05, 0.95, n)))
        grid_y = torch.tensor(norm.ppf(torch.linspace(0.05, 0.95, n)))

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = torch.tensor([[xi, yi]])
                z_sample = z_sample.repeat(16, 1)
                x_hat = self.model.vae_decoder(z_sample.to(self.device))
                
                digit = x_hat[0].reshape(28, 28)
                figure[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='Greys_r')
        plt.show()
        
        self.logger.experiment.add_figure("val/image_grid", plt.gcf(), self.current_epoch)
        
