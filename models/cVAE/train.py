import os
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision

from models.cVAE.model import cVAE

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
        self.model = cVAE(latent_dim=self.latent_dim)
        
        self.lr = self.hparams.optim.lr
        self.loss=  self.model.loss_function
        
    def forward(self, x, labels):
        return self.model(x, labels)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
    

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        optimizer = self.optimizers()

        imgs_hat, mean, logvar = self.forward(imgs, labels)
        loss_bce, loss_kld = self.loss(x = imgs, x_hat = imgs_hat, z_mean = mean, z_logvar = logvar)
        
        loss = loss_bce + loss_kld
        
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/loss_bce", loss_bce, prog_bar=True)
        self.log("train/loss_kld", loss_kld, prog_bar=True)
        
        self.manual_backward(loss)
        optimizer.step()
        optimizer.zero_grad()
         
    def validation_step(self, batch, batch_idx):
        imgs, labels = batch

        imgs_hat, mean, logvar = self.forward(imgs, labels)
        
        loss_bce, loss_kld = self.loss(x = imgs, x_hat = imgs_hat, z_mean = mean, z_logvar = logvar)
  
        loss = loss_bce + loss_kld
        
        self.log("val/loss", loss)
        self.log("val/loss_bce", loss_bce)
        self.log("val/loss_kld", loss_kld)
        
    def on_validation_end(self):
        #sample 이미지 로깅
        z = torch.randn(10, self.latent_dim)
        z = z.type_as(self.model.cvae_encoder.convinx[0].weight)
        
        labels = torch.tensor([0,1,2,3,4,5,6,7,8,9], device = z.device)
        
        sample_imgs = self.model.cvae_decoder(z, labels)
        
        grid = torchvision.utils.make_grid(sample_imgs, nrow=5)
        self.logger.experiment.add_image("val/sample_images", grid, self.current_epoch)
    
    
    # def plot_latent_space(self):
    #     #from https://taeu.github.io/paper/deeplearning-paper-vae/
    
    #     n = 20
    #     figure = torch.zeros((28 * n, 28 * n))
    #     grid_x = torch.tensor(norm.ppf(torch.linspace(0.05, 0.95, n)))
    #     grid_y = torch.tensor(norm.ppf(torch.linspace(0.05, 0.95, n)))

    #     for i, yi in enumerate(grid_x):
    #         for j, xi in enumerate(grid_y):
    #             z_sample = torch.tensor([[xi, yi]])
    #             z_sample = z_sample.repeat(16, 1)
    #             # 예측할 모델에 텐서 전달하고 예측 결과 얻는 과정을 수정해야 합니다.
    #             x_hat = self.model.vae_decoder(z_sample.type_as(self.model.vae_encoder.convin[0].weight))
                
    #             digit = x_hat[0].reshape(28, 28)
    #             figure[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = digit

    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(figure, cmap='Greys_r')
    #     plt.show()
        
    #     self.logger.experiment.add_figure("val/image_grid", plt.gcf(), self.current_epoch)
        
