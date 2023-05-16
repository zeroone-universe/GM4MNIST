import os
from typing import Any, Optional

import pytorch_lightning as L
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn.functional as F
import torchvision

from models.cGAN.model import Generator, Discriminator

class Train(L.LightningModule):
    def __init__(
        self,
        hparams,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.automatic_optimization = False

        self.image_shape = (1, 28, 28)
        self.latent_dim = self.hparams.model.latent_dim
        
        # networks
        self.generator = Generator(latent_dim=self.latent_dim, img_shape=self.image_shape)
        self.discriminator = Discriminator(img_shape= self.image_shape)

        self.loss= torch.nn.BCELoss()

    def forward(self, z, labels):
        return self.generator(z, labels)
    
    def configure_optimizers(self):
        lr = self.hparams.optim.lr
        b1 = self.hparams.optim.b1
        b2 = self.hparams.optim.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        optimizer_g, optimizer_d = self.optimizers()

        # sample noise
        z = torch.randn(imgs.shape[0], self.latent_dim)
        z = z.type_as(imgs)

        #valid/fake index
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)
        
        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)   

        # train generator
        # generate images
        self.toggle_optimizer(optimizer_g)

        # adversarial loss is binary cross-entropy
        g_loss = self.loss(self.discriminator(self.forward(z, labels), labels), valid)
        self.log("train/g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        self.toggle_optimizer(optimizer_d)

        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        real_loss = self.loss(self.discriminator(imgs, labels), valid)

        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)

        fake_loss = self.loss(self.discriminator(self.forward(z, labels).detach(), labels), fake)

        d_loss = (real_loss + fake_loss) / 2
        self.log("train/d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        
        self.untoggle_optimizer(optimizer_d)

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch

        # sample noise
        z = torch.randn(imgs.shape[0], self.latent_dim)
        z = z.type_as(imgs)

        #valid/fake index
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)
        
        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)   

        # adversarial loss is binary cross-entropy
        g_loss = self.loss(self.discriminator(self.forward(z, labels), labels), valid)
        self.log("val/g_loss", g_loss)

        real_loss = self.loss(self.discriminator(imgs, labels), valid)
        fake_loss = self.loss(self.discriminator(self.forward(z, labels).detach(), labels), fake)

        d_loss = (real_loss + fake_loss) / 2
        self.log("val/d_loss", d_loss)

    def on_validation_end(self):
        #sample 이미지 로깅
        z = torch.randn(10, self.latent_dim)
        z = z.type_as(self.generator.model[0].weight)
        
        labels = torch.tensor([0,1,2,3,4,5,6,7,8,9], device = z.device)
        
        sample_imgs = self.forward(z, labels)
        sample_imgs = (0.5 * sample_imgs + 0.5)
        
        grid = torchvision.utils.make_grid(sample_imgs, nrow=5)
        self.logger.experiment.add_image("sample_images", grid, self.current_epoch)