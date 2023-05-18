import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision

from models.PixelCNN.model import PixelCNN

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
        self.model = PixelCNN()
        
        self.lr = self.hparams.optim.lr
        self.loss_function = self.model.loss_function
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
    

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        
        target = (imgs[:,0,:,:]*255).type(torch.LongTensor).to(self.device)

        optimizer = self.optimizers()
        imgs_hat = self.forward(imgs)
        
        loss = self.loss_function(imgs_hat,target)
        
        self.log("train/loss", loss, prog_bar=True)
        
        self.manual_backward(loss)
        optimizer.step()
        optimizer.zero_grad()
         
    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        
        target = (imgs[:,0,:,:]*255).type(torch.LongTensor).to(self.device)
        imgs_hat = self.forward(imgs)
        
        loss = self.loss_function(imgs_hat,target)

        self.log("val/loss", loss)

    def on_validation_end(self):

        samples = torch.zeros(10,1,28,28).to(self.device)
        
        for row in range(28):
            for column in range(28):
                probs = self.forward(samples)
                probs = F.softmax(probs[:,:,row,column], dim=-1).data
                samples[:,:, row,column] = torch.multinomial(probs, 1).float() / 255.0

        grid = torchvision.utils.make_grid(samples, nrow=5)
        self.logger.experiment.add_image("sample_images", grid, self.current_epoch)