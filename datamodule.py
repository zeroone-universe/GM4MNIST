import torch
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl

from torchvision import datasets, transforms

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        self.data_dir = hparams.datamodule.data_dir
        self.batch_size = hparams.datamodule.batch_size
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        self.prepare_data_per_node = True
        self.allow_zero_length_dataloader_with_multiple_devices = False
        
        self._log_hyperparams = False
        
    def prepare_data(self):
        # download
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)
        
    def setup(self, stage = None):
        self.mnist_train = datasets.MNIST(self.data_dir, train = True, transform = self.transform)
        self.mnist_val = datasets.MNIST(self.data_dir, train=False, transform = self.transform)
            
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size = self.batch_size, shuffle = True, num_workers= 16)
    
    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size = self.batch_size, shuffle = False, num_workers= 16)
    