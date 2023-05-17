import torch
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl

from torchvision import datasets, transforms

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, hparams, args):
        self.data_dir = hparams.datamodule.data_dir
        self.batch_size = hparams.datamodule.batch_size
        
        self.args = args
        
        if hparams.datamodule.resize32:
            self.transform = [transforms.Resize((32,32)), transforms.ToTensor(),]
                
        else:
            self.transform = [transforms.ToTensor(),]
            
        if hparams.datamodule.scaling:
            self.transform.append(transforms.Normalize((0.5,), (0.5,)))
            
        self.transform = transforms.Compose(self.transform)
        
        self.prepare_data_per_node = True
        self.allow_zero_length_dataloader_with_multiple_devices = False
        
        self._log_hyperparams = False
        
    def prepare_data(self):
        # download
        if self.args.dataset == "mnist":
            datasets.MNIST(self.data_dir, train=True, download=True)
            datasets.MNIST(self.data_dir, train=False, download=True)
        elif self.args.dataset== "fmnist":
            datasets.FashionMNIST(self.data_dir, train=True, download=True)
            datasets.FashionMNIST(self.data_dir, train=False, download=True)
        
    def setup(self, stage = None):
        if self.args.dataset == "mnist":
            self.dataset_train = datasets.MNIST(self.data_dir, train = True, transform = self.transform)
            self.dataset_val = datasets.MNIST(self.data_dir, train=False, transform = self.transform)
        elif self.args.dataset == "fmnist":
            self.dataset_train = datasets.FashionMNIST(self.data_dir, train = True, transform = self.transform)
            self.dataset_val = datasets.FashionMNIST(self.data_dir, train=False, transform = self.transform)
            
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size = self.batch_size, shuffle = True, num_workers= 16)
    
    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size = self.batch_size, shuffle = False, num_workers= 16)
    