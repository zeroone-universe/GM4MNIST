import os
import argparse
import importlib.util

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from datamodule import MNISTDataModule

from omegaconf import OmegaConf as OC
from utils import *


def main(args):
    hparams = OC.load(f"models/{args.model}/hparams.yaml")
    # Train 클래스 사용
    train_path = f"models/{args.model}/train.py"
    spec = importlib.util.spec_from_file_location("train_module", train_path)
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    Train = getattr(train_module, "Train")
    
    pl.seed_everything(hparams.default.random_seed, workers=True)
    
    check_dir_exist(hparams.train.logger_path)
    tb_logger = pl_loggers.TensorBoardLogger(hparams.train.logger_path, name=f"{args.model}_logs")
    
    train = Train(hparams)
    mnist_datamodule = MNISTDataModule(hparams)
    
    trainer=pl.Trainer(devices=hparams.train.devices, accelerator="gpu",
        max_epochs=hparams.train.max_epochs,
        logger=tb_logger
    )
    
    trainer.fit(train, mnist_datamodule)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, required = True, help = "Enter model name (cGAN, CVAE, ...)")
    args = parser.parse_args()
    
    main(args)