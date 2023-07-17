import os
import argparse
import importlib.util

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from datamodule import MNISTDataModule

from omegaconf import OmegaConf as OC
from utils import *


def main(args):
    hparams = OC.load(f"hparams.yaml")[args.model]
    
    # Train 클래스 사용
    train_path = f"models/{args.model}/train.py"
    spec = importlib.util.spec_from_file_location("train_module", train_path)
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    Train = getattr(train_module, "Train")
    
    pl.seed_everything(args.random_seed, workers=True)
    
    check_dir_exist(args.logger_path)
    tb_logger = pl_loggers.TensorBoardLogger(args.logger_path, name=f"{args.model}_logs")
    
    train = Train(hparams)
    mnist_datamodule = MNISTDataModule(hparams, args)
    
    trainer=pl.Trainer(devices=args.devices, accelerator="gpu",
        max_epochs=hparams.train.max_epochs,
        logger=tb_logger
    )
    
    trainer.fit(train, mnist_datamodule)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type = int, default = 0b011011, help = "Enter random seed (default: 0b011011)")
    
    parser.add_argument("-m", "--model", type = str, required = True, help = "Enter model name (cGAN, CVAE, ...)")
    parser.add_argument("-t", "--type_dataset", type = str, default = "mnist", help = "mnist or fmnist")
    parser.add_argument("-p", "--path_dataset", type = str, required = True, help = "Enter path to dataset")
    parser.add_argument("-d", "--devices", type = list, default = [0], help = "Enter GPU number ([0], [1], [0,1], ...)")
                 
    parser.add_argument("--logger_path", type = str, default = "./logger", help = "Enter logger path (default: ./logger)")       
    args = parser.parse_args()
    
    main(args)