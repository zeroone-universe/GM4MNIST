# Generative Models with MNIST and FMNIST

This repository provides simple implementations of various generative models using MNIST and FMNIST datasets. The models are implemented using PyTorch Lightning, a lightweight PyTorch wrapper for high-level training.

## Models

The following generative models are implemented, or planned to be implemented:

- Variational Autoencoder (VAE)
- Conditional Variational Autoencoder (cVAE)
- Conditional Generative Adversarial Network (cGAN)
- Conditional Deep Convolutional Generative Adversarial Network (cDCGAN)
- Adversarial Autoencoder (WIP)
- DDPM (WIP)

## Dataset

The MNIST (Modified National Institute of Standards and Technology) and FMNIST (Fashion MNIST) datasets are used for training and validation. MNIST consists of grayscale images of handwritten digits, while FMNIST contains grayscale images of various fashion items.

This code will download the dataset automatically while running.

## Requirements
 
To run this code, you will need:

- matplotlib==3.7.1
- numpy==1.23.5
- omegaconf==2.3.0
- pytorch_lightning==2.0.0
- scipy==1.10.1
- torch==2.0.0
- torchvision==0.15.0
- tensorboard


To automatically install these libraries, run the following command:

```pip install -r requirements.txt```

Alternatively, you can utilize the included Dockerfile.

## Usage

To run the code on your own machine, follow these steps:

1. Open the hparams.yaml. Modify the path for datasets, training configurations, and hyperparameters as needed.
2. Run the 'main.py' file to start training the model.
```python main.py -m <abbreviated name of model to use> -t <"mnist" or "fmnist"> -p <dataset path> -d <devices>```

## Notes
- This repository is intended for practice purposes, and there are no plans to implement inference files for the generative models. If you would like to see the output of the models, please refer to the TensorBoard log files.

