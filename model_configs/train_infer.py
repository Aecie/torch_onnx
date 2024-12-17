import os
import sys
import torch
import torchvision
import numpy as np

sys.path.append(os.getcwd())
from model_framework.AlexNet import AlexNet
from datasets.builtin_dataset import *


config = {
    'workspace': os.getcwd(),
    'dataset_space': os.path.join(os.getcwd(), 'datasets'),  # store the script to load dataset
    'onnx_model_space': os.path.join(os.getcwd(), '../onnx_models'),
    'torch_model_space': os.path.join(os.getcwd(), '../torch_models'),
    'onnx_engine_space': os.path.join(os.getcwd(), '../onnx_engines'),
    'AlexNet_MNIST': {
        'model_name': 'alexnet_mnist',
        'use_cuda': torch.cuda.is_available(),
        'batch_size': 64,
        'iterations': 20,
        'learning_rate': 1e-5,
        'dataset': MNIST_dataset(
            data_transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(128, 128)),
                torchvision.transforms.ToTensor()
            ]),
            label_transform=torchvision.transforms.Compose([
                lambda x: torch.eye(10)[x]
            ])
        ),
        'model': AlexNet(in_channels=1, out_features=10),
        'loss_function': torch.nn.MSELoss(),
        'input_shape': (1, 1, 128, 128),
        'output_shape': (1, 10),
    }  
}
