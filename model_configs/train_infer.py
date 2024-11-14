import os
import torch
from model_framework.AlexNet import AlexNet
from datasets.MNIST.MNIST_dataset import *


config = {
    'workspace': os.getcwd(),
    'onnx_model_space': os.path.join(os.getcwd(), '../onnx_models'),
    'torch_model_space': os.path.join(os.getcwd(), '../torch_models'),
    'AlexNet_MNIST': {
        'model_name': 'alexnet_mnist',
        'use_cuda': torch.cuda.is_available(),
        'batch_size': 64,
        'iterations': 20,
        'learning_rate': 1e-5,
        'train_set': MNIST_Dataset(is_train=True),
        'test_set': MNIST_Dataset(is_train=False),
        'model': AlexNet(in_channels=1, out_features=10),
        'loss_function': torch.nn.MSELoss()
    }  
}
