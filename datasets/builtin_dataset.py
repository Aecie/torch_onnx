import os
from torchvision.datasets import MNIST


def MNIST_dataset(data_transform=None, label_transform=None):
    train_set = MNIST(
        root=os.path.join(os.getcwd(), 'datasets'),
        train=True,
        transform=data_transform,
        target_transform=label_transform,
        download=True,
    )
    
    test_set = MNIST(
        root=os.path.join(os.getcwd(), 'datasets'),
        train=False,
        transform=data_transform,
        target_transform=label_transform,
        download=True,
    )
    
    return train_set, test_set