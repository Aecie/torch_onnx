import os
import torch
import struct
import numpy as np
from array import array
from torchvision import transforms
from torch.utils.data import Dataset


class MNIST_Dataset(Dataset):
    def __init__(self, is_train: bool=True, image_transform=None, label_transform=None):
        super().__init__()
        self.is_train = is_train
        self.__one_hot_encoder__ = np.eye(10)
        self.__train_image_path__ = os.path.join(os.getcwd(), 'datasets/MNIST/train-images.idx3-ubyte')
        self.__train_label_path__ = os.path.join(os.getcwd(), 'datasets/MNIST/train-labels.idx1-ubyte')
        self.__test_image_path__ = os.path.join(os.getcwd(), 'datasets/MNIST/t10k-images.idx3-ubyte')
        self.__test_label_path__ = os.path.join(os.getcwd(), 'datasets/MNIST/t10k-labels.idx1-ubyte')

        if self.is_train:
            self.__X__, self.__Y__ =  self.__read_image_label__(self.__train_image_path__, self.__train_label_path__)
        else:
            self.__X__, self.__Y__ = self.__read_image_label__(self.__test_image_path__, self.__test_label_path__)
        self.__X__ = torch.nn.functional.interpolate(self.__X__, (128, 128))

    def __len__(self):
        return self.__X__.shape[0]

    def __getitem__(self, index):
        X = self.__X__[index]
        Y = self.__Y__[index]
        return X, Y

    def __read_image_label__(self, image_path, label_path):
        labels = []
        with open(label_path, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        labels = [self.__one_hot_encoder__[label] for label in labels]

        with open(image_path, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols: (i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img
        
        return torch.from_numpy(np.array(images, dtype=np.float32).reshape((-1, 1, 28, 28))), torch.from_numpy(np.array(labels, dtype=np.float32).reshape((-1, 10))*1.)


