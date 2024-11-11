import os
import torch
from model_framework.AlexNet import AlexNet
from torch.utils.data import DataLoader
from datasets.MNIST.MNIST_dataset import *


batch_size = 64
iterations = 20
learning_rate = 1e-5
use_cuda = torch.cuda.is_available()

train_set = MNIST_Dataset(is_train=True)
test_set = MNIST_Dataset(is_train=False)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size)

model = AlexNet(in_channels=1, out_features=10)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model_name = 'alexnet_mnist'


def train_loop(data_loader, model, loss_function, optimizer, use_cuda):
    size = len(data_loader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, Y) in enumerate(data_loader):
        if use_cuda:
            X, Y, model, loss_function = X.cuda(), Y.cuda(), model.cuda(), loss_function.cuda()
        # Compute prediction and loss
        pred = model(X)
        loss = loss_function(pred, Y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"\tloss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        


def test_loop(data_loader, model, loss_function, use_cuda, load_model=None):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    if load_model is not None:
        model.load_state_dict(torch.load(f'torch_models/{load_model}.pt', weights_only=True))
    model.eval()
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, Y in data_loader:
            if use_cuda:
                X, Y, model, loss_function = X.cuda(), Y.cuda(), model.cuda(), loss_function.cuda()
            pred = model(X)
            test_loss += loss_function(pred, Y).item()
            correct += (pred.argmax(1) == Y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    with torch.no_grad():
        dataset = data_loader.dataset
        input_X, input_Y = dataset[0]
        if use_cuda:
            input_X, input_Y, model = input_X.cuda(), input_Y.cuda(), model.cuda()
        torch.onnx.export(model, input_X.unsqueeze(0), f'onnx_models/{load_model}.onnx', opset_version=18, input_names=['input'], output_names=['output'])


def train_process(data_loader, model, iteration, loss_function, optimizer, use_cuda, save_model=None):
    for epoch in range(iteration):
        print(f'epoch {epoch}:')
        train_loop(data_loader, model, loss_function, optimizer, use_cuda)

    if save_model:
        model.eval()
        with torch.no_grad():
            torch.save(model.state_dict(), os.path.join(os.getcwd(), f'torch_models/{save_model}.pt'))


train_process(train_loader, model, iterations, loss_function, optimizer, use_cuda, model_name)
test_loop(test_loader, model, loss_function, use_cuda, model_name)