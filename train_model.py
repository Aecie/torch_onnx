import os
import onnx
import torch
from torch.utils.data import DataLoader
from model_configs.train_infer import config


if not os.path.exists(config['onnx_model_space']):
    os.mkdir(config['onnx_model_space'])
if not os.path.exists(config['torch_model_space']):
    os.mkdir(config['torch_model_space'])


class ModelTrainer():
    def __init__(self, train_config):
        self.model_name = train_config['model_name']
        self.batch_size = train_config['batch_size']
        self.iterations = train_config['iterations']
        self.learning_rate = train_config['learning_rate']
        self.use_cuda = train_config['use_cuda']

        self.train_set = train_config['train_set']
        self.test_set = train_config['test_set']
        self.train_loader = DataLoader(dataset=self.train_set, batch_size=self.batch_size)
        self.test_loader = DataLoader(dataset=self.test_set, batch_size=self.batch_size)

        self.model = train_config['model']
        self.loss_function = train_config['loss_function']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)        

        if self.use_cuda:
            self.model = self.model.cuda()
            self.loss_function = self.loss_function.cuda()

    def proceed(self):
        for epoch in range(self.iterations):
            print(f'epoch {epoch}:')
            self.train_process()

        if self.model_name:
            self.model.eval()
            with torch.no_grad():
                torch.save(self.model.state_dict(), f'{config["torch_model_space"]}/{self.model_name}.pt')
        self.test_process(load_model=self.model_name)

    def train_process(self):
        size = len(self.train_loader.dataset)
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        self.model.train()
        for batch, (X, Y) in enumerate(self.train_loader):
            if self.use_cuda:
                X, Y = X.cuda(), Y.cuda()
            # Compute prediction and loss
            pred = self.model(X)
            loss = self.loss_function(pred, Y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * self.batch_size + len(X)
                print(f"\tloss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    def test_process(self, load_model=None):
        # Set the model to evaluation mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        if load_model is not None:
            self.model.load_state_dict(torch.load(f'{config["torch_model_space"]}/{load_model}.pt', weights_only=True))
        self.model.eval()
        size = len(self.test_loader.dataset)
        num_batches = len(self.test_loader)
        test_loss, correct = 0, 0

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for X, Y in self.test_loader:
                if self.use_cuda:
                    X, Y = X.cuda(), Y.cuda()
                pred = self.model(X)
                test_loss += self.loss_function(pred, Y).item()
                correct += (pred.argmax(1) == Y.argmax(1)).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        
        # save the model
        with torch.no_grad():
            dataset = self.test_loader.dataset
            input_X, input_Y = dataset[0]
            if self.use_cuda:
                input_X, input_Y, self.model = input_X.cuda(), input_Y.cuda(), self.model.cuda()
            torch.onnx.export(self.model, input_X.unsqueeze(0), f'{config["onnx_model_space"]}/{load_model}.onnx', opset_version=18, input_names=['input'], output_names=['output'])


trainer = ModelTrainer(train_config=config['AlexNet_MNIST'])
trainer.proceed()