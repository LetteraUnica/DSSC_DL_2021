import torch
from torch import nn

class MLP(nn.Module):
    def build_linear_layer(self, inputs, outputs, activation, batchnorm=True, bias=True):
        layers = []
        if batchnorm:
            layers.append(nn.BatchNorm1d(inputs))
        layers.append(nn.Linear(inputs, outputs, bias=bias))
        layers.append(activation())

        return layers

    def __init__(self, nodes, activation): #aggiungere dopo tipi di nodi
        super().__init__()
    







class MLP(nn.Module):
    """
    Implements a neural network to train on MNIST 
    """
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(784, 384),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(384),
            torch.nn.Linear(384, 384),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(384),
            torch.nn.Linear(384, 384),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(384),
            torch.nn.Linear(384, 10)
        )
        
    def forward(self, X):        
        return self.layers(X)