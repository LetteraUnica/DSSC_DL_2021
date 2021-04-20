import torch
from torch import nn

def build_list(a, n, name=""):
    if isinstance(a, list):
        assert len(a) == n, f"The list {name} isn't long {n}"
    return a*n

class MLP(nn.Module):
    def add_linear_layer(self, inputs, outputs, activation, batchnorm=True, bias=True):
        layer = []
        if batchnorm:
            layer.append(nn.BatchNorm1d(inputs))
        layer.append(nn.Linear(inputs, outputs, bias=bias))
        if activation is not None:
            layer.append(activation())

        self.layers.extend(layer)
        

    def __init__(self, nodes, activations = nn.ReLU, batchnorm=True, bias=True):
        super().__init__()
        self.layers = [nn.Flatten()]

        n_layers = len(nodes)
        activations = build_list(activations, n_layers-1, "activations")
        batchnorm = build_list(batchnorm, n_layers-1, "batchnorm")
        bias = build_list(bias, n_layers-1, "bias")
        
        # In the first layer I don't need batchnorm
        self.add_linear_layer(nodes[0], nodes[1], activations[0], False, bias[0])
        for i in range(1, n_layers-1):
            self.add_linear_layer(nodes[i], nodes[i+1], activation[i], batchnorm[i], bias[i])

        self.layers = nn.Sequential(*self.layers)

    def forward(self, X):
        self.layers(X)