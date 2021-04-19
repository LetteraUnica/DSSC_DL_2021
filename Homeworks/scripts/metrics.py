import torch


def misclass_rate(output: torch.Tensor, target: torch.Tensor):
    """
    Computes the misclassification rate of two tensors
    """
    misclassified = torch.sum(torch.max(output, 1).indices != target)
    return misclassified/output.size()[0]

def accuracy(output: torch.Tensor, target: torch.Tensor):
    """
    Computes the accuracy measure of two tensors
    """
    return 1 - misclass_rate(output, target)