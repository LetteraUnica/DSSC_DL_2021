import torch

class metric:
    def __init__(self, dataset, score, name=None, steps_to_update=1):
        self.dataset = dataset
        self.score = score
        self.score_history = []
        self.name = score.__name__ if name is None else name
        self.steps_to_update = steps_to_update
        self.steps = 0

    def update(self):
        if self.steps % self.steps_to_update == 0:
            self.score_history.append(self.score(self.dataset))
        self.steps += 1
        

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