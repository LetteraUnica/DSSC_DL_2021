import torch
from torch import nn

class metric:
    def __init__(self, model, dataloader, score, name=None, steps_to_update=1):
        self.model = model
        self.dataloader = dataloader
        self.score = score
        self.score_history = []
        self.name = score.__name__ if name is None else name
        self.steps_to_update = steps_to_update
        self.steps = 0

    def update(self):
        if self.steps % self.steps_to_update == 0:
            update = eval_model(self.model, self.dataloader, self.score)
            self.score_history.append(update)
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


def eval_model(model: nn.Module, val_set: DataLoader, criterion):
    """
    Evaluates a model on val_set and returns the loss
    
    Parameters
    ----------
    model: nn.Module
        Model to evaluate the loss
    
    val_set: DataLoader
        Validation set where to evaluate the loss
        
    criterion: loss function
        Loss function criterion
    
    Returns
    -------
    float: 
        Loss on the validation set
    """
    model.eval()
    meter = AverageMeter()
    
    for data in val_set:
        inputs, labels = data
        outputs = torch.squeeze(model(inputs))
        loss = criterion(outputs, labels)
        meter.update(loss.item(), inputs.size()[0])
        
    return meter.avg