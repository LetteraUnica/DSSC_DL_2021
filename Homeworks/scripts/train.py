import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pylab as pl
import seaborn as sns


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

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


def train_epoch(model: nn.Module, train_set: DataLoader, criterion, optimizer, scheduler=None):
    """
    Evaluates a model on val_set and returns the loss
    
    Parameters
    ----------
    model: nn.Module
        Model to train
    
    train_set: DataLoader
        Training set
        
    criterion: loss function
        Loss function criterion
    
    optimizer: torch optimizer
        Optimizer to use to train the model
    
    scheduler: torch.optim.lr_scheduler
        Learning rate scheduler, default: None
    
    Returns
    -------
    float: 
        Loss on the training epoch
    """
    model.train()
    meter = AverageMeter()
    
    for data in train_set:
        inputs, labels = data
        optimizer.zero_grad()

        outputs = torch.squeeze(model(inputs))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        meter.update(loss.item(), inputs.size()[0])
    
    if scheduler is not None: 
        scheduler.step()
    return meter.avg


def train_model(model, train_set, criterion, optimizer, n_epochs, 
                val_set=None, val_epochs=1, val_criterion=None,
                stopping_criterion=None, scheduler=None):    
    """
    Trains a model, can keep track of learning metrics on validation set
    and implements early stopping
    
    Parameters
    ----------
    model: nn.Module
        Model to train
    
    train_set: DataLoader
        Training set
        
    criterion: loss function
        Loss function criterion
    
    optimizer: torch optimizer
        Optimizer to use to train the model
    
    n_epochs: int
        Number of epochs to train the model
    
    val_set: DataLoader
        Validation set to keep track of the generalization error, Default: None
        
    val_epochs: int
        Distance in epochs when to evaluate the generalization error,
        default: 1 i.e. the generalizatione error is evaluated at each epoch
    
    val_criterion: loss function with signature f(model, val_set) -> float
        Evaluation criterion on the test set, Default: Same as criterion
                
    stopping_criterion: function with signature f(val_loss, model, **kwargs) -> bool
        If this returns true it stops the training and the model
        is reverted to the one with the best validation error,
        it only works if val_set is not None, Default: None
    
    scheduler: torch.optim.lr_scheduler
        Learning rate scheduler, default: None
    
    Returns
    -------
    float: 
        Loss on the training epoch
    """
    train_losses = []
    val_losses = []
    if val_criterion is None:
        val_criterion = criterion
        
    for epoch in range(n_epochs):
        print(f"\nEpoch: {epoch}") 
        
        # Model training
        train_loss = train_epoch(model, train_set, criterion, optimizer, scheduler=scheduler)
        train_losses.append(train_loss)
        print(f"Train loss: {train_loss}")
        
        # Model testing
        if val_set is not None and epoch%val_epochs==0:
            val_loss = eval_model(model, val_set, val_criterion)
            val_losses.append(val_loss)
            print(f"Validation loss: {val_loss}")
        
            # Stopping criterion
            if stopping_criterion is not None:
                if stopping_criterion(val_loss, model):
                    print("Reached early stopping!")
                    model.load_state_dict(stopping_criterion.best_model)
                    break
    
    # Return statements
    if val_set is not None:
        return train_losses, val_losses
    
    return train_losses