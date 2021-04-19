from copy import deepcopy
import torch
from torch import nn

class stopping_criterion:
    def __init__(self):
        self.E_opt = None
        self.best_model = None
    
    def _update_best_model(self, new_E_opt, new_best_model):
        self.E_opt = new_E_opt
        self.best_model = deepcopy(new_best_model.state_dict())

    def get_best_model(self):
        return self.best_model


# 2.a Implement the first class of early stopping criteria described on the paper of Prechelt et al.
class stop_by_threshold(stopping_criterion):
    """
    Implements a early stopping criteria based on the validation loss
    Training is stopped when Loss_current > a*Loss_best
    i.e. the loss on the validation set gets bigger than the minimum
    loss ever achivied on the validation set times a constant a>1
    
    Parameters
    
    threshold: float, optional
        Threshold when to stop training, Default=1.05
        
    Attributes
    ----------
    E_opt: float
        Best validation loss ever reached so far
    best_model: nn.state_dict()
        Stores the model with the best validation loss
    """
    def __init__(self, threshold = 1.05):
        super().__init__()
        self.threshold = threshold
    
    def __call__(self, val_loss, model):
        """
        Parameters
        ----------
        val_loss: float
            Validation loss of the model

        model: nn.Module
            Current model, only needed to get the state_dict()
            
        Returns
        -------
        stop_training: bool
            Tells whether to stop the training or not
        """
        # Update E_opt if I get a smaller validation loss
        if self.E_opt is None or val_loss < self.E_opt:
            self._update_best_model(val_loss, model)
            return False
        
        # Continue training if GL is lower than the threshold
        GL = val_loss/self.E_opt
        if GL < self.threshold:
            return False
        
        # Otherwise stop training
        return True
    
    
# 2.b Implement the third class of early stopping criteria described on the paper of Prechelt et al.
class stop_by_increase(stopping_criterion):
    """
    Implements a early stopping criteria based on the validation loss
    Training is stopped when Loss_t > Loss_t-gap
    i.e. the loss on the validation set gets bigger than the
    validation loss gap epochs before
    
    Parameters
    
    gap: float, optional
        The gap parameter explained above, Default=5
        
    Attributes
    ----------
    E_opt: float
        Best validation loss ever reached so far
    best_model: nn.state_dict()
        Stores the model with the best validation loss
    val_losses: list of float
        Stores the validation losses so far
    """
    def __init__(self, gap = 5):
        super().__init__()
        self.gap = gap
        self.val_losses = []
        
    def __call__(self, val_loss, model):
        """
        Parameters
        ----------
        val_loss: float
            Validation loss of the model

        model: nn.Module
            Current model, only needed to get the state_dict()
        
        Returns
        -------
        stop_training: bool
            Tells whether to stop the training or not
        """
        self.val_losses.append(val_loss)
        # Update E_opt if I get a smaller validation loss
        if self.E_opt is None or val_loss < self.E_opt:
            self._update_best_model(val_loss, model)
            return False
        
        # Continue training if the loss is less than the loss 'gap' iterations before
        if val_loss < self.val_losses[-self.gap - 1]:
            return False
        
        # Otherwise stop training
        return True