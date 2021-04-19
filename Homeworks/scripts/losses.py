import torch
from torch import nn

class L1_loss:
    """
    Implements the L1 norm regularization loss function
    
    Parameters
    ----------
    model: nn.Module
        Model to apply the regularization
    
    n_observations: int
        Number of observations in the training data, required to make the amount
        of regularization independent from the training set size
        
    loss: function with signature loss(output, target, **kwargs) -> float, optional
        Loss function to penalize, Default: nn.MSELoss
    
    L1_coef: float, optional
        L1 regularization parameter, the higher the more regularization is applied  
        Default: 1e-4    
        
    number_of_classes: int, optional
        Set this as the number of classes of the target if you want to use a regression loss
        (for example MSELoss) when the problem is a classification one. Default=None
    """
    def __init__(self, model: nn.Module, n_observations: int, loss: torch.nn = nn.MSELoss(),
                 L1_coef: float = 1e-4, number_of_classes = None) -> None:
        self.model = model
        self.n_observations = n_observations
        self.loss = loss
        self.l1 = L1_coef
        self.num_classes = number_of_classes
    
    
    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        output: torch.Tensor
            Prediction of the model

        target: torch.Tensor
            True label of the predictions
            
        Returns
        -------
        loss: float
            L1 penalized loss on the given data and model
        """
        if self.num_classes is not None:
            target = nn.functional.one_hot(target.long(), self.num_classes).float()
        loss = self.loss(output, target)
        regularization = 0
        for name, param in self.model.named_parameters():
            if '.weight' in name:
                regularization += torch.norm(param, 1)
        
        return loss + (self.l1/self.n_observations)*regularization



class CCQL:
    """
    Correct class Quadratic Loss loss function

    Parameters
    ----------
    loss: function with signature loss(output, target, **kwargs) -> float
        Loss function to penalize, Default: nn.MSELoss
    
    CC_bias: float, optional
        Correct Class bias parameter, the higher the more the loss is biased
        towards the correct class, good values are CC_bias=sqrt(K-1)-1 with
        K=number of classes, Default: 2
        Default: 1e-4    
    
    number_of_classes: int, optional
        Set this as the number of classes of the target if you want to use a regression loss,
        for example MSELoss, when the problem is a classification one. Default=None
    """          
    def __init__(self, loss: torch.nn = nn.MSELoss(), CC_bias: float = 2, number_of_classes = None) -> None:
        self.w = CC_bias
        self.loss = loss
        self.num_classes = number_of_classes
        

    def __call__(self, output: torch.Tensor, target: torch.Tensor):
        """
        Parameters
        ----------
        output: torch.Tensor
            Prediction of the model

        target: torch.Tensor
            True label of the predictions
        
        Returns
        -------
        loss: float
            CCQL loss on the given data
        """ 
        # Correct class bias
        correct_class = [output[i, target[i]] for i in range(len(target))]
        loss = 0.5 * self.w * torch.mean((1-torch.Tensor(correct_class))**2)
        
        # Baseline loss
        if self.num_classes is not None:
            target = nn.functional.one_hot(target.long(), self.num_classes).float()
        loss += self.loss(output, target)
    
        return loss