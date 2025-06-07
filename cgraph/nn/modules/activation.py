from .module import Module
import cgraph.nn.functional as F

class Activation(Module):
    """
    Base class for activation functions in neural networks.
    This class inherits from Module and provides a common interface for all activation functions.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Forward pass for the activation function.
        This method should be overridden by subclasses to implement the specific activation function.
        
        Args:
            x: Input tensor.
        
        Returns:
            Activated tensor.
        """
        raise NotImplementedError("Forward method must be implemented in subclasses.")
    
class Sigmoid(Activation):
    """
    Sigmoid activation function.
    Applies the sigmoid function element-wise to the input tensor.
    """

    def forward(self, x):
        """
        Forward pass for the sigmoid activation function.
        
        Args:
            x: Input tensor.
        
        Returns:
            Activated tensor with sigmoid applied.
        """
        return F.sigmoid(x)
    
    def inner_repr(self, indent=0):
        return "Sigmoid()"