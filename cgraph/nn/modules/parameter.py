from cgraph.node import Node
import numpy as np

class Parameter(Node):
    """
    A Parameter is a Node that is specifically designed to be used as a model parameter.
    It inherits from Node and has the same properties, but it is intended to be used in
    neural network modules.
    """
    
    def __init__(self, shape: tuple, requires_grad: bool = True):
        """
        Initializes a Parameter with random data of the given shape.
        Args:
            shape (tuple): The shape of the parameter.
            requires_grad (bool): Whether to compute gradients for this parameter.
        """
        data = np.random.randn(*shape).astype(np.float32)
        super().__init__(data, requires_grad=requires_grad)