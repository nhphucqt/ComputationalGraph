from typing import List, Tuple
import numpy as np

class GradFunc:
    def __init__(self):
        self.nodes = []
    
    def push(self, node) -> None:
        self.nodes.append(node)

    def items(self) -> List:
        return self.nodes
    
    def exists_grad(self) -> bool:
        return any(node.requires_grad for node in self.nodes)

    def backward(self, gradient: np.ndarray) -> List[np.ndarray]:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def _reshape_gradient(self, gradient: np.ndarray, shape: Tuple[int]) -> np.ndarray:
        if gradient.shape == shape:
            return gradient
        while len(gradient.shape) > len(shape):
            gradient = gradient.sum(axis=0)
        for i, dim in enumerate(shape):
            if dim == 1:
                gradient = np.sum(gradient, axis=i, keepdims=True)
        return gradient
    
    @staticmethod
    def reshape_decorator(backward_func):
        def wrapper(self, gradient, *args, **kwargs):
            grads = backward_func(self, gradient, *args, **kwargs)
            reshaped_grads = []
            for node, grad in zip(self.items(), grads):
                reshaped_grads.append(self._reshape_gradient(grad, node.data.shape))
            return reshaped_grads
        return wrapper
    
class AddFunc(GradFunc):
    def __init__(self, x, y):
        super().__init__()
        self.push(x)
        self.push(y)

    @GradFunc.reshape_decorator
    def backward(self, gradient: np.ndarray) -> List[any]:
        return [gradient, gradient]

class SubFunc(GradFunc):
    def __init__(self, x, y):
        super().__init__()
        self.push(x)
        self.push(y)

    @GradFunc.reshape_decorator
    def backward(self, gradient) -> List[any]:
        return [gradient, -gradient]
    
class MatMulFunc(GradFunc):
    def __init__(self, x, y):
        super().__init__()
        self.push(x)
        self.push(y)

    @GradFunc.reshape_decorator
    def backward(self, gradient: np.ndarray) -> List[any]:
        x, y = self.items()
        return [
            gradient @ np.moveaxis(y.data, -1, 0), 
            np.moveaxis(x.data, -1, 0) @ gradient
        ]

class MulFunc(GradFunc):
    def __init__(self, x, y):
        super().__init__()
        self.push(x)
        self.push(y)

    @GradFunc.reshape_decorator
    def backward(self, gradient) -> List[any]:
        x, y = self.items()
        return [gradient * y.data, gradient * x.data]