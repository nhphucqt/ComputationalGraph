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
        if gradient is None:
            return None
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
    
class NegBackward(GradFunc):
    def __init__(self, x):
        super().__init__()
        self.push(x)

    @GradFunc.reshape_decorator
    def backward(self, gradient: np.ndarray) -> List[any]:
        return [-gradient]
    
class SumBackward(GradFunc):
    def __init__(self, x, dim: int = None, keepdims: bool = False):
        super().__init__()
        self.push(x)
        self.dim = dim
        self.keepdims = keepdims

    @GradFunc.reshape_decorator
    def backward(self, gradient: np.ndarray) -> List[any]:
        x = self.items()[0]
        if self.dim is None:
            return [np.full_like(x.data, gradient)]
        else:
            if not self.keepdims:
                gradient = np.expand_dims(gradient, axis=self.dim)            
            return gradient + np.zeros_like(x.data, shape=x.data.shape, dtype=x.data.dtype)

class AddBackward(GradFunc):
    def __init__(self, x, y):
        super().__init__()
        self.push(x)
        self.push(y)

    @GradFunc.reshape_decorator
    def backward(self, gradient: np.ndarray) -> List[any]:
        return [gradient, gradient]

class SubBackward(GradFunc):
    def __init__(self, x, y):
        super().__init__()
        self.push(x)
        self.push(y)

    @GradFunc.reshape_decorator
    def backward(self, gradient) -> List[any]:
        return [gradient, -gradient]
    
class MatMulBackward(GradFunc):
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

class MulBackward(GradFunc):
    def __init__(self, x, y):
        super().__init__()
        self.push(x)
        self.push(y)

    @GradFunc.reshape_decorator
    def backward(self, gradient) -> List[any]:
        x, y = self.items()
        return [gradient * y.data, gradient * x.data]
    
class DivBackward(GradFunc):
    def __init__(self, x, y):
        super().__init__()
        self.push(x)
        self.push(y)

    @GradFunc.reshape_decorator
    def backward(self, gradient: np.ndarray) -> List[any]:
        x, y = self.items()
        return [gradient / y.data, -gradient * x.data / (y.data ** 2)]
    
class LogBackward(GradFunc):
    def __init__(self, x):
        super().__init__()
        self.push(x)

    @GradFunc.reshape_decorator
    def backward(self, gradient: np.ndarray) -> List[any]:
        x = self.items()[0]
        return [gradient / x.data]
    
class SigmoidBackward(GradFunc):
    def __init__(self, x):
        super().__init__()
        self.push(x)

    @GradFunc.reshape_decorator
    def backward(self, gradient: np.ndarray) -> List[any]:
        x = self.items()[0]
        sigmoid_x = 1 / (1 + np.exp(-x.data))
        return [gradient * sigmoid_x * (1 - sigmoid_x)]
    
class CrossEntropyBackward(GradFunc):
    def __init__(self, input, target):
        super().__init__()
        self.push(input)
        self.push(target)

    @GradFunc.reshape_decorator
    def backward(self, gradient: np.ndarray) -> List[any]:
        input, target = self.items()
        # print("CrossEntropyBackward: input:", input.data, "target:", target.data)
        num_batch = input.data.shape[0] if input.data.ndim == 2 else 1

        return [gradient * (input.softmax(dim=-1).data - target.data) / num_batch, None]