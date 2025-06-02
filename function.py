from typing import List, Tuple
import numpy as np

class GradFunc:
    def __init__(self):
        self.nodes: List['Node'] = []
    
    def push(self, node: 'Node') -> None:
        self.nodes.append(node)

    def items(self) -> List['Node']:
        return self.nodes
    
    def exists_grad(self) -> bool:
        return any(node.requires_grad for node in self.nodes)

    def backward(self, gradient: np.ndarray) -> List[np.ndarray]:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def _reshape_gradient(self, gradient: np.ndarray, shape: Tuple[int]) -> np.ndarray:
        # print(f"Reshaping gradient from {gradient.shape} to {shape}")
        if gradient.shape == shape:
            return gradient
        while len(gradient.shape) != len(shape):
            if gradient.shape[-1] == 1:
                gradient = np.squeeze(gradient, axis=-1)
            elif gradient.shape[0] == 1:
                gradient = np.squeeze(gradient, axis=0)
            else:
                raise ValueError("Cannot reshape gradient to match the shape of the node.")
        for i, dim in enumerate(shape):
            if dim == 1:
                gradient = np.sum(gradient, axis=i, keepdims=True)
        return gradient.reshape(shape)
    
class AddFunc(GradFunc):
    def __init__(self, x: 'Node', y: 'Node'):
        super().__init__()
        self.push(x)
        self.push(y)

    def backward(self, gradient: np.ndarray) -> List[any]:
        x, y = self.items()
        grad_x = self._reshape_gradient(gradient, x.data.shape)
        grad_y = self._reshape_gradient(gradient, y.data.shape)
        return [grad_x, grad_y]

class SubFunc(GradFunc):
    def __init__(self, x: 'Node', y: 'Node'):
        super().__init__()
        self.push(x)
        self.push(y)

    def backward(self, gradient) -> List[any]:
        x, y = self.items()
        grad_x = self._reshape_gradient(gradient, x.data.shape)
        grad_y = self._reshape_gradient(-gradient, y.data.shape)
        return [grad_x, grad_y]
    
class MatMulFunc(GradFunc):
    def __init__(self, x: 'Node', y: 'Node'):
        super().__init__()
        self.push(x)
        self.push(y)

    def backward(self, gradient: np.ndarray) -> List[any]:
        x, y = self.items()
        grad_x = self._reshape_gradient(gradient @ np.moveaxis(y.data, -1, 0), x.data.shape)
        grad_y = self._reshape_gradient(np.moveaxis(x.data, -1, 0) @ gradient, y.data.shape)
        return [grad_x, grad_y]

class MulFunc(GradFunc):
    def __init__(self, x: 'Node', y: 'Node'):
        super().__init__()
        self.push(x)
        self.push(y)

    def backward(self, gradient) -> List[any]:
        x, y = self.items()
        grad_x = self._reshape_gradient(gradient * y.data, x.data.shape)
        grad_y = self._reshape_gradient(gradient * x.data, y.data.shape)
        return [grad_x, grad_y]