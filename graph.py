# https://medium.com/data-science/recreating-pytorch-from-scratch-with-gpu-support-and-automatic-differentiation-8f565122a3cc
# https://github.com/lucasdelimanogueira/PyNorch/blob/main/norch/tensor.py

# Matrix Calculus
# https://explained.ai/matrix-calculus/
# https://cs231n.stanford.edu/vecDerivs.pdf
# https://atmos.washington.edu/~dennis/MatrixCalculus.pdf

# Matrix Multiplication
# https://docs.pytorch.org/docs/stable/generated/torch.matmul.html
# https://peps.python.org/pep-0465/

from typing import Tuple
import numpy as np
from function import *

def autocast_to_node(func):
    def wrapper(self, other):
        if not isinstance(other, Node):
            other = Node(other)
        return func(self, other)
    return wrapper

class Node:
    def __init__(self, data: np.ndarray, grad_fn: GradFunc = None, requires_grad: bool = False):
        self.data = np.array(data)
        self.grad_fn = grad_fn
        self.grad = None
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"Node(data={self.data}, requires_grad={self.requires_grad}, grad_fn={self.grad_fn})"

    @autocast_to_node
    def __add__(a: 'Node', b: 'Node') -> 'Node':
        c = Node(a.data + b.data)
        if c.grad_fn.exists_grad():
            c.grad_fn = AddFunc(a, b)
            c.requires_grad = True
        return c
    
    @autocast_to_node
    def __radd__(a: 'Node', b: 'Node') -> 'Node':
        return b.__add__(a)

    @autocast_to_node
    def __sub__(a: 'Node', b: 'Node') -> 'Node':
        c = Node(a.data - b.data)
        if c.grad_fn.exists_grad():
            c.grad_fn = SubFunc(a, b)
            c.requires_grad = True
        return c
    
    @autocast_to_node
    def __rsub__(a: 'Node', b: 'Node') -> 'Node':
        return b.__sub__(a)
    
    @autocast_to_node
    def __mul__(a: 'Node', b: 'Node') -> 'Node':
        c = Node(a.data * b.data)
        if c.grad_fn.exists_grad():
            c.grad_fn = MulFunc(a, b)
            c.requires_grad = True
        return c
    
    @autocast_to_node
    def __rmul__(a: 'Node', b: 'Node') -> 'Node':
        return b.__mul__(a)

    @autocast_to_node
    def __matmul__(a: 'Node', b: 'Node') -> 'Node':
        c = Node(a.data @ b.data)
        if c.grad_fn.exists_grad():
            c.grad_fn = MatMulFunc(a, b)
            c.requires_grad = True
        return c
    
    @autocast_to_node
    def __rmatmul__(a: 'Node', b: 'Node') -> 'Node':
        return b.__mul__(a)

    def backward(self, gradient=np.array(1)) -> None:
        self.grad = gradient
        for grad, node in zip(self.grad_fn.backward(gradient), self.grad_fn.items()):
            if node.requires_grad:
                if node.grad is None:
                    node.grad = grad
                else:
                    node.grad += grad
                if node.grad_fn:
                    node.backward(grad)

    def zero_grad(self):
        self.grad = None
        
    def detach(self) -> 'Node':
        self.grad = None
        self.grad_fn = None