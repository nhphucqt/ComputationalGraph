# https://medium.com/data-science/recreating-pytorch-from-scratch-with-gpu-support-and-automatic-differentiation-8f565122a3cc
# https://github.com/lucasdelimanogueira/PyNorch/blob/main/norch/tensor.py

# Matrix Calculus
# https://cs231n.stanford.edu/vecDerivs.pdf
# https://atmos.washington.edu/~dennis/MatrixCalculus.pdf
# https://explained.ai/matrix-calculus/
# https://arxiv.org/pdf/2501.14787

# Matrix Multiplication
# https://docs.pytorch.org/docs/stable/generated/torch.matmul.html
# https://peps.python.org/pep-0465/

import numpy as np
from cgraph.autograd.functions import *

def autocast_to_node(func):
    def wrapper(self, other):
        if not isinstance(other, Node):
            other = Node(other)
        return func(self, other)
    return wrapper

class Node:
    def __init__(self, data: np.ndarray, requires_grad: bool = False):
        self.data = np.array(data)
        self.grad_fn = None
        self.grad = None
        self.requires_grad = requires_grad

    @property
    def shape(self):
        return self.data.shape
    @property
    def dtype(self):
        return self.data.dtype
    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return f"Node(data={self.data}, requires_grad={self.requires_grad}, grad_fn={self.grad_fn})"
    
    def __neg__(self) -> 'Node':
        c = Node(-self.data)
        if self.requires_grad:
            c.grad_fn = NegBackward(self)
            c.requires_grad = True
        return c

    @autocast_to_node
    def __add__(a: 'Node', b: 'Node') -> 'Node':
        c = Node(a.data + b.data)
        if a.requires_grad or b.requires_grad:
            c.grad_fn = AddBackward(a, b)
            c.requires_grad = True
        return c
    
    @autocast_to_node
    def __radd__(a: 'Node', b: 'Node') -> 'Node':
        return b.__add__(a)

    @autocast_to_node
    def __sub__(a: 'Node', b: 'Node') -> 'Node':
        c = Node(a.data - b.data)
        if a.requires_grad or b.requires_grad:
            c.grad_fn = SubBackward(a, b)
            c.requires_grad = True
        return c
    
    @autocast_to_node
    def __rsub__(a: 'Node', b: 'Node') -> 'Node':
        return b.__sub__(a)
    
    @autocast_to_node
    def __mul__(a: 'Node', b: 'Node') -> 'Node':
        c = Node(a.data * b.data)
        if a.requires_grad or b.requires_grad:
            c.grad_fn = MulBackward(a, b)
            c.requires_grad = True
        return c
    
    @autocast_to_node
    def __rmul__(a: 'Node', b: 'Node') -> 'Node':
        return b.__mul__(a)
    
    @autocast_to_node
    def __truediv__(a: 'Node', b: 'Node') -> 'Node':
        if np.any(b.data == 0):
            raise ZeroDivisionError("Division by zero encountered in Node division")
        c = Node(a.data / b.data)
        if a.requires_grad or b.requires_grad:
            c.grad_fn = DivBackward(a, b)
            c.requires_grad = True
        return c
    
    @autocast_to_node
    def __rtruediv__(a: 'Node', b: 'Node') -> 'Node':
        return b.__truediv__(a)

    @autocast_to_node
    def __matmul__(a: 'Node', b: 'Node') -> 'Node':
        c = Node(a.data @ b.data)
        if a.requires_grad or b.requires_grad:
            c.grad_fn = MatMulBackward(a, b)
            c.requires_grad = True
        return c
    
    @autocast_to_node
    def __rmatmul__(a: 'Node', b: 'Node') -> 'Node':
        return b.__mul__(a)
    
    def log(self) -> 'Node':
        c = Node(np.log(self.data))
        if self.requires_grad:
            c.grad_fn = LogBackward(self)
            c.requires_grad = True
        return c
    
    def sigmoid(self) -> 'Node':
        c = Node(1 / (1 + np.exp(-self.data)))
        if self.requires_grad:
            c.grad_fn = SigmoidBackward(self)
            c.requires_grad = True
        return c
    
    def softmax(self, dim: int) -> 'Node':
        exp_data = np.exp(self.data - np.max(self.data, axis=dim, keepdims=True))
        c = Node(exp_data / np.sum(exp_data, axis=dim, keepdims=True))
        # if self.requires_grad:
        #     c.grad_fn = SoftmaxBackward(self)
        #     c.requires_grad = True
        return c
    
    def gather(self, indices: 'Node', dim: int) -> 'Node':
        # print(self.data.shape, indices.data.shape, dim)
        c = Node(np.take_along_axis(self.data, indices.data, axis=dim))
        # if self.requires_grad:
        #     c.grad_fn = GatherBackward(self, indices, dim)
        #     c.requires_grad = True
        return c
    
    def sum(self, dim: int = None, keepdim: bool = False) -> 'Node':
        if keepdim and dim is None:
            raise ValueError("keepdim=True requires a specific dimension to be specified")
        if dim is None:
            c = Node(np.sum(self.data))
        else:
            c = Node(np.sum(self.data, axis=dim, keepdims=keepdim))
        if self.requires_grad:
            c.grad_fn = SumBackward(self, dim, keepdims=keepdim)
            c.requires_grad = True
        return c
    
    def item(self) -> float:
        if self.data.size != 1:
            raise ValueError("item() can only be called on a single-element tensor")
        return self.data.item()

    def backward(self, gradient=np.array(1)) -> None:
        if self.grad_fn is None:
            return
        for grad, node in zip(self.grad_fn.backward(gradient), self.grad_fn.items()):
            if grad is None:
                continue
            if node.requires_grad:
                if node.grad is None:
                    node.grad = grad
                else:
                    node.grad += grad
                node.backward(node.grad)

    def zero_grad(self):
        self.grad = None
        
    def detach(self) -> 'Node':
        self.grad = None
        self.grad_fn = None

    # def __getitem__(self, key):
        