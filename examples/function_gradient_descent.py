from typing import Tuple
import numpy as np
from graph import Node

def loss_fn(x):
    # Derivative of the function 7*x^2 + 3*x + 2 is 14*x + 3
    # Local minimum is at x = -3/14
    return 7*x*x + 3*x + 2

def train_step(x: Node, lr: float) -> Tuple[Node, Node]:
    y = loss_fn(x)
    y.backward(np.ones_like(y.data))  # Compute gradients
    grad = x.grad
    x.data -= lr * grad  # Simple gradient descent step
    x.zero_grad()  # Reset gradient after step
    return y, x, lr * grad

def train(x: Node, lr: float, eps: float = 1e-6) -> Node:
    i = 0
    while True:
        i += 1
        y, x, move_step = train_step(x, lr)
        print(f"Step {i}:", x.data)
        if abs(move_step) < eps:
            break
    return y

if __name__ == "__main__":
    x = Node(1.0, requires_grad=True)  # Initialize x as a Node with requires_grad=True
    y = train(x, lr=0.001, eps=1e-12)
    print(f"Final value of x: {x.data}, Final loss: {y.data}")
    print(f"Expected value of x: {-3/14}, Expected loss: {loss_fn(-3/14)}")
    # Expected output: Final value of x close to the minimum point of the function
    # and the gradient should be close to zero.