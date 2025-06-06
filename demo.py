from typing import Tuple
import numpy as np
from graph import Node

def loss_fn(x: Node) -> Node:
    return 7*x*x + 3*x + 2

def train_step(x: Node, lr: float) -> Tuple[Node, Node]:
    y = loss_fn(x)
    y.backward(np.ones_like(y.data))  # Compute gradients
    grad = x.grad
    x.data -= lr * grad  # Simple gradient descent step
    x.zero_grad()  # Reset gradient after step
    return y, x, lr * grad

def train(x: Node, lr: float, eps: float = 1e-6) -> Node:
    while True:
        y, x, move_step = train_step(x, lr)
        print(x.data)
        if abs(move_step) < eps:
            break
    return y

if __name__ == "__main__":
    x = Node([
        [1.0, 2.0],
        [3.0, 4.0],
    ], requires_grad=True)
    y = Node([
        [1.5],
        [3.2],
    ], requires_grad=True)
    # z = Node([
    #     [1.0, 2.0],
    # ], requires_grad=True)
    ans = x @ y
    print(ans)
    ans.backward(gradient=np.ones_like(ans.data))
    print(f"x.grad: {x.grad} \ny.grad: {y.grad}")
    # x = Node(1.0, requires_grad=True)  # Initialize x as a Node with requires_grad=True
    # y = train(x, lr=0.001)
    # print(f"Final value of x: {x.data}, Final loss: {y.data}, Gradient: {x.grad}")
    # Expected output: Final value of x close to the minimum point of the function
    # and the gradient should be close to zero.