import numpy as np
from cgraph.node import Node  

x = Node([
    [1.0, 2.0],
    [3.0, 4.0],
], requires_grad=True)

y = Node([
    [1.5],
    [3.2],
], requires_grad=True)

ans = x @ y

print(ans)
ans.backward(gradient=np.ones_like(ans.data))
print(f"x.grad: {x.grad} \ny.grad: {y.grad}")