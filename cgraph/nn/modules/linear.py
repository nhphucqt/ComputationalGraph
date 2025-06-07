from .parameter import Parameter
from .module import Module

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter((in_features, out_features))
        self.bias = Parameter((out_features,)) if bias else None

        # print("Weight grad:", self.weight.requires_grad)
        # if self.bias is not None:
        #     print("Bias grad:", self.bias.requires_grad)
        # else:
        #     print("No bias in this layer")

    def forward(self, x):
        """
        Forward pass for the linear layer.
        """
        return x @ self.weight + self.bias if self.bias is not None else x @ self.weight

    def inner_repr(self, indent=0):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"