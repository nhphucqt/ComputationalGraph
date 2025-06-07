import numpy as np
from cgraph.node import Node

def sigmoid(x: Node) -> Node:
    """
    Computes the sigmoid activation function.
    
    Args:
        x (Node): Input node.
        
    Returns:
        Node: Output node with sigmoid applied.
    """
    return x.sigmoid()

def softmax(x: Node, dim: int) -> Node:
    """
    Computes the softmax activation function.
    
    Args:
        x (Node): Input node.
        dim (int): Dimension along which to compute softmax.
    Returns:
        Node: Output node with softmax applied.
    """
    return x.softmax(dim)

def one_hot_encode(indices: Node, num_classes: int) -> Node:
    """
    One-hot encodes the input indices.
    
    Args:
        indices (Node): Input node containing class indices.
        num_classes (int): Total number of classes.
        
    Returns:
        Node: Output node with one-hot encoded representation.
    """

    assert indices.data.ndim == 1, "Indices must be a 1D array."
    encode = Node(np.eye(num_classes)[indices.data])
    return encode