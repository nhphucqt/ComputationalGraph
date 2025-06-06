# ComputationalGraph

A minimal deep learning framework that implements a computational graph with automatic differentiation, inspired by PyTorch's `Tensor` and autograd features. This project wraps NumPy's `ndarray` to provide gradient tracking and backpropagation capabilities. It also includes a simple dataset and dataloader system for efficient data loading during training.

## Features

- **Tensor-like API:** Core data structure wraps NumPy arrays, supporting basic tensor operations and autograd.
- **Automatic Differentiation:** Enables gradient computation for building and training neural networks.
- **Dataset & DataLoader:** Utilities for loading and batching data, similar to PyTorch's `Dataset` and `DataLoader`.
- **Extensible Design:** Future features (e.g., more operations, GPU support) are planned for further PyTorch-like functionality.

## Getting Started

1. Clone the repository.
2. Install dependencies (e.g. NumPy) in `requirements.txt`.
3. Explore the `Node` in Graph, `Dataset`, and `DataLoader` modules in `examples`. For example, `python -m matrix_operations`.

## References

### Core Concepts

- [Recreating PyTorch from Scratch with GPU Support and Automatic Differentiation](https://medium.com/data-science/recreating-pytorch-from-scratch-with-gpu-support-and-automatic-differentiation-8f565122a3cc)  
    A comprehensive guide on building a PyTorch-like framework, covering computational graphs, autograd, and GPU support.

- [PyNorch Tensor Implementation](https://github.com/lucasdelimanogueira/PyNorch/blob/main/norch/tensor.py)  
    An open-source project that implements a minimal PyTorch-like tensor with autograd, serving as a reference for design and implementation.

### Matrix Calculus

- [CS231n: Vector Calculus Review and Reference](https://cs231n.stanford.edu/vecDerivs.pdf)  
    A concise summary of vector and matrix calculus, essential for understanding backpropagation and gradient computation.

- [Matrix Calculus for Deep Learning](https://atmos.washington.edu/~dennis/MatrixCalculus.pdf)  
    A concise document explaining matrix calculus rules, useful for implementing gradients in computational graphs.

- [Explained.ai: Matrix Calculus](https://explained.ai/matrix-calculus/)  
    An intuitive explanation of matrix calculus concepts, with visualizations and practical examples.

- [Matrix Calculus
(for Machine Learning and Beyond)](https://arxiv.org/pdf/2501.14787)  
    A comprehensive and modern treatment of calculus involving matrices, designed to bridge the gap between traditional calculus and the advanced differentiation needed for machine learning, optimization, and scientific computing.

### Matrix Multiplication

- [PyTorch Documentation: torch.matmul](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html):  
    Official documentation for matrix multiplication in PyTorch, describing broadcasting and operation semantics.

- [PEP 465: A dedicated infix operator for matrix multiplication](https://peps.python.org/pep-0465/):  
    Python Enhancement Proposal introducing the `@` operator for matrix multiplication, relevant for implementing intuitive APIs.

---

This project is a work in progress. Contributions and suggestions are welcome!
