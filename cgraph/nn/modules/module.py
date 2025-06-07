from .parameter import Parameter

class Module:
    def __init__(self):
        self.training = True
        self._parameters = {}

    def forward(self, *args, **kwargs):
        """
        Forward pass of the module. This method should be overridden by subclasses.
        """
        raise NotImplementedError("Forward method must be implemented in subclasses.")
    
    def __call__(self, *args, **kwargs):
        """
        Calls the forward method of the module.
        Args:
            *args: Positional arguments for the forward method.
            **kwargs: Keyword arguments for the forward method.
        Returns:
            Output of the forward method.
        """
        return self.forward(*args, **kwargs)

    def named_parameters(self):
        """
        Returns an iterator over module parameters.
        """
        if len(self._parameters) == 0:
            self._parameters = {
                f'{self.__class__.__name__}.{id}.{name}': param for id, (name, param) in enumerate(self.__get_parameters())
            }
        # print(type(self._parameters))
        # print("Parameters:", self._parameters)
        for name, param in self._parameters.items():
            yield name, param

    def parameters(self):
        """
        Returns an iterator over module parameters.
        This method is a shorthand for named_parameters without names.
        """
        for _, param in self.named_parameters():
            yield param

    def __get_parameters(self):
        for name, param in self.__getstate__().items():
            # print(f"Checking {name} of type {type(param)}")
            if isinstance(param, Parameter):
                yield name, param
            elif isinstance(param, Module):
                yield from param.named_parameters()

    def train(self):
        """
        Sets the module in training mode.
        """
        self.training = True
        for param in self.parameters():
            param.requires_grad = True
        return self
    
    def eval(self):
        """
        Sets the module in evaluation mode.
        """
        self.training = False
        for param in self.parameters():
            param.requires_grad = False
        return self
    
    def zero_grad(self):
        """
        Resets the gradients of all parameters to zero.
        """
        for param in self.parameters():
            if param.grad is not None:
                param.grad.fill(0)
        return self
    
    def __getstate__(self):
        """
        Returns the state of the module for serialization.
        """
        state = {}
        for name, param in self.__dict__.items():
            if isinstance(param, Parameter) or isinstance(param, Module):
                state[name] = param
        return state
    
    def __setstate__(self, state):
        """
        Restores the state of the module from serialization.
        """
        for name, param in state.items():
            if isinstance(param, Parameter) or isinstance(param, Module):
                self.__dict__[name] = param
            else:
                raise TypeError(f"Unsupported type {type(param)} for parameter {name}")
            
    def inner_repr(self, indent=0):
        rep_str = self.__class__.__name__ + "(\n"
        indent += 2
        for name, param in self.__getstate__().items():
            if isinstance(param, Module):
                rep_str += " " * indent + f"{name}={param.inner_repr(indent + 2)},\n"
        return rep_str + " " * (indent - 2) + ")"
    
    def __repr__(self):
        return self.inner_repr()