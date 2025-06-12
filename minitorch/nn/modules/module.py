from minitorch.nn.parameter import Parameter

class Module:
    def __init__(self):
        super().__setattr__('_parameters', {})
        super().__setattr__('_modules', {})

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def __getattr__(self, name):
        parameters = self.__dict__.get('_parameters', {})
        if name in parameters:
            return parameters[name]

        modules = self.__dict__.get('_modules', {})
        if name in modules:
            return modules[name]

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def parameters(self):
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the forward method.")
    
    def __repr__(self):
        rep = f"{self.__class__.__name__}(\n"
        for name, param in self._parameters.items():
            rep += f"  ({name}): {param},\n"
        for name, module in self._modules.items():
            rep += f"  ({name}): {module},\n"
        rep += ")"
        return rep
