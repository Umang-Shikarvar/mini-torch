from .module import Module

class Sequential(Module):

    def __init__(self, *modules):
        super().__init__()
        self._sequence = []
        for idx, module in enumerate(modules):
            if not isinstance(module, Module):
                raise TypeError(f"All elements must be Module instances, got {type(module)} at position {idx}.")
            self._modules[str(idx)] = module
            self._sequence.append(module)

    def __init__(self, *modules):
        super().__init__()
        self._sequence = []
        for idx, module in enumerate(modules):
            if not isinstance(module, Module):
                raise TypeError(f"All elements must be Module instances, got {type(module)} at position {idx}.")
            self._modules[str(idx)] = module
            self._sequence.append(module)
    
    def _rebuild_modules(self) -> None:
        self._modules = type(self._modules)([(str(i), m) for i, m in enumerate(self._sequence)])

    def __len__(self):
        return len(self._sequence)

    def __repr__(self):
        rep = f"{self.__class__.__name__}(\n"
        for idx, module in enumerate(self._sequence):
            rep += f"  ({idx}): {module},\n"
        rep += ")"
        return rep

    def __getitem__(self, idx: int) -> Module:
        return self._sequence[idx]

    def __setitem__(self, idx: int, module: Module) -> None:
        if not isinstance(module, Module):
            raise TypeError(f"module should be of type: {Module}")
        self._sequence[idx] = module
        self._modules[str(idx)] = module

    def __delitem__(self, idx: int) -> None:
        # Remove from sequence and modules dict
        del self._sequence[idx]
        # Rebuild _modules to preserve correct indices
        self._rebuild_modules()

    def __iter__(self):
        return iter(self._sequence)

    def __add__(self, other: 'Sequential') -> 'Sequential':
        if not isinstance(other, Sequential):
            raise TypeError(f"add operator supports only objects of Sequential class, but {type(other)} is given.")
        result = Sequential()
        for module in self._sequence:
            result.append(module)
        for module in other._sequence:
            result.append(module)
        return result

    def parameters(self):
        params = []
        for module in self._sequence:
            params.extend(module.parameters())
        return params

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def append(self, module: Module) -> 'Sequential':
        if not isinstance(module, Module):
            raise TypeError(f"module should be of type: {Module}")
        self._sequence.append(module)
        self._modules[str(len(self._sequence)-1)] = module
        return self

    def insert(self, idx: int, module: Module) -> 'Sequential':
        if not isinstance(module, Module):
            raise TypeError(f"module should be of type: {Module}")
        self._sequence.insert(idx, module)
        # Rebuild _modules to preserve correct indices
        self._rebuild_modules()
        return self

    def extend(self, sequential: 'Sequential') -> 'Sequential':
        if not isinstance(sequential, Sequential):
            raise TypeError(f"sequential should be of type: {Sequential}")
        for module in sequential:
            self.append(module)
        return self

    def pop(self, idx: int = -1) -> Module:
        module = self._sequence.pop(idx)
        # Rebuild _modules to preserve correct indices
        self._rebuild_modules()
        return module


    def forward(self, x):
        for module in self._sequence:
            x = module(x)
        return x

    def zero_grad(self):
        for module in self._sequence:
            module.zero_grad()
        return self
