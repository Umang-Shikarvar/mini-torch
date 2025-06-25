
# miniTorch

**miniTorch** is a minimal, educational deep learning framework inspired by PyTorch, designed for learning, research, and rapid prototyping. Built on top of NumPy, miniTorch provides a clear and concise implementation of core deep learning concepts, including tensors, autograd, neural network modules, optimizers, and data utilities.

## 👤 Maintainers
- [Shardul Junagade](https://github.com/ShardulJunagade)
- [Umang Shikarvar](https://github.com/Umang-Shikarvar)
- [Soham Gaonkar](https://github.com/Soham-Gaonkar)



## ✨ Key Features

- **NumPy-based Tensor Engine**: Custom `Tensor` class with full support for broadcasting, dtype management, and automatic differentiation (autograd).
- **Neural Network Building Blocks**:
  - PyTorch-like `Module` base class for custom layers and models
  - Prebuilt layers: `Linear`, `Sequential`, and activation modules (`ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`)
  - Parameter management and easy extensibility
- **Functional API**: Core tensor operations (sum, exp, log, pow, transpose, relu, sigmoid, tanh, leaky_relu, softmax) for building custom computations
- **Loss Functions**: Ready-to-use `MSELoss` and `BCELoss` for regression and classification tasks
- **Optimizers**: Implementations of `SGD` and `Adam` for training neural networks
- **Data Utilities**: Simple `Dataset` and `DataLoader` classes for batching, shuffling, and iterating over data
- **Computation Graph Visualization**: Visualize your model's computation graph using Graphviz for better debugging and understanding.
- **Educational Codebase**: Clean, well-documented code ideal for students, educators, and researchers



## 🚀 Installation

Clone the repository:
```bash
git clone https://github.com/Umang-Shikarvar/miniTorch
cd miniTorch
pip install -r requirements.txt
```

To use `minitorch` as a library in your own projects, install it in editable mode:
```bash
pip install -e ./minitorch
```
Now you can import `minitorch` from anywhere in your environment and your changes will be picked up automatically.

<!-- to uninstall, use:
pip uninstall minitorch -->

Example:
```python
import minitorch
from minitorch.nn.modules import Linear, Sequential
from minitorch.optim import SGD

# Define a simple model
model = Sequential(
    Linear(4, 8),
    minitorch.nn.modules.ReLU(),
    Linear(8, 1)
)

# Create optimizer
optimizer = SGD(model.parameters(), lr=0.01)
```



## 🛠️ Documentation & Tips

- For tips and useful commands, see [`docs/tips_and_commands.md`](./docs/tips_and_commands.md).
- Explore the [`minitorch/`](./minitorch/) directory for source code and examples.



## 🤝 Contributing

We welcome contributions from everyone!
- Maintainers: see [`docs/CONTRIBUTING (maintainers).md`](./docs/CONTRIBUTING%20(maintainers).md)
- Non-maintainers: see [`docs/CONTRIBUTING.md`](./docs/CONTRIBUTING.md)

For questions, suggestions, or discussions, open an issue or start a discussion on GitHub.



## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.



**Happy learning and building with miniTorch!** ⭐
