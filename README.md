# CNN in C++

A lightweight, educational deep learning library implemented in C++ from scratch, designed to build, train, and deploy Convolutional Neural Networks.

## Project Goal

**The main objective is to create an API powerful enough to build, train, load/store, and run a CNN on the MNIST dataset**, providing a clear understanding of neural network internals without relying on heavy external frameworks.

## Features

- **Tensor Operations**: Multi-dimensional array support with mathematical operations (addition, subtraction, multiplication, matrix multiplication, transpose, Hadamard product)
- **Layers**: Dense (fully connected) and Activation layers (ReLU, Sigmoid, Tanh, Softmax)
- **Models**: Sequential model architecture for stacking layers
- **Loss Functions**: Mean Squared Error (MSE)
- **Optimizers**: Stochastic Gradient Descent (SGD)
- **Training Pipeline**: Complete forward/backward propagation with automatic gradient computation

## Project Structure

```
cnn-in-cpp/
├── tensor/          # Core tensor implementation
├── layers/          # Neural network layers (Dense, Activation)
├── model/           # Model architecture (Sequential)
├── loss/            # Loss functions (MSE)
├── optimizer/       # Optimization algorithms (SGD)
└── tests/           # Unit tests for all components
```

## Current Status

- Tensor operations and matrix math
- Dense and Activation layers
- Sequential model
- MSE loss and SGD optimizer
- Training pipeline (forward/backward passes)
- Comprehensive unit tests
- Model serialization (save/load) - In Progress
- Convolutional layers - Planned
- MNIST dataset support - Planned

## Getting Started

See [QUICKSTART.md](QUICKSTART.md) for a guide on using the current features.

## Building

Each module has its own Makefile. To build tests:

```bash
cd tests
make
make run
```

## Requirements

- C++17 compatible compiler (g++ recommended)
- Make

## License

Educational project - feel free to use and learn from it.