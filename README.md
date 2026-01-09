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

## Implementation Details

### Tensor Representation

Tensors are implemented using **row-major order** (C-style) memory layout, where elements are stored contiguously in memory with the last dimension varying fastest. This design choice offers several advantages:

- **Cache-friendly**: Sequential memory access patterns improve CPU cache utilization
- **Compatibility**: Matches C/C++ native array layout for intuitive indexing
- **Efficient operations**: Matrix multiplication and element-wise operations benefit from contiguous storage

**Example**: A tensor with shape `[2, 3]` is stored in memory as:

$$[a_{00}, a_{01}, a_{02}, a_{10}, a_{11}, a_{12}]$$

**3D Example**: A tensor with shape `[2, 3, 3]` (2 matrices of 3×3) is stored as:

$$[a_{000}, a_{001}, a_{002}, a_{010}, a_{011}, a_{012}, a_{020}, a_{021}, a_{022}, a_{100}, a_{101}, a_{102}, a_{110}, a_{111}, a_{112}, a_{120}, a_{121}, a_{122}]$$

To access element at indices $[i, j, k]$:
- Stride for dimension 0: $3 \times 3 = 9$
- Stride for dimension 1: $3$
- Stride for dimension 2: $1$
- Flat index: $i \times 9 + j \times 3 + k$

Example: Element at $[1, 2, 1]$ → flat index = $1 \times 9 + 2 \times 3 + 1 = 16$ → $a_{121}$

The internal representation uses:
```cpp
std::vector<size_t> shape;   /* Dimensions of the tensor */
std::vector<double> data;    /* Flattened data in row-major order */
```

Index computation from multi-dimensional indices to flat index follows the formula:

$$\text{flat\_index} = i_0 \times \text{stride}_0 + i_1 \times \text{stride}_1 + \ldots + i_n \times \text{stride}_n$$

where $\text{stride}_k = \prod_{j=k+1}^{n-1} \text{shape}[j]$

### Backpropagation Implementation

The library implements **reverse-mode automatic differentiation** based on fundamental calculus principles:

#### Chain Rule Application

For a composite function $L = f(g(x))$, the derivative is:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial g} \times \frac{\partial g}{\partial x}$$

In neural networks, this translates to propagating gradients backward through layers:

$$\text{gradInput} = \text{backward}(\text{gradOutput})$$

where $\text{gradOutput} = \frac{\partial L}{\partial \text{output}}$ and $\text{gradInput} = \frac{\partial L}{\partial \text{input}}$

#### Gradient Caching Strategy

Each layer caches necessary values during the forward pass for efficient backward computation:

- **Dense Layer**: Caches input tensor to compute weight gradients $\frac{\partial L}{\partial W} = \text{gradOutput} \times \text{input}^T$
- **Activation Layer**: Caches input to compute derivatives (e.g., ReLU: $\text{grad} \times (\text{input} > 0)$)

This design follows the **computational graph** paradigm where:

#### Gradient Accumulation

Weight gradients are accumulated across batch samples:
```cpp
for each sample in batch:
    weightGrad += gradOutput * input  // Outer product for Dense layer
```

This enables mini-batch training where gradients are averaged over multiple samples before updating weights.

#### Mathematical Foundations

**Dense Layer (Linear Transform)**:
- Forward: $y = Wx + b$
- Backward:
  - $\frac{\partial L}{\partial x} = W^T \times \frac{\partial L}{\partial y}$ (input gradient)
  - $\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \times x^T$ (weight gradient)
  - $\frac{\partial L}{\partial b} = \frac{\partial L}{\partial y}$ (bias gradient)

**Activation Functions**:
- ReLU: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \times \mathbb{1}_{x > 0}$
- Sigmoid: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \times \sigma(x) \times (1 - \sigma(x))$
- Tanh: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \times (1 - \tanh^2(x))$

**Loss Function (MSE)**:
- Forward: $L = \frac{1}{n} \sum_{i=1}^{n} (\text{predictions}_i - \text{targets}_i)^2$
- Backward: $\frac{\partial L}{\partial \text{predictions}} = \frac{2}{n} \times (\text{predictions} - \text{targets})$

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

This project follows MIT license [./LICENSE](LICENSE) and is intended for educational purposes.