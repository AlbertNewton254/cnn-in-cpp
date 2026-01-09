# Quick Start Guide

This guide shows you how to use the current features to build and train a neural network.

## Building Your First Model

### 1. Include Required Headers

```cpp
#include "sequential.hpp"
#include "dense.hpp"
#include "activation.hpp"
#include "mse.hpp"
#include "sgd.hpp"
#include <memory>
```

### 2. Create Model Architecture

```cpp
Sequential model;

/* Add layers */
model.addLayer(std::make_shared<Dense>(inputSize, hiddenSize));
model.addLayer(std::make_shared<Activation>(ActivationType::ReLU));
model.addLayer(std::make_shared<Dense>(hiddenSize, outputSize));
```

**Available Activation Types:**
- `ActivationType::ReLU` - Rectified Linear Unit
- `ActivationType::Sigmoid` - Logistic function
- `ActivationType::Tanh` - Hyperbolic tangent
- `ActivationType::Softmax` - Normalized exponential

### 3. Initialize Loss and Optimizer

```cpp
MSE loss;                    /* Mean Squared Error */
SGD optimizer(0.01);         /* Learning rate = 0.01 */
```

### 4. Prepare Training Data

```cpp
/* Input data: shape {batchSize, inputSize} */
Tensor X({4, 2});
X.at({0, 0}) = 0.0; X.at({0, 1}) = 0.0;
X.at({1, 0}) = 0.0; X.at({1, 1}) = 1.0;
X.at({2, 0}) = 1.0; X.at({2, 1}) = 0.0;
X.at({3, 0}) = 1.0; X.at({3, 1}) = 1.0;

/* Target labels: shape {batchSize, outputSize} */
Tensor y({4, 1});
y.at({0, 0}) = 0.0;
y.at({1, 0}) = 1.0;
y.at({2, 0}) = 1.0;
y.at({3, 0}) = 0.0;
```

### 5. Training Loop

```cpp
int epochs = 1000;
for (int epoch = 0; epoch < epochs; epoch++) {
    /* Forward pass */
    Tensor predictions = model.forward(X);

    /* Compute loss */
    Tensor lossValue = loss.forward(predictions, y);

    /* Backward pass */
    Tensor gradLoss = loss.backward(predictions, y);
    model.backward(gradLoss);

    /* Update parameters */
    std::vector<Tensor*> params = model.getParameters();
    std::vector<Tensor*> grads = model.getGradients();
    optimizer.step(params, grads);
    optimizer.zeroGrad(grads);

    /* Print progress */
    if (epoch % 100 == 0) {
        std::printf("Epoch %d - Loss: %.6f\n", epoch, lossValue.get({0}));
    }
}
```

### 6. Make Predictions

```cpp
Tensor testInput({1, 2});
testInput.at({0, 0}) = 1.0;
testInput.at({0, 1}) = 0.0;

Tensor prediction = model.forward(testInput);
std::printf("Prediction: %.4f\n", prediction.get({0, 0}));
```

## Working with Tensors

### Creating Tensors

```cpp
/* Zero-initialized tensor */
Tensor a({2, 3});  // Shape: 2x3

/* With initial values */
Tensor b({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

/* Filled with specific value */
Tensor c({2, 3}, 5.0);

/* Factory methods */
Tensor zeros = Tensor::zeros({2, 3});
Tensor ones = Tensor::ones({2, 3});
```

### Tensor Operations

```cpp
Tensor a = Tensor::ones({2, 3});
Tensor b = Tensor::ones({2, 3});

/* Element-wise operations */
Tensor sum = a + b;
Tensor diff = a - b;
Tensor prod = a * 2.0;
Tensor hadamard = a.hadamard(b);  /* Element-wise multiplication */

/* Matrix operations */
Tensor c = a.matmul(b.transpose());  /* Matrix multiplication */
Tensor transposed = a.transpose();

/* Access elements */
double value = a.get({0, 1});  /* Read */
a.at({0, 1}) = 3.5;            /* Write */
```

## Running Tests

```bash
cd tests
make
make run
```

This will build and run all unit tests for:
- Tensor operations
- Layer implementations
- Model architecture
- Loss functions
- Optimizers

## Tips and Best Practices

1. **Learning Rate**: Start with 0.01 and adjust based on convergence
2. **Architecture**: Begin with small networks (e.g., 2-8-4-1) and scale up
3. **Data Normalization**: Scale inputs to [0, 1] or [-1, 1] range
4. **Batch Training**: Use batched inputs (2D tensors) for better efficiency
5. **Monitor Loss**: Print loss every N epochs to track training progress

## Example: XOR Problem

The XOR problem is a classic test for neural networks:

```cpp
/* Create model */
Sequential model;
model.addLayer(std::make_shared<Dense>(2, 8));
model.addLayer(std::make_shared<Activation>(ActivationType::ReLU));
model.addLayer(std::make_shared<Dense>(8, 1));

/* Prepare XOR data */
Tensor X({4, 2});
X.at({0, 0}) = 0.0; X.at({0, 1}) = 0.0;  /* 0 XOR 0 = 0 */
X.at({1, 0}) = 0.0; X.at({1, 1}) = 1.0;  /* 0 XOR 1 = 1 */
X.at({2, 0}) = 1.0; X.at({2, 1}) = 0.0;  /* 1 XOR 0 = 1 */
X.at({3, 0}) = 1.0; X.at({3, 1}) = 1.0;  /* 1 XOR 1 = 0 */

Tensor y({4, 1});
y.at({0, 0}) = 0.0;
y.at({1, 0}) = 1.0;
y.at({2, 0}) = 1.0;
y.at({3, 0}) = 0.0;

/* Train (see step 5 above) */
```

## Next Steps

- Experiment with different architectures and hyperparameters
- Try different activation functions
- Prepare your own datasets
- Stay tuned for CNN layers and MNIST support!

## Common Issues

**Problem**: Model not converging
**Solution**: Lower the learning rate or add more hidden units

**Problem**: Loss increases during training
**Solution**: Learning rate might be too high, try reducing it by 10x

**Problem**: Compilation errors
**Solution**: Ensure you're using C++17 and all required headers are included
