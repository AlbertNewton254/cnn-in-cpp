/* mse.cpp */

#include "../include/mse.hpp"

Tensor MSE::forward(const Tensor& predictions, const Tensor& targets) {
    if (predictions.getShape() != targets.getShape()) {
        throw LossShapeMismatchError("Predictions and targets must have the same shape");
    }

    Tensor diff = predictions - targets;
    Tensor squared = diff.hadamard(diff);

    double sum = 0.0;
    for (size_t i = 0; i < squared.size(); i++) {
        sum += squared.getData()[i];
    }

    double mse = sum / squared.size();
    return Tensor({1}, {mse});
}

Tensor MSE::backward(const Tensor& predictions, const Tensor& targets) {
    if (predictions.getShape() != targets.getShape()) {
        throw LossShapeMismatchError("Predictions and targets must have the same shape");
    }

    Tensor diff = predictions - targets;
    double scale = 2.0 / diff.size();
    return diff * scale;
}