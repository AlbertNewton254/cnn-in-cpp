#ifndef LOSS_HPP
#define LOSS_HPP

#include <exception>
#include <string>
#include "../../tensor/include/tensor.hpp"

/**
 * Exception thrown when predictions and targets have mismatched shapes
 *
 * message: Error message describing the mismatch
 */
class LossShapeMismatchError : public std::exception {
private:
    std::string message;
public:
    explicit LossShapeMismatchError(const std::string& msg) : message(msg) {}
    const char* what() const noexcept override { return message.c_str(); }
};

/**
 * Abstract base class for loss functions
 */
class Loss {
public:
    virtual ~Loss() = default;

    /**
     * Compute the loss value
     *
     * predictions: Model predictions
     * targets: Ground truth targets
     * Output: Tensor containing the loss value (typically scalar)
     */
    virtual Tensor forward(const Tensor& predictions, const Tensor& targets) = 0;

    /**
     * Compute gradient with respect to predictions
     *
     * predictions: Model predictions
     * targets: Ground truth targets
     * Output: Tensor containing gradient of loss w.r.t. predictions
     */
    virtual Tensor backward(const Tensor& predictions, const Tensor& targets) = 0;
};

#endif