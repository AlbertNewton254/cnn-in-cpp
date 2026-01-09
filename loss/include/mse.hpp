#ifndef MSE_HPP
#define MSE_HPP

#include "loss.hpp"

/**
 * Mean Squared Error loss function
 *
 * Computes L = (1/n) * sum((predictions - targets)^2)
 * Gradient: dL/d(predictions) = (2/n) * (predictions - targets)
 */
class MSE : public Loss {
public:
    /**
     * Compute MSE loss value
     *
     * predictions: Model predictions
     * targets: Ground truth targets
     * Output: Scalar tensor containing mean squared error
     */
    Tensor forward(const Tensor& predictions, const Tensor& targets) override;

    /**
     * Compute gradient of MSE w.r.t. predictions
     *
     * predictions: Model predictions
     * targets: Ground truth targets
     * Output: Tensor containing gradient with same shape as predictions
     */
    Tensor backward(const Tensor& predictions, const Tensor& targets) override;
};

#endif
