/* activation.hpp */

#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "layer.hpp"

/**
 * Supported activation function types
 */
enum class ActivationType {
	ReLU,
	Sigmoid,
	Tanh,
	Softmax
};

/**
 * Activation function layer
 *
 * type: Type of activation function to apply
 * inputCache: Cached input/output from forward pass for backward computation
 */
class Activation : public Layer {
private:
	ActivationType type;
	Tensor inputCache;

public:
	/**
	 * Create an activation layer
	 *
	 * activationType: Type of activation function
	 */
	Activation(ActivationType activationType);

	/**
	 * Apply activation function element-wise
	 *
	 * input: Input tensor
	 * Output: Activated tensor
	 */
	Tensor forward(const Tensor& input) override;

	/**
	 * Compute gradient through activation function
	 *
	 * gradOutput: Gradient of loss with respect to output
	 * Output: Gradient of loss with respect to input
	 */
	Tensor backward(const Tensor& gradOutput) override;

	/**
	 * Check if layer has trainable parameters (always false for Activation)
	 *
	 * Output: False
	 */
	bool hasWeights() const override { return false; }
};

#endif