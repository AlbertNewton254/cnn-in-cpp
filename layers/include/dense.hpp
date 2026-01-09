/* dense.hpp */

#ifndef DENSE_HPP
#define DENSE_HPP

#include "layer.hpp"

/**
 * Fully connected (dense) layer: y = Wx + b
 *
 * Implements pure mathematical convention where W in R^(m x n)
 * transforms input x in R^n to output y in R^m
 *
 * weights: Weight matrix of shape {outputSize, inputSize}
 * biases: Bias vector of shape {outputSize}
 * weightGrad: Gradient of weights
 * biasGrad: Gradient of biases
 * inputCache: Cached input from forward pass for backward computation
 */
class Dense : public Layer {
private:
	Tensor weights;
	Tensor biases;
	Tensor weightGrad;
	Tensor biasGrad;
	Tensor inputCache;

public:
	/**
	 * Create a dense layer with random initialization
	 *
	 * inputSize: Number of input features
	 * outputSize: Number of output features
	 */
	Dense(size_t inputSize, size_t outputSize);

	/**
	 * Forward pass: y = Wx + b
	 *
	 * input: Input tensor
	 *
	 * Output: Output tensor
	 */
	Tensor forward(const Tensor& input) override;

	/**
	 * Backward pass: dL/dx = W^T (dL/dy)
	 *
	 * gradOutput: Gradient of loss with respect to output
	 *
	 * Output: Gradient of loss with respect to input
	 */
	Tensor backward(const Tensor& gradOutput) override;

	/**
	 * Check if layer has trainable parameters (always true for Dense)
	 *
	 * Output: True
	 */
	bool hasWeights() const override { return true; }

	/**
	 * Get pointers to trainable parameters
	 *
	 * Output: Vector containing pointers to weights and biases
	 */
	std::vector<Tensor*> getWeights() override;

	/**
	 * Get pointers to parameter gradients
	 *
	 * Output: Vector containing pointers to weight and bias gradients
	 */
	std::vector<Tensor*> getGradients() override;
};

#endif