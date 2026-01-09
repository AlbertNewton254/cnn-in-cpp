/* dense.hpp */

#ifndef DENSE_HPP
#define DENSE_HPP

#include "layer.hpp"

/**
 * Fully connected (dense) layer
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
	 * Forward pass: output = input * weights^T + biases
	 *
	 * input: Input tensor of shape {batchSize, inputSize} or {inputSize}
	 * Output: Output tensor of shape {batchSize, outputSize} or {outputSize}
	 */
	Tensor forward(const Tensor& input) override;

	/**
	 * Backward pass: compute input gradients and parameter gradients
	 *
	 * gradOutput: Gradient of loss with respect to output
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