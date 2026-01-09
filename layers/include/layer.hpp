/* layer.hpp */

#ifndef LAYER_HPP
#define LAYER_HPP

#include "../../tensor/include/tensor.hpp"
#include <vector>
#include <exception>

/**
 * Exception thrown when layer dimensions are incompatible
 */
class LayerDimensionError : public std::exception {
public:
	const char* what() const noexcept override {
		return "Layer dimensions are incompatible for this operation.";
	}
};

/**
 * Exception thrown when layer input is invalid
 */
class InvalidLayerInputError : public std::exception {
public:
	const char* what() const noexcept override {
		return "Invalid input provided to layer.";
	}
};

/**
 * Abstract base class for neural network layers
 *
 * Defines the interface for forward and backward propagation
 */
class Layer {
public:
	virtual ~Layer() = default;

	/**
	 * Forward pass through the layer
	 *
	 * input: Input tensor
	 * Output: Output tensor after applying layer transformation
	 */
	virtual Tensor forward(const Tensor& input) = 0;

	/**
	 * Backward pass through the layer
	 *
	 * gradOutput: Gradient of loss with respect to output
	 * Output: Gradient of loss with respect to input
	 */
	virtual Tensor backward(const Tensor& gradOutput) = 0;

	/**
	 * Check if layer has trainable parameters
	 *
	 * Output: True if layer has weights/biases
	 */
	virtual bool hasWeights() const { return false; }

	/**
	 * Get pointers to all trainable parameters
	 *
	 * Output: Vector of pointers to weight tensors
	 */
	virtual std::vector<Tensor*> getWeights() { return {}; }

	/**
	 * Get pointers to all parameter gradients
	 *
	 * Output: Vector of pointers to gradient tensors
	 */
	virtual std::vector<Tensor*> getGradients() { return {}; }
};

#endif