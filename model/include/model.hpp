/* model.hpp */

#ifndef MODEL_HPP
#define MODEL_HPP

#include "../../tensor/include/tensor.hpp"
#include <vector>
#include <exception>

/**
 * Exception thrown when model structure is invalid
 */
class InvalidModelError : public std::exception {
public:
	const char* what() const noexcept override {
		return "Invalid model structure or configuration.";
	}
};

/**
 * Exception thrown when accessing invalid layer index
 */
class LayerIndexOutOfRangeError : public std::exception {
public:
	const char* what() const noexcept override {
		return "Layer index out of range.";
	}
};

/**
 * Abstract base class for neural network models
 *
 * Defines the interface for models composed of layers
 * Inspired by PyTorch's nn.Module
 */
class Model {
protected:
	bool training;

public:
	Model() : training(true) {}
	virtual ~Model() = default;

	/**
	 * Forward pass through the model
	 *
	 * input: Input tensor
	 * Output: Model output tensor
	 */
	virtual Tensor forward(const Tensor& input) = 0;

	/**
	 * Get all trainable parameters from the model
	 *
	 * Output: Vector of pointers to all weight tensors
	 */
	virtual std::vector<Tensor*> getParameters() = 0;

	/**
	 * Get all parameter gradients from the model
	 *
	 * Output: Vector of pointers to all gradient tensors
	 */
	virtual std::vector<Tensor*> getGradients() = 0;

	/**
	 * Set model to training mode
	 */
	void train() {
		training = true;
	}

	/**
	 * Set model to evaluation mode
	 */
	void eval() {
		training = false;
	}

	/**
	 * Check if model is in training mode
	 *
	 * Output: True if in training mode
	 */
	bool isTraining() const {
		return training;
	}
};

#endif
