#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "../../tensor/include/tensor.hpp"
#include <vector>
#include <exception>

/**
 * Exception thrown when parameters and gradients size mismatch
 */
class OptimizerSizeMismatchError : public std::exception {
public:
	const char* what() const noexcept override {
		return "Parameters and gradients size mismatch.";
	}
};

/**
 * Abstract base class for optimization algorithms
 */
class Optimizer {
public:
	virtual ~Optimizer() = default;

	/**
	 * Perform a single optimization step
	 *
	 * parameters: Vector of pointers to parameter tensors
	 * gradients: Vector of pointers to gradient tensors
	 */
	virtual void step(std::vector<Tensor*>& parameters, std::vector<Tensor*>& gradients) = 0;

	/**
	 * Zero out all gradients
	 *
	 * gradients: Vector of pointers to gradient tensors
	 */
	virtual void zeroGrad(std::vector<Tensor*>& gradients);
};

#endif
