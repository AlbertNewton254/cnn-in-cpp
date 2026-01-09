#ifndef SGD_HPP
#define SGD_HPP

#include "optimizer.hpp"

/**
 * Stochastic Gradient Descent optimizer
 *
 * learningRate: Step size for parameter updates
 */
class SGD : public Optimizer {
private:
	double learningRate;

public:
	/**
	 * Create SGD optimizer with given learning rate
	 *
	 * lr: Learning rate (default 0.01)
	 */
	SGD(double lr = 0.01);

	/**
	 * Perform SGD update: param = param - learningRate * grad
	 *
	 * parameters: Vector of pointers to parameter tensors
	 * gradients: Vector of pointers to gradient tensors
	 */
	void step(std::vector<Tensor*>& parameters, std::vector<Tensor*>& gradients) override;

	/**
	 * Get current learning rate
	 *
	 * Output: Learning rate value
	 */
	double getLearningRate() const;

	/**
	 * Set new learning rate
	 *
	 * lr: New learning rate value
	 */
	void setLearningRate(double lr);
};

#endif