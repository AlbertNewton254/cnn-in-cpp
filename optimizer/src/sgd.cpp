/* sgd.cpp */

#include "../include/sgd.hpp"

SGD::SGD(double lr) : learningRate(lr) {}

void SGD::step(std::vector<Tensor*>& parameters, std::vector<Tensor*>& gradients) {
	if (parameters.size() != gradients.size()) {
		throw OptimizerSizeMismatchError();
	}

	for (size_t i = 0; i < parameters.size(); i++) {
		Tensor* param = parameters[i];
		Tensor* grad = gradients[i];

		if (param->size() != grad->size()) {
			throw OptimizerSizeMismatchError();
		}

		for (size_t j = 0; j < param->size(); j++) {
			param->getData()[j] -= learningRate * grad->getData()[j];
		}
	}
}

double SGD::getLearningRate() const {
	return learningRate;
}

void SGD::setLearningRate(double lr) {
	learningRate = lr;
}