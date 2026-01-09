/* activation.cpp */

#include "../include/activation.hpp"
#include <cmath>
#include <algorithm>

Activation::Activation(ActivationType activationType)
	: type(activationType), inputCache({1}) {}

Tensor Activation::forward(const Tensor& input) {
	inputCache = input;
	Tensor output(input.getShape());

	switch (type) {
		case ActivationType::ReLU:
			for (size_t i = 0; i < input.size(); i++) {
				output.getData()[i] = std::max(0.0, input.getData()[i]);
			}
			break;

		case ActivationType::Sigmoid:
			for (size_t i = 0; i < input.size(); i++) {
				output.getData()[i] = 1.0 / (1.0 + std::exp(-input.getData()[i]));
			}
			break;

		case ActivationType::Tanh:
			for (size_t i = 0; i < input.size(); i++) {
				output.getData()[i] = std::tanh(input.getData()[i]);
			}
			break;

		case ActivationType::Softmax: {
			if (input.ndim() == 1) {
				double maxVal = *std::max_element(input.getData().begin(), input.getData().end());
				double sumExp = 0.0;

				for (size_t i = 0; i < input.size(); i++) {
					output.getData()[i] = std::exp(input.getData()[i] - maxVal);
					sumExp += output.getData()[i];
				}

				for (size_t i = 0; i < input.size(); i++) {
					output.getData()[i] /= sumExp;
				}
			} else if (input.ndim() == 2) {
				size_t batchSize = input.getShape()[0];
				size_t numClasses = input.getShape()[1];

				for (size_t b = 0; b < batchSize; b++) {
					double maxVal = input.get({b, 0});
					for (size_t i = 1; i < numClasses; i++) {
						maxVal = std::max(maxVal, input.get({b, i}));
					}

					double sumExp = 0.0;
					for (size_t i = 0; i < numClasses; i++) {
						output.at({b, i}) = std::exp(input.get({b, i}) - maxVal);
						sumExp += output.get({b, i});
					}

					for (size_t i = 0; i < numClasses; i++) {
						output.at({b, i}) /= sumExp;
					}
				}
			} else {
				throw TensorDismatchError();
			}
			break;
		}
	}

	return output;
}

Tensor Activation::backward(const Tensor& gradOutput) {
	Tensor gradInput(inputCache.getShape());

	switch (type) {
		case ActivationType::ReLU:
			for (size_t i = 0; i < inputCache.size(); i++) {
				gradInput.getData()[i] = inputCache.getData()[i] > 0.0 ? gradOutput.getData()[i] : 0.0;
			}
			break;

		case ActivationType::Sigmoid: {
			Tensor sigmoidOutput = forward(inputCache);
			for (size_t i = 0; i < inputCache.size(); i++) {
				double s = sigmoidOutput.getData()[i];
				gradInput.getData()[i] = gradOutput.getData()[i] * s * (1.0 - s);
			}
			break;
		}

		case ActivationType::Tanh: {
			for (size_t i = 0; i < inputCache.size(); i++) {
				double t = std::tanh(inputCache.getData()[i]);
				gradInput.getData()[i] = gradOutput.getData()[i] * (1.0 - t * t);
			}
			break;
		}

		case ActivationType::Softmax: {
			Tensor softmaxOutput = forward(inputCache);

			if (inputCache.ndim() == 1) {
				size_t n = inputCache.size();
				for (size_t i = 0; i < n; i++) {
					double sum = 0.0;
					for (size_t j = 0; j < n; j++) {
						double delta = (i == j) ? 1.0 : 0.0;
						sum += gradOutput.getData()[j] * softmaxOutput.getData()[i] * (delta - softmaxOutput.getData()[j]);
					}
					gradInput.getData()[i] = sum;
				}
			} else if (inputCache.ndim() == 2) {
				size_t batchSize = inputCache.getShape()[0];
				size_t numClasses = inputCache.getShape()[1];

				for (size_t b = 0; b < batchSize; b++) {
					for (size_t i = 0; i < numClasses; i++) {
						double sum = 0.0;
						for (size_t j = 0; j < numClasses; j++) {
							double delta = (i == j) ? 1.0 : 0.0;
							sum += gradOutput.get({b, j}) * softmaxOutput.get({b, i}) * (delta - softmaxOutput.get({b, j}));
						}
						gradInput.at({b, i}) = sum;
					}
				}
			} else {
				throw LayerDimensionError();
			}
			break;
		}
	}

	return gradInput;
}
