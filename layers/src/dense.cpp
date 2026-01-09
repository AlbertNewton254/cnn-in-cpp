/* dense.cpp */

#include "../include/dense.hpp"
#include <cmath>
#include <cstdlib>
#include <ctime>

Dense::Dense(size_t inputSize, size_t outputSize)
	: weights({outputSize, inputSize}),
	  biases({outputSize}),
	  weightGrad({outputSize, inputSize}),
	  biasGrad({outputSize}),
	  inputCache({1}) {

	std::srand(static_cast<unsigned int>(std::time(nullptr)));
	double limit = std::sqrt(6.0 / (inputSize + outputSize));

	for (size_t i = 0; i < weights.size(); i++) {
		weights.getData()[i] = ((double)std::rand() / RAND_MAX) * 2 * limit - limit;
	}

	biases.fill(0.0);
}

Tensor Dense::forward(const Tensor& input) {
	inputCache = input;

	if (input.ndim() == 1) {
		Tensor inputReshaped({1, input.getShape()[0]});
		for (size_t i = 0; i < input.size(); i++) {
			inputReshaped.getData()[i] = input.getData()[i];
		}

		Tensor output = inputReshaped.matmul(weights.transpose());
		Tensor result({biases.getShape()[0]});
		for (size_t i = 0; i < biases.size(); i++) {
			result.at({i}) = output.get({0, i}) + biases.get({i});
		}
		return result;
	} else if (input.ndim() == 2) {
		Tensor output = input.matmul(weights.transpose());
		for (size_t b = 0; b < output.getShape()[0]; b++) {
			for (size_t i = 0; i < output.getShape()[1]; i++) {
				output.at({b, i}) += biases.get({i});
			}
		}
		return output;
	} else {
		throw LayerDimensionError();
	}
}

Tensor Dense::backward(const Tensor& gradOutput) {
	if (inputCache.ndim() == 1) {
		for (size_t i = 0; i < weightGrad.getShape()[0]; i++) {
			for (size_t j = 0; j < weightGrad.getShape()[1]; j++) {
				weightGrad.at({i, j}) = gradOutput.get({i}) * inputCache.get({j});
			}
		}

		for (size_t i = 0; i < biasGrad.getShape()[0]; i++) {
			biasGrad.at({i}) = gradOutput.get({i});
		}

		Tensor gradOutputReshaped({1, gradOutput.getShape()[0]});
		for (size_t i = 0; i < gradOutput.size(); i++) {
			gradOutputReshaped.getData()[i] = gradOutput.getData()[i];
		}

		Tensor gradInputBatch = gradOutputReshaped.matmul(weights);
		Tensor gradInput({inputCache.getShape()[0]});
		for (size_t i = 0; i < gradInput.size(); i++) {
			gradInput.at({i}) = gradInputBatch.get({0, i});
		}
		return gradInput;
	} else if (inputCache.ndim() == 2) {
		weightGrad = gradOutput.transpose().matmul(inputCache);

		biasGrad.fill(0.0);
		for (size_t b = 0; b < gradOutput.getShape()[0]; b++) {
			for (size_t i = 0; i < biasGrad.getShape()[0]; i++) {
				biasGrad.at({i}) += gradOutput.get({b, i});
			}
		}

		Tensor gradInput = gradOutput.matmul(weights);
		return gradInput;
	} else {
		throw LayerDimensionError();
	}
}

std::vector<Tensor*> Dense::getWeights() {
	return {&weights, &biases};
}

std::vector<Tensor*> Dense::getGradients() {
	return {&weightGrad, &biasGrad};
}
