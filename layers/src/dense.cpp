/* dense.cpp */

#include "../include/dense.hpp"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cassert>

Dense::Dense(size_t inputSize, size_t outputSize)
	: weights({outputSize, inputSize}),
	  biases({outputSize}),
	  weightGrad({outputSize, inputSize}),
	  biasGrad({outputSize}),
	  inputCache({1}) {

	std::srand(static_cast<unsigned int>(std::time(nullptr)));
	double limit = std::sqrt(6.0 / (inputSize + outputSize));

	double* weightsData = weights.getData().data();
	for (size_t i = 0; i < weights.size(); i++) {
		weightsData[i] = ((double)std::rand() / RAND_MAX) * 2 * limit - limit;
	}

	biases.fill(0.0);
}

Tensor Dense::forward(const Tensor& input) {
	inputCache = input;

	if (input.ndim() == 1) {
		size_t outputSize = biases.getShape()[0];
		size_t inputSize = input.getShape()[0];
		assert(weights.getShape()[0] == outputSize);
		assert(weights.getShape()[1] == inputSize);

		Tensor result({outputSize});
		double* resultData = result.getData().data();
		const double* inputData = input.getData().data();
		const double* weightsData = weights.getData().data();
		const double* biasData = biases.getData().data();

		for (size_t i = 0; i < outputSize; i++) {
			double sum = biasData[i];
			for (size_t j = 0; j < inputSize; j++) {
				sum += weightsData[i * inputSize + j] * inputData[j];
			}
			resultData[i] = sum;
		}
		return result;
	} else if (input.ndim() == 2) {
		assert(input.getShape()[1] == weights.getShape()[1]);

		Tensor inputT = input.transpose();
		Tensor outputT = weights.matmul(inputT);
		Tensor output = outputT.transpose();

		double* outputData = output.getData().data();
		const double* biasData = biases.getData().data();
		size_t batchSize = output.getShape()[0];
		size_t outputSize = output.getShape()[1];

		for (size_t b = 0; b < batchSize; b++) {
			for (size_t i = 0; i < outputSize; i++) {
				outputData[b * outputSize + i] += biasData[i];
			}
		}
		return output;
	} else {
		throw LayerDimensionError();
	}
}

Tensor Dense::backward(const Tensor& gradOutput) {
	if (inputCache.ndim() == 1) {
		size_t outputSize = weightGrad.getShape()[0];
		size_t inputSize = weightGrad.getShape()[1];
		assert(gradOutput.getShape()[0] == outputSize);
		assert(inputCache.getShape()[0] == inputSize);

		double* weightGradData = weightGrad.getData().data();
		const double* gradOutData = gradOutput.getData().data();
		const double* inputData = inputCache.getData().data();

		for (size_t i = 0; i < outputSize; i++) {
			for (size_t j = 0; j < inputSize; j++) {
				weightGradData[i * inputSize + j] = gradOutData[i] * inputData[j];
			}
		}

		double* biasGradData = biasGrad.getData().data();
		for (size_t i = 0; i < outputSize; i++) {
			biasGradData[i] = gradOutData[i];
		}

		Tensor gradInput({inputSize});
		double* gradInputData = gradInput.getData().data();
		const double* weightsData = weights.getData().data();

		for (size_t j = 0; j < inputSize; j++) {
			double sum = 0.0;
			for (size_t i = 0; i < outputSize; i++) {
				sum += weightsData[i * inputSize + j] * gradOutData[i];
			}
			gradInputData[j] = sum;
		}
		return gradInput;
	} else if (inputCache.ndim() == 2) {
		assert(gradOutput.getShape()[1] == weightGrad.getShape()[0]);
		assert(inputCache.getShape()[1] == weightGrad.getShape()[1]);

		weightGrad = gradOutput.transpose().matmul(inputCache);

		double* biasGradData = biasGrad.getData().data();
		const double* gradOutData = gradOutput.getData().data();
		size_t batchSize = gradOutput.getShape()[0];
		size_t outputSize = biasGrad.getShape()[0];

		for (size_t i = 0; i < outputSize; i++) {
			double sum = 0.0;
			for (size_t b = 0; b < batchSize; b++) {
				sum += gradOutData[b * outputSize + i];
			}
			biasGradData[i] = sum;
		}

		Tensor weightsT = weights.transpose();
		Tensor gradInput = gradOutput.matmul(weightsT);
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