#include "dense.hpp"
#include "activation.hpp"
#include <cassert>
#include <cstdio>
#include <cmath>

void testDenseForward() {
	Dense layer(3, 2);

	Tensor input({3});
	input.at({0}) = 1.0;
	input.at({1}) = 2.0;
	input.at({2}) = 3.0;

	Tensor output = layer.forward(input);

	assert(output.ndim() == 1);
	assert(output.getShape()[0] == 2);
	std::printf("Dense forward (1D) passed.\n");

	Tensor batchInput({2, 3});
	for (size_t i = 0; i < 2; i++) {
		for (size_t j = 0; j < 3; j++) {
			batchInput.at({i, j}) = i * 3.0 + j + 1.0;
		}
	}

	Tensor batchOutput = layer.forward(batchInput);
	assert(batchOutput.ndim() == 2);
	assert(batchOutput.getShape()[0] == 2);
	assert(batchOutput.getShape()[1] == 2);
	std::printf("Dense forward (2D batch) passed.\n");
}

void testDenseBackward() {
	Dense layer(3, 2);

	Tensor input({3});
	input.at({0}) = 1.0;
	input.at({1}) = 2.0;
	input.at({2}) = 3.0;

	Tensor output = layer.forward(input);

	Tensor gradOutput({2});
	gradOutput.at({0}) = 0.5;
	gradOutput.at({1}) = 0.3;

	Tensor gradInput = layer.backward(gradOutput);

	assert(gradInput.ndim() == 1);
	assert(gradInput.getShape()[0] == 3);

	std::vector<Tensor*> grads = layer.getGradients();
	assert(grads.size() == 2);
	assert(grads[0]->getShape()[0] == 2);
	assert(grads[0]->getShape()[1] == 3);
	assert(grads[1]->getShape()[0] == 2);

	std::printf("Dense backward passed.\n");
}

void testReLU() {
	Activation relu(ActivationType::ReLU);

	Tensor input({4});
	input.at({0}) = -2.0;
	input.at({1}) = -0.5;
	input.at({2}) = 0.5;
	input.at({3}) = 2.0;

	Tensor output = relu.forward(input);

	assert(output.get({0}) == 0.0);
	assert(output.get({1}) == 0.0);
	assert(output.get({2}) == 0.5);
	assert(output.get({3}) == 2.0);

	Tensor gradOutput({4});
	gradOutput.fill(1.0);

	Tensor gradInput = relu.backward(gradOutput);
	assert(gradInput.get({0}) == 0.0);
	assert(gradInput.get({1}) == 0.0);
	assert(gradInput.get({2}) == 1.0);
	assert(gradInput.get({3}) == 1.0);

	std::printf("ReLU activation passed.\n");
}

void testSigmoid() {
	Activation sigmoid(ActivationType::Sigmoid);

	Tensor input({3});
	input.at({0}) = -1.0;
	input.at({1}) = 0.0;
	input.at({2}) = 1.0;

	Tensor output = sigmoid.forward(input);

	assert(std::abs(output.get({0}) - 0.2689) < 0.001);
	assert(std::abs(output.get({1}) - 0.5) < 0.001);
	assert(std::abs(output.get({2}) - 0.7311) < 0.001);

	std::printf("Sigmoid activation passed.\n");
}

void testSoftmax() {
	Activation softmax(ActivationType::Softmax);

	Tensor input({3});
	input.at({0}) = 1.0;
	input.at({1}) = 2.0;
	input.at({2}) = 3.0;

	Tensor output = softmax.forward(input);

	double sum = output.get({0}) + output.get({1}) + output.get({2});
	assert(std::abs(sum - 1.0) < 0.0001);
	assert(output.get({2}) > output.get({1}));
	assert(output.get({1}) > output.get({0}));

	std::printf("Softmax activation passed.\n");
}

void testLayerInterface() {
	Dense dense(4, 3);
	Activation relu(ActivationType::ReLU);

	assert(dense.hasWeights() == true);
	assert(relu.hasWeights() == false);

	assert(dense.getWeights().size() == 2);
	assert(dense.getGradients().size() == 2);
	assert(relu.getWeights().size() == 0);
	assert(relu.getGradients().size() == 0);

	std::printf("Layer interface passed.\n");
}

int main(void) {
	testDenseForward();
	testDenseBackward();
	testReLU();
	testSigmoid();
	testSoftmax();
	testLayerInterface();

	std::printf("\nAll layer tests passed successfully.\n");
	return 0;
}