#include "sequential.hpp"
#include "dense.hpp"
#include "activation.hpp"
#include <cassert>
#include <cstdio>
#include <memory>
#include <cmath>

void testSequentialCreation() {
	Sequential model;

	assert(model.numLayers() == 0);
	assert(model.isTraining() == true);

	std::printf("Sequential creation passed.\n");
}

void testAddLayers() {
	Sequential model;

	model.addLayer(std::make_shared<Dense>(4, 8));
	model.addLayer(std::make_shared<Activation>(ActivationType::ReLU));
	model.addLayer(std::make_shared<Dense>(8, 2));

	assert(model.numLayers() == 3);
	std::printf("Add layers passed.\n");
}

void testSequentialForward() {
	Sequential model;

	model.addLayer(std::make_shared<Dense>(3, 5));
	model.addLayer(std::make_shared<Activation>(ActivationType::ReLU));
	model.addLayer(std::make_shared<Dense>(5, 2));

	Tensor input({3});
	input.at({0}) = 1.0;
	input.at({1}) = 2.0;
	input.at({2}) = 3.0;

	Tensor output = model.forward(input);

	assert(output.ndim() == 1);
	assert(output.getShape()[0] == 2);
	std::printf("Sequential forward passed.\n");
}

void testSequentialBackward() {
	Sequential model;

	model.addLayer(std::make_shared<Dense>(3, 4));
	model.addLayer(std::make_shared<Activation>(ActivationType::ReLU));
	model.addLayer(std::make_shared<Dense>(4, 2));

	Tensor input({3});
	input.at({0}) = 1.0;
	input.at({1}) = 2.0;
	input.at({2}) = 3.0;

	Tensor output = model.forward(input);

	Tensor gradOutput({2});
	gradOutput.at({0}) = 0.5;
	gradOutput.at({1}) = 0.3;

	Tensor gradInput = model.backward(gradOutput);

	assert(gradInput.ndim() == 1);
	assert(gradInput.getShape()[0] == 3);
	std::printf("Sequential backward passed.\n");
}

void testGetParameters() {
	Sequential model;

	model.addLayer(std::make_shared<Dense>(3, 4));
	model.addLayer(std::make_shared<Activation>(ActivationType::ReLU));
	model.addLayer(std::make_shared<Dense>(4, 2));
	model.addLayer(std::make_shared<Activation>(ActivationType::Softmax));

	std::vector<Tensor*> params = model.getParameters();

	assert(params.size() == 4);
	std::printf("Get parameters passed.\n");
}

void testGetGradients() {
	Sequential model;

	model.addLayer(std::make_shared<Dense>(3, 4));
	model.addLayer(std::make_shared<Activation>(ActivationType::ReLU));
	model.addLayer(std::make_shared<Dense>(4, 2));

	Tensor input({3});
	input.fill(1.0);

	Tensor output = model.forward(input);

	Tensor gradOutput({2});
	gradOutput.fill(1.0);

	model.backward(gradOutput);

	std::vector<Tensor*> grads = model.getGradients();

	assert(grads.size() == 4);
	std::printf("Get gradients passed.\n");
}

void testTrainEvalMode() {
	Sequential model;

	assert(model.isTraining() == true);

	model.eval();
	assert(model.isTraining() == false);

	model.train();
	assert(model.isTraining() == true);

	std::printf("Train/eval mode passed.\n");
}

void testBatchedForward() {
	Sequential model;

	model.addLayer(std::make_shared<Dense>(3, 5));
	model.addLayer(std::make_shared<Activation>(ActivationType::ReLU));
	model.addLayer(std::make_shared<Dense>(5, 2));

	Tensor batchInput({4, 3});
	for (size_t i = 0; i < 4; i++) {
		for (size_t j = 0; j < 3; j++) {
			batchInput.at({i, j}) = i * 3.0 + j + 1.0;
		}
	}

	Tensor output = model.forward(batchInput);

	assert(output.ndim() == 2);
	assert(output.getShape()[0] == 4);
	assert(output.getShape()[1] == 2);
	std::printf("Batched forward passed.\n");
}

void testMLPExample() {
	Sequential mlp;

	mlp.addLayer(std::make_shared<Dense>(784, 128));
	mlp.addLayer(std::make_shared<Activation>(ActivationType::ReLU));
	mlp.addLayer(std::make_shared<Dense>(128, 64));
	mlp.addLayer(std::make_shared<Activation>(ActivationType::ReLU));
	mlp.addLayer(std::make_shared<Dense>(64, 10));
	mlp.addLayer(std::make_shared<Activation>(ActivationType::Softmax));

	assert(mlp.numLayers() == 6);

	Tensor input({784});
	input.fill(0.5);

	Tensor output = mlp.forward(input);

	assert(output.ndim() == 1);
	assert(output.getShape()[0] == 10);

	double sum = 0.0;
	for (size_t i = 0; i < 10; i++) {
		sum += output.get({i});
	}
	assert(std::abs(sum - 1.0) < 0.0001);

	std::printf("MLP example passed.\n");
}

int main(void) {
	testSequentialCreation();
	testAddLayers();
	testSequentialForward();
	testSequentialBackward();
	testGetParameters();
	testGetGradients();
	testTrainEvalMode();
	testBatchedForward();
	testMLPExample();

	std::printf("\nAll model tests passed successfully.\n");
	return 0;
}