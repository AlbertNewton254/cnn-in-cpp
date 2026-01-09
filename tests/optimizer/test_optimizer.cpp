#include "sgd.hpp"
#include "optimizer.hpp"
#include <cassert>
#include <cstdio>
#include <cmath>

void testSGDCreation() {
	SGD optimizer(0.01);

	assert(optimizer.getLearningRate() == 0.01);
	std::printf("SGD creation passed.\n");
}

void testSGDSetLearningRate() {
	SGD optimizer(0.01);

	optimizer.setLearningRate(0.1);
	assert(optimizer.getLearningRate() == 0.1);

	optimizer.setLearningRate(0.001);
	assert(optimizer.getLearningRate() == 0.001);

	std::printf("SGD set learning rate passed.\n");
}

void testSGDStep() {
	SGD optimizer(0.1);

	Tensor param1({3});
	param1.at({0}) = 1.0;
	param1.at({1}) = 2.0;
	param1.at({2}) = 3.0;

	Tensor param2({2});
	param2.at({0}) = 5.0;
	param2.at({1}) = 6.0;

	Tensor grad1({3});
	grad1.at({0}) = 0.1;
	grad1.at({1}) = 0.2;
	grad1.at({2}) = 0.3;

	Tensor grad2({2});
	grad2.at({0}) = 0.5;
	grad2.at({1}) = 0.6;

	std::vector<Tensor*> parameters = {&param1, &param2};
	std::vector<Tensor*> gradients = {&grad1, &grad2};

	optimizer.step(parameters, gradients);

	assert(std::abs(param1.get({0}) - 0.99) < 1e-6);
	assert(std::abs(param1.get({1}) - 1.98) < 1e-6);
	assert(std::abs(param1.get({2}) - 2.97) < 1e-6);
	assert(std::abs(param2.get({0}) - 4.95) < 1e-6);
	assert(std::abs(param2.get({1}) - 5.94) < 1e-6);

	std::printf("SGD step passed.\n");
}

void testSGDMultipleSteps() {
	SGD optimizer(0.1);

	Tensor param({2});
	param.at({0}) = 1.0;
	param.at({1}) = 2.0;

	Tensor grad({2});
	grad.at({0}) = 0.1;
	grad.at({1}) = 0.2;

	std::vector<Tensor*> parameters = {&param};
	std::vector<Tensor*> gradients = {&grad};

	for (int i = 0; i < 5; i++) {
		optimizer.step(parameters, gradients);
	}

	assert(std::abs(param.get({0}) - 0.95) < 1e-6);
	assert(std::abs(param.get({1}) - 1.90) < 1e-6);

	std::printf("SGD multiple steps passed.\n");
}

void testZeroGrad() {
	SGD optimizer(0.1);

	Tensor grad1({3});
	grad1.at({0}) = 1.0;
	grad1.at({1}) = 2.0;
	grad1.at({2}) = 3.0;

	Tensor grad2({2});
	grad2.at({0}) = 4.0;
	grad2.at({1}) = 5.0;

	std::vector<Tensor*> gradients = {&grad1, &grad2};

	optimizer.zeroGrad(gradients);

	assert(grad1.get({0}) == 0.0);
	assert(grad1.get({1}) == 0.0);
	assert(grad1.get({2}) == 0.0);
	assert(grad2.get({0}) == 0.0);
	assert(grad2.get({1}) == 0.0);

	std::printf("Zero grad passed.\n");
}

void testOptimizerSizeMismatch() {
	SGD optimizer(0.1);

	Tensor param1({3});
	param1.fill(1.0);

	Tensor grad1({3});
	grad1.fill(0.1);

	Tensor grad2({2});
	grad2.fill(0.2);

	std::vector<Tensor*> parameters = {&param1};
	std::vector<Tensor*> gradients = {&grad1, &grad2};

	bool exceptionThrown = false;
	try {
		optimizer.step(parameters, gradients);
	} catch (const OptimizerSizeMismatchError& e) {
		exceptionThrown = true;
	}

	assert(exceptionThrown);
	std::printf("Optimizer size mismatch detection passed.\n");
}

void testParameterGradientSizeMismatch() {
	SGD optimizer(0.1);

	Tensor param1({3});
	param1.fill(1.0);

	Tensor grad1({2});
	grad1.fill(0.1);

	std::vector<Tensor*> parameters = {&param1};
	std::vector<Tensor*> gradients = {&grad1};

	bool exceptionThrown = false;
	try {
		optimizer.step(parameters, gradients);
	} catch (const OptimizerSizeMismatchError& e) {
		exceptionThrown = true;
	}

	assert(exceptionThrown);
	std::printf("Parameter-gradient size mismatch detection passed.\n");
}

int main(void) {
	testSGDCreation();
	testSGDSetLearningRate();
	testSGDStep();
	testSGDMultipleSteps();
	testZeroGrad();
	testOptimizerSizeMismatch();
	testParameterGradientSizeMismatch();

	std::printf("\nAll optimizer tests passed!\n");
	return 0;
}