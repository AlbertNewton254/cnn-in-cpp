#include "mse.hpp"
#include <cassert>
#include <cstdio>
#include <cmath>

void testMSEForward() {
	MSE loss;

	Tensor predictions({4});
	predictions.at({0}) = 1.0;
	predictions.at({1}) = 2.0;
	predictions.at({2}) = 3.0;
	predictions.at({3}) = 4.0;

	Tensor targets({4});
	targets.at({0}) = 1.5;
	targets.at({1}) = 2.5;
	targets.at({2}) = 2.5;
	targets.at({3}) = 3.5;

	Tensor lossValue = loss.forward(predictions, targets);

	assert(lossValue.ndim() == 1);
	assert(lossValue.getShape()[0] == 1);

	double expected = (0.25 + 0.25 + 0.25 + 0.25) / 4.0;
	assert(std::abs(lossValue.get({0}) - expected) < 1e-6);

	std::printf("MSE forward passed.\n");
}

void testMSEBackward() {
	MSE loss;

	Tensor predictions({4});
	predictions.at({0}) = 1.0;
	predictions.at({1}) = 2.0;
	predictions.at({2}) = 3.0;
	predictions.at({3}) = 4.0;

	Tensor targets({4});
	targets.at({0}) = 1.5;
	targets.at({1}) = 2.5;
	targets.at({2}) = 2.5;
	targets.at({3}) = 3.5;

	Tensor gradient = loss.backward(predictions, targets);

	assert(gradient.ndim() == 1);
	assert(gradient.getShape()[0] == 4);

	double scale = 2.0 / 4.0;
	assert(std::abs(gradient.get({0}) - (-0.5 * scale)) < 1e-6);
	assert(std::abs(gradient.get({1}) - (-0.5 * scale)) < 1e-6);
	assert(std::abs(gradient.get({2}) - (0.5 * scale)) < 1e-6);
	assert(std::abs(gradient.get({3}) - (0.5 * scale)) < 1e-6);

	std::printf("MSE backward passed.\n");
}

void testMSEShapeMismatch() {
	MSE loss;

	Tensor predictions({4});
	predictions.fill(1.0);

	Tensor targets({3});
	targets.fill(1.0);

	bool exceptionThrown = false;
	try {
		loss.forward(predictions, targets);
	} catch (const LossShapeMismatchError& e) {
		exceptionThrown = true;
	}

	assert(exceptionThrown);
	std::printf("MSE shape mismatch detection passed.\n");
}

void testMSEBatch() {
	MSE loss;

	Tensor predictions({2, 3});
	predictions.at({0, 0}) = 1.0;
	predictions.at({0, 1}) = 2.0;
	predictions.at({0, 2}) = 3.0;
	predictions.at({1, 0}) = 4.0;
	predictions.at({1, 1}) = 5.0;
	predictions.at({1, 2}) = 6.0;

	Tensor targets({2, 3});
	targets.at({0, 0}) = 1.0;
	targets.at({0, 1}) = 2.0;
	targets.at({0, 2}) = 3.0;
	targets.at({1, 0}) = 4.0;
	targets.at({1, 1}) = 5.0;
	targets.at({1, 2}) = 6.0;

	Tensor lossValue = loss.forward(predictions, targets);

	assert(lossValue.ndim() == 1);
	assert(lossValue.get({0}) == 0.0);

	Tensor gradient = loss.backward(predictions, targets);
	assert(gradient.getShape()[0] == 2);
	assert(gradient.getShape()[1] == 3);

	for (size_t i = 0; i < 2; i++) {
		for (size_t j = 0; j < 3; j++) {
			assert(gradient.get({i, j}) == 0.0);
		}
	}

	std::printf("MSE batch processing passed.\n");
}

int main(void) {
	testMSEForward();
	testMSEBackward();
	testMSEShapeMismatch();
	testMSEBatch();

	std::printf("\nAll loss tests passed!\n");
	return 0;
}
