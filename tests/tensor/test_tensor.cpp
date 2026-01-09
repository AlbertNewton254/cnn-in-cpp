#include "../../tensor/include/tensor.hpp"
#include <cassert>
#include <cstdio>

int main(void) {
	// Test 2D tensor creation
	Tensor A({2, 3});
	assert(A.getShape()[0] == 2);
	assert(A.getShape()[1] == 3);
	assert(A.ndim() == 2);

	// Test initialization and indexing
	for (size_t i = 0; i < A.getShape()[0]; i++) {
		for (size_t j = 0; j < A.getShape()[1]; j++) {
			assert(A.get({i, j}) == 0.0);
			A.at({i, j}) = static_cast<double>(i * A.getShape()[1] + j + 1);
		}
	}

	assert(A.get({0, 0}) == 1.0);
	assert(A.get({0, 1}) == 2.0);
	assert(A.get({0, 2}) == 3.0);
	assert(A.get({1, 0}) == 4.0);
	assert(A.get({1, 1}) == 5.0);
	assert(A.get({1, 2}) == 6.0);
	std::printf("Tensor A initialized successfully.\n");

	// Test ones factory method
	Tensor B = Tensor::ones({2, 3});

	assert(B.get({0, 0}) == 1.0);
	assert(B.get({0, 1}) == 1.0);
	assert(B.get({0, 2}) == 1.0);
	assert(B.get({1, 0}) == 1.0);
	assert(B.get({1, 1}) == 1.0);
	assert(B.get({1, 2}) == 1.0);
	std::printf("Tensor B initialized successfully.\n");

	// Test element-wise addition
	Tensor C = A + B;

	assert(C.get({0, 0}) == 2.0);
	assert(C.get({0, 1}) == 3.0);
	assert(C.get({0, 2}) == 4.0);
	assert(C.get({1, 0}) == 5.0);
	assert(C.get({1, 1}) == 6.0);
	assert(C.get({1, 2}) == 7.0);
	std::printf("Tensor C (A + B) computed successfully.\n");

	// Test transpose
	Tensor D = B.transpose();

	assert(D.getShape()[0] == 3);
	assert(D.getShape()[1] == 2);
	std::printf("Tensor D (B transposed) has correct dimensions.\n");

	assert(D.get({0, 0}) == 1.0);
	assert(D.get({0, 1}) == 1.0);
	assert(D.get({1, 0}) == 1.0);
	assert(D.get({1, 1}) == 1.0);
	assert(D.get({2, 0}) == 1.0);
	assert(D.get({2, 1}) == 1.0);
	std::printf("Tensor D (B transposed) values are correct.\n");

	// Test matrix multiplication
	Tensor E = A.matmul(D);

	assert(E.getShape()[0] == 2);
	assert(E.getShape()[1] == 2);
	std::printf("Tensor E (A * D) has correct dimensions.\n");

	assert(E.get({0, 0}) == 6.0);
	assert(E.get({0, 1}) == 6.0);
	assert(E.get({1, 0}) == 15.0);
	assert(E.get({1, 1}) == 15.0);
	std::printf("Tensor E (A * D) values are correct.\n");

	// Test 3D tensor
	Tensor F = Tensor::zeros({2, 3, 4});
	assert(F.ndim() == 3);
	assert(F.getShape()[0] == 2);
	assert(F.getShape()[1] == 3);
	assert(F.getShape()[2] == 4);
	assert(F.size() == 24);
	std::printf("Tensor F (3D) created successfully.\n");

	// Test reshape
	Tensor G = A.reshape({6});
	assert(G.ndim() == 1);
	assert(G.getShape()[0] == 6);
	assert(G.get({0}) == 1.0);
	assert(G.get({5}) == 6.0);
	std::printf("Tensor G (reshaped) created successfully.\n");

	// Test flatten
	Tensor H = A.flatten();
	assert(H.ndim() == 1);
	assert(H.size() == 6);
	std::printf("Tensor H (flattened) created successfully.\n");

	std::printf("All tests passed successfully.\n");

	return 0;
}