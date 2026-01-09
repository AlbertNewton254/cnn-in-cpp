/* tensor.cpp */

#include "../include/tensor.hpp"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <numeric>

size_t Tensor::computeIndex(const std::vector<size_t>& indices) const {
	if (indices.size() != shape.size()) {
		throw TensorDismatchError();
	}

	size_t index = 0;
	size_t stride = 1;

	for (int i = shape.size() - 1; i >= 0; i--) {
		if (indices[i] >= shape[i]) {
			throw IndexOutOfBoundsError();
		}
		index += indices[i] * stride;
		stride *= shape[i];
	}

	return index;
}

Tensor::Tensor(const std::vector<size_t>& shape) : shape(shape) {
	size_t totalSize = 1;
	for (size_t dim : shape) {
		totalSize *= dim;
	}
	data.resize(totalSize, 0.0);
}

Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<double>& values)
	: shape(shape), data(values) {
	size_t totalSize = 1;
	for (size_t dim : shape) {
		totalSize *= dim;
	}
	if (values.size() != totalSize) {
		throw TensorDismatchError();
	}
}

Tensor::Tensor(const std::vector<size_t>& shape, double fillValue) : shape(shape) {
	size_t totalSize = 1;
	for (size_t dim : shape) {
		totalSize *= dim;
	}
	data.resize(totalSize, fillValue);
}

const std::vector<size_t>& Tensor::getShape() const {
	return shape;
}

size_t Tensor::ndim() const {
	return shape.size();
}

size_t Tensor::size() const {
	return data.size();
}

double Tensor::get(const std::vector<size_t>& indices) const {
	return data[computeIndex(indices)];
}

double& Tensor::at(const std::vector<size_t>& indices) {
	return data[computeIndex(indices)];
}

const std::vector<double>& Tensor::getData() const {
	return data;
}

std::vector<double>& Tensor::getData() {
	return data;
}

Tensor Tensor::zeros(const std::vector<size_t>& shape) {
	return Tensor(shape, 0.0);
}

Tensor Tensor::ones(const std::vector<size_t>& shape) {
	return Tensor(shape, 1.0);
}

Tensor Tensor::random(const std::vector<size_t>& shape) {
	Tensor result(shape);
	std::srand(static_cast<unsigned int>(std::time(nullptr)));
	for (size_t i = 0; i < result.data.size(); i++) {
		result.data[i] = static_cast<double>(std::rand()) / RAND_MAX;
	}
	return result;
}

Tensor Tensor::reshape(const std::vector<size_t>& newShape) const {
	size_t newSize = 1;
	for (size_t dim : newShape) {
		newSize *= dim;
	}

	if (newSize != data.size()) {
		throw TensorDismatchError();
	}

	return Tensor(newShape, data);
}

Tensor Tensor::flatten() const {
	return Tensor({data.size()}, data);
}

Tensor Tensor::operator+(const Tensor& other) const {
	if (shape != other.shape) {
		throw TensorDismatchError();
	}

	Tensor result(shape);
	for (size_t i = 0; i < data.size(); i++) {
		result.data[i] = data[i] + other.data[i];
	}
	return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
	if (shape != other.shape) {
		throw TensorDismatchError();
	}

	Tensor result(shape);
	for (size_t i = 0; i < data.size(); i++) {
		result.data[i] = data[i] - other.data[i];
	}
	return result;
}

Tensor Tensor::operator*(double scalar) const {
	Tensor result(shape);
	for (size_t i = 0; i < data.size(); i++) {
		result.data[i] = data[i] * scalar;
	}
	return result;
}

Tensor Tensor::hadamard(const Tensor& other) const {
	if (shape != other.shape) {
		throw TensorDismatchError();
	}

	Tensor result(shape);
	for (size_t i = 0; i < data.size(); i++) {
		result.data[i] = data[i] * other.data[i];
	}
	return result;
}

Tensor Tensor::matmul(const Tensor& other) const {
	if (shape.size() != 2 || other.shape.size() != 2) {
		throw TensorDismatchError();
	}

	size_t rows1 = shape[0];
	size_t cols1 = shape[1];
	size_t rows2 = other.shape[0];
	size_t cols2 = other.shape[1];

	if (cols1 != rows2) {
		throw TensorDismatchError();
	}

	Tensor result({rows1, cols2});
	for (size_t i = 0; i < rows1; i++) {
		for (size_t j = 0; j < cols2; j++) {
			double sum = 0.0;
			for (size_t k = 0; k < cols1; k++) {
				sum += get({i, k}) * other.get({k, j});
			}
			result.at({i, j}) = sum;
		}
	}
	return result;
}

Tensor Tensor::transpose() const {
	if (shape.size() != 2) {
		throw TensorDismatchError();
	}

	Tensor result({shape[1], shape[0]});
	for (size_t i = 0; i < shape[0]; i++) {
		for (size_t j = 0; j < shape[1]; j++) {
			result.at({j, i}) = get({i, j});
		}
	}
	return result;
}

void Tensor::fill(double value) {
	for (size_t i = 0; i < data.size(); i++) {
		data[i] = value;
	}
}