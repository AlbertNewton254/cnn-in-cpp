/* tensor.hpp */

#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <exception>

/**
 * Exception thrown when tensor dimensions do not match for an operation
 */
class TensorDismatchError : public std::exception {
public:
	const char* what() const noexcept override {
		return "Tensor dimensions do not match for this operation.";
	}
};

/**
 * Exception thrown when accessing tensor with out-of-bounds indices
 */
class IndexOutOfBoundsError : public std::exception {
public:
	const char* what() const noexcept override {
		return "Index out of bounds.";
	}
};

/**
 * Multi-dimensional array (tensor) for numerical computations
 *
 * shape: Vector containing the size of each dimension
 * data: Flattened array storing all elements in row-major order
 */
class Tensor {
private:
	std::vector<size_t> shape;
	std::vector<double> data;

	/**
	 * Compute flat index from multi-dimensional indices
	 *
	 * indices: Vector of indices for each dimension
	 * Output: Flat index in the data array
	 */
	size_t computeIndex(const std::vector<size_t>& indices) const;

public:
	/**
	 * Create a tensor with given shape, initialized to zeros
	 *
	 * shape: Vector containing size of each dimension
	 */
	Tensor(const std::vector<size_t>& shape);

	/**
	 * Create a tensor with given shape and initial values
	 *
	 * shape: Vector containing size of each dimension
	 * values: Initial values in row-major order
	 */
	Tensor(const std::vector<size_t>& shape, const std::vector<double>& values);

	/**
	 * Create a tensor with given shape, filled with a specific value
	 *
	 * shape: Vector containing size of each dimension
	 * fillValue: Value to fill all elements
	 */
	Tensor(const std::vector<size_t>& shape, double fillValue);

	Tensor(const Tensor& other) = default;

	/**
	 * Get the shape of the tensor
	 *
	 * Output: Reference to shape vector
	 */
	const std::vector<size_t>& getShape() const;

	/**
	 * Get the number of dimensions
	 *
	 * Output: Number of dimensions
	 */
	size_t ndim() const;

	/**
	 * Get the total number of elements
	 *
	 * Output: Total number of elements
	 */
	size_t size() const;

	/**
	 * Get element value at given indices (read-only)
	 *
	 * indices: Vector of indices for each dimension
	 * Output: Value at the specified position
	 */
	double get(const std::vector<size_t>& indices) const;

	/**
	 * Get reference to element at given indices (read-write)
	 *
	 * indices: Vector of indices for each dimension
	 * Output: Reference to value at the specified position
	 */
	double& at(const std::vector<size_t>& indices);

	/**
	 * Get read-only access to underlying data array
	 *
	 * Output: Const reference to data vector
	 */
	const std::vector<double>& getData() const;

	/**
	 * Get read-write access to underlying data array
	 *
	 * Output: Reference to data vector
	 */
	std::vector<double>& getData();

	/**
	 * Create a tensor filled with zeros
	 *
	 * shape: Vector containing size of each dimension
	 * Output: New tensor filled with zeros
	 */
	static Tensor zeros(const std::vector<size_t>& shape);

	/**
	 * Create a tensor filled with ones
	 *
	 * shape: Vector containing size of each dimension
	 * Output: New tensor filled with ones
	 */
	static Tensor ones(const std::vector<size_t>& shape);

	/**
	 * Create a tensor filled with random values [0, 1)
	 *
	 * shape: Vector containing size of each dimension
	 * Output: New tensor with random values
	 */
	static Tensor random(const std::vector<size_t>& shape);

	/**
	 * Reshape tensor to new dimensions without changing data
	 *
	 * newShape: New shape (must have same total size)
	 * Output: New tensor with specified shape
	 */
	Tensor reshape(const std::vector<size_t>& newShape) const;

	/**
	 * Flatten tensor to 1D array
	 *
	 * Output: New 1D tensor with all elements
	 */
	Tensor flatten() const;

	/**
	 * Element-wise addition
	 *
	 * other: Tensor to add (must have same shape)
	 * Output: New tensor with element-wise sum
	 */
	Tensor operator+(const Tensor& other) const;

	/**
	 * Element-wise subtraction
	 *
	 * other: Tensor to subtract (must have same shape)
	 * Output: New tensor with element-wise difference
	 */
	Tensor operator-(const Tensor& other) const;

	/**
	 * Scalar multiplication
	 *
	 * scalar: Value to multiply all elements by
	 * Output: New tensor with scaled values
	 */
	Tensor operator*(double scalar) const;

	/**
	 * Element-wise multiplication (Hadamard product)
	 *
	 * other: Tensor to multiply (must have same shape)
	 * Output: New tensor with element-wise product
	 */
	Tensor hadamard(const Tensor& other) const;

	/**
	 * Matrix multiplication (2D tensors only)
	 *
	 * other: Right operand (inner dimensions must match)
	 * Output: Result of matrix multiplication
	 */
	Tensor matmul(const Tensor& other) const;

	/**
	 * Transpose matrix (2D tensors only)
	 *
	 * Output: Transposed tensor
	 */
	Tensor transpose() const;

	/**
	 * Fill all elements with a specific value
	 *
	 * value: Value to fill all elements with
	 */
	void fill(double value);
};

#endif