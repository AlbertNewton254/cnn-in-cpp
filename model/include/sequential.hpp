/* sequential.hpp */

#ifndef SEQUENTIAL_HPP
#define SEQUENTIAL_HPP

#include "model.hpp"
#include "../../layers/include/layer.hpp"
#include <vector>
#include <memory>

/**
 * Sequential model that stacks layers in order
 *
 * layers: Ordered list of layers to apply
 *
 * Inspired by PyTorch's nn.Sequential
 */
class Sequential : public Model {
private:
	std::vector<std::shared_ptr<Layer>> layers;

public:
	Sequential();
	~Sequential() override = default;

	/**
	 * Add a layer to the end of the sequence
	 *
	 * layer: Shared pointer to layer to add
	 */
	void addLayer(std::shared_ptr<Layer> layer);

	/**
	 * Forward pass through all layers in sequence
	 *
	 * input: Input tensor
	 * Output: Output after passing through all layers
	 */
	Tensor forward(const Tensor& input) override;

	/**
	 * Backward pass through all layers in reverse order
	 *
	 * gradOutput: Gradient of loss with respect to output
	 * Output: Gradient of loss with respect to input
	 */
	Tensor backward(const Tensor& gradOutput);

	/**
	 * Get all trainable parameters from all layers
	 *
	 * Output: Vector of pointers to all weight tensors
	 */
	std::vector<Tensor*> getParameters() override;

	/**
	 * Get all parameter gradients from all layers
	 *
	 * Output: Vector of pointers to all gradient tensors
	 */
	std::vector<Tensor*> getGradients() override;

	/**
	 * Get number of layers in the model
	 *
	 * Output: Number of layers
	 */
	size_t numLayers() const;

	/**
	 * Get a specific layer by index
	 *
	 * index: Layer index (0-based)
	 * Output: Shared pointer to the layer
	 */
	std::shared_ptr<Layer> getLayer(size_t index);
};

#endif