/* sequential.cpp */

#include "../include/sequential.hpp"

Sequential::Sequential() : Model() {}

void Sequential::addLayer(std::shared_ptr<Layer> layer) {
	layers.push_back(layer);
}

Tensor Sequential::forward(const Tensor& input) {
	if (layers.empty()) {
		return input;
	}

	Tensor output = layers[0]->forward(input);
	for (size_t i = 1; i < layers.size(); i++) {
		output = layers[i]->forward(output);
	}

	return output;
}

Tensor Sequential::backward(const Tensor& gradOutput) {
	if (layers.empty()) {
		return gradOutput;
	}

	Tensor gradInput = gradOutput;
	for (int i = layers.size() - 1; i >= 0; i--) {
		gradInput = layers[i]->backward(gradInput);
	}

	return gradInput;
}

std::vector<Tensor*> Sequential::getParameters() {
	std::vector<Tensor*> params;

	for (auto& layer : layers) {
		if (layer->hasWeights()) {
			std::vector<Tensor*> layerParams = layer->getWeights();
			params.insert(params.end(), layerParams.begin(), layerParams.end());
		}
	}

	return params;
}

std::vector<Tensor*> Sequential::getGradients() {
	std::vector<Tensor*> grads;

	for (auto& layer : layers) {
		if (layer->hasWeights()) {
			std::vector<Tensor*> layerGrads = layer->getGradients();
			grads.insert(grads.end(), layerGrads.begin(), layerGrads.end());
		}
	}

	return grads;
}

size_t Sequential::numLayers() const {
	return layers.size();
}

std::shared_ptr<Layer> Sequential::getLayer(size_t index) {
	if (index >= layers.size()) {
		throw LayerIndexOutOfRangeError();
	}
	return layers[index];
}