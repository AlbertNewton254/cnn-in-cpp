/* optimizer.cpp */

#include "../include/optimizer.hpp"

void Optimizer::zeroGrad(std::vector<Tensor*>& gradients) {
	for (auto* grad : gradients) {
		grad->fill(0.0);
	}
}