# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -I../tensor/include -I../layers/include -I../model/include -I../loss/include -I../optimizer/include

# Directories
TENSOR_SRC_DIR = ../tensor/src
LAYERS_SRC_DIR = ../layers/src
MODEL_SRC_DIR = ../model/src
LOSS_SRC_DIR = ../loss/src
OPTIMIZER_SRC_DIR = ../optimizer/src
BUILD_DIR = build

# Source files
TENSOR_TEST_SOURCES = $(wildcard tensor/*.cpp)
LAYERS_TEST_SOURCES = $(wildcard layers/*.cpp)
MODEL_TEST_SOURCES = $(wildcard model/*.cpp)
LOSS_TEST_SOURCES = $(wildcard loss/*.cpp)
OPTIMIZER_TEST_SOURCES = $(wildcard optimizer/*.cpp)
TENSOR_SOURCES = $(wildcard $(TENSOR_SRC_DIR)/*.cpp)
LAYERS_SOURCES = $(wildcard $(LAYERS_SRC_DIR)/*.cpp)
MODEL_SOURCES = $(wildcard $(MODEL_SRC_DIR)/*.cpp)
LOSS_SOURCES = $(wildcard $(LOSS_SRC_DIR)/*.cpp)
OPTIMIZER_SOURCES = $(wildcard $(OPTIMIZER_SRC_DIR)/*.cpp)
TENSOR_TEST_BINARIES = $(patsubst tensor/%.cpp,$(BUILD_DIR)/%,$(TENSOR_TEST_SOURCES))
LAYERS_TEST_BINARIES = $(patsubst layers/%.cpp,$(BUILD_DIR)/%,$(LAYERS_TEST_SOURCES))
MODEL_TEST_BINARIES = $(patsubst model/%.cpp,$(BUILD_DIR)/%,$(MODEL_TEST_SOURCES))
LOSS_TEST_BINARIES = $(patsubst loss/%.cpp,$(BUILD_DIR)/%,$(LOSS_TEST_SOURCES))
OPTIMIZER_TEST_BINARIES = $(patsubst optimizer/%.cpp,$(BUILD_DIR)/%,$(OPTIMIZER_TEST_SOURCES))
TEST_BINARIES = $(TENSOR_TEST_BINARIES) $(LAYERS_TEST_BINARIES) $(MODEL_TEST_BINARIES) $(LOSS_TEST_BINARIES) $(OPTIMIZER_TEST_BINARIES)

# Targets
.PHONY: all clean run

all: $(BUILD_DIR) $(TEST_BINARIES)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/test_tensor: tensor/test_tensor.cpp $(TENSOR_SOURCES)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/test_layers: layers/test_layers.cpp $(TENSOR_SOURCES) $(LAYERS_SOURCES)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/test_sequential: model/test_sequential.cpp $(TENSOR_SOURCES) $(LAYERS_SOURCES) $(MODEL_SOURCES)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/test_loss: loss/test_loss.cpp $(TENSOR_SOURCES) $(LOSS_SOURCES)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/test_optimizer: optimizer/test_optimizer.cpp $(TENSOR_SOURCES) $(OPTIMIZER_SOURCES)
	$(CXX) $(CXXFLAGS) $^ -o $@

run: all
	@for test in $(TEST_BINARIES); do \
		echo "Running $$test..."; \
		$$test || exit 1; \
		echo ""; \
	done

clean:
	rm -rf $(BUILD_DIR)
