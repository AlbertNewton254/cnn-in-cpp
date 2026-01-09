# Compiler and flags
CC = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -Iinclude -I../tensor/include -I../layers/include

# Directories
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
TENSOR_SRC_DIR = ../tensor/src
LAYERS_SRC_DIR = ../layers/src

# Source files
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
TENSOR_SOURCES = $(wildcard $(TENSOR_SRC_DIR)/*.cpp)
LAYERS_SOURCES = $(wildcard $(LAYERS_SRC_DIR)/*.cpp)
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
TENSOR_OBJECTS = $(patsubst $(TENSOR_SRC_DIR)/%.cpp,$(BUILD_DIR)/tensor_%.o,$(TENSOR_SOURCES))
LAYERS_OBJECTS = $(patsubst $(LAYERS_SRC_DIR)/%.cpp,$(BUILD_DIR)/layers_%.o,$(LAYERS_SOURCES))

# Targets
.PHONY: all clean

all: $(BUILD_DIR) $(OBJECTS) $(TENSOR_OBJECTS) $(LAYERS_OBJECTS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(wildcard $(INC_DIR)/*.hpp)
	$(CC) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/tensor_%.o: $(TENSOR_SRC_DIR)/%.cpp
	$(CC) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/layers_%.o: $(LAYERS_SRC_DIR)/%.cpp
	$(CC) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)
