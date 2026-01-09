# Compiler and flags
CC = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -Iinclude -I../tensor/include

# Directories
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
TENSOR_SRC_DIR = ../tensor/src

# Source files
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
TENSOR_SOURCES = $(wildcard $(TENSOR_SRC_DIR)/*.cpp)
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
TENSOR_OBJECTS = $(patsubst $(TENSOR_SRC_DIR)/%.cpp,$(BUILD_DIR)/tensor_%.o,$(TENSOR_SOURCES))

# Targets
.PHONY: all clean

all: $(BUILD_DIR) $(OBJECTS) $(TENSOR_OBJECTS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(wildcard $(INC_DIR)/*.hpp)
	$(CC) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/tensor_%.o: $(TENSOR_SRC_DIR)/%.cpp
	$(CC) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)
