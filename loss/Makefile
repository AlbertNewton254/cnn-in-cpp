CC = g++
CFLAGS = -std=c++17 -Wall -Wextra -O2
INCLUDES = -Iinclude -I../tensor/include

SRC_DIR = src
OBJ_DIR = build
TENSOR_DIR = ../tensor

SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))
TENSOR_OBJS = $(TENSOR_DIR)/build/tensor.o

all: $(OBJS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(TENSOR_OBJS):
	$(MAKE) -C $(TENSOR_DIR)

clean:
	rm -rf $(OBJ_DIR)

.PHONY: all clean
