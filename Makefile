CC        := g++
AR				:= ar
LD        := g++

HDF5_DIR 		:= $(HOME)/.local
HDF5_INCDIR := $(HDF5_DIR)/include
HDF5_LIBDIR := $(HDF5_DIR)/lib

MPI_DIR			:= /mnt/hdf/packages/mpich/new/x86_64/EL7
MPI_INCDIR	:= $(MPI_DIR)/include
MPI_LIBDIR	:= $(MPI_DIR)/lib

MODULES   := io/hdf5
INC_DIR   := $(addprefix include/,$(MODULES))
SRC_DIR   := $(addprefix src/,$(MODULES))
BUILD_DIR := $(addprefix build/,$(MODULES))

SRC       := $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cc))
OBJ       := $(patsubst src/%.cc,build/%.o,$(SRC))
INCLUDES  := $(addprefix -I,$(INC_DIR) ./include $(HDF5_INCDIR) $(MPI_INCDIR))

vpath %.cc $(SRC_DIR)

define make-goal
$1/%.o: %.cc
	$(CC) -std=c++11 -Wall $(INCLUDES) -c $$< -o $$@
endef

.PHONY: all checkdirs clean

all: checkdirs build/libngh5.io.hdf5.a

build/libngh5.io.hdf5.a: $(OBJ)
	$(AR) cr $@ $^


checkdirs: $(BUILD_DIR)

$(BUILD_DIR):
	@mkdir -p $@

clean:
	@rm -rf $(BUILD_DIR)

$(foreach bdir,$(BUILD_DIR),$(eval $(call make-goal,$(bdir))))
