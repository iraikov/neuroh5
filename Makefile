AR		:= ar
CC		:= g++
LD        	:= g++

HDF5_DIR	:= $(HOME)/.local
HDF5_INCDIR 	:= $(HDF5_DIR)/include
HDF5_LIBDIR 	:= $(HDF5_DIR)/lib

MPI_DIR		:= /mnt/hdf/packages/mpich/new/x86_64/EL7
MPI_INCDIR	:= $(MPI_DIR)/include
MPI_LIBDIR	:= $(MPI_DIR)/lib

PARMETIS_DIR	:= $(HOME)/work/packages/parmetis-4.0.3
PARMETIS_INCDIR	:= $(PARMETIS_DIR)/include
PARMETIS_LIBDIR	:= $(PARMETIS_DIR)/lib

MODULES   	:= driver graph io io/hdf5 model
INC_DIR   	:= $(addprefix include/,$(MODULES))
SRC_DIR   	:= $(addprefix src/,$(MODULES))
BUILD_DIR	:= $(addprefix build/,$(MODULES))

SRC      	:= $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cc))
OBJ       	:= $(patsubst src/%.cc,build/%.o,$(SRC))
INCLUDES  	:= $(addprefix -I,$(INC_DIR) include $(HDF5_INCDIR) \
			$(PARMETIS_INCDIR) $(MPI_INCDIR))

vpath %.cc $(SRC_DIR)

define make-goal
$1/%.o: %.cc
	$(CC) -std=c++11 -Wall $(INCLUDES) -O2 -c $$< -o $$@
endef

.PHONY: all checkdirs clean

all: checkdirs build/reader build/scatter build/parts

build/parts: build/libngh5.graph.a build/libngh5.io.a build/libngh5.io.hdf5.a
	$(LD) -o $@ $^ -L$(HDF5_LIBDIR) -L$(MPI_LIBDIR) -lhdf5 -lmpi

build/reader: build/libngh5.graph.a build/libngh5.io.a build/libngh5.io.hdf5.a
	$(LD) -o $@ $^ -L$(HDF5_LIBDIR) -L$(MPI_LIBDIR) -lhdf5 -lmpi

build/scatter: build/libngh5.graph.a build/libngh5.io.a build/libngh5.io.hdf5.a
	$(LD) -o $@ $^ -L$(HDF5_LIBDIR) -L$(MPI_LIBDIR) -lhdf5 -lmpi

build/libngh5.io.a: $(OBJ)
	$(AR) cr $@ $^

build/libngh5.io.hdf5.a: $(OBJ)
	$(AR) cr $@ $^

build/libngh5.graph.a: $(OBJ)
	$(AR) cr $@ $^

checkdirs: $(BUILD_DIR)

$(BUILD_DIR):
	@mkdir -p $@

clean:
	@rm -rf $(BUILD_DIR)

$(foreach bdir,$(BUILD_DIR),$(eval $(call make-goal,$(bdir))))
