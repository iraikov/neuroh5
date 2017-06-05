DRIVER_SRC_DIR   	:= src/driver
DRIVER_BUILD_DIR	:= build/driver
DRIVER_SRC      	:= $(foreach sdir,$(DRIVER_SRC_DIR),$(wildcard $(sdir)/*.cc))
DRIVER_OBJ       	:= $(patsubst src/%.cc,build/%.o,$(DRIVER_SRC))
ifndef PARMETIS
DRIVER_OBJ       	:= $(filter-out build/driver/neurograph_parts.o,$(DRIVER_OBJ))
endif
DRIVER_INCLUDES  	:= $(addprefix -I,$(INC_DIR) include $(HDF5_INCDIR) $(MPI_INCDIR) $(PARMETIS_INCDIR))

MODULES   	:= cell graph io mpi data hdf5 ngraph
INC_DIR   	:= $(addprefix include/,$(MODULES))
SRC_DIR   	:= $(addprefix src/,$(MODULES))
BUILD_DIR	:= $(addprefix build/,$(MODULES))

SRC      	:= $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cc))
OBJ       	:= $(patsubst src/%.cc,build/%.o,$(SRC))
INCLUDES  	:= $(addprefix -I,$(INC_DIR) include $(HDF5_INCDIR) $(MPI_INCDIR) $(PARMETIS_INCDIR))
ifndef PARMETIS
OBJ       	:= $(filter-out build/graph/partition_graph.o,$(OBJ))
endif

vpath %.cc $(SRC_DIR):$(DRIVER_SRC_DIR)

define make-goal
$1/%.o: %.cc
	$(CC) -std=c++11 -Wall -Wno-unused-but-set-variable -DUSE_EDGE_DELIM $(INCLUDES) -g -c $$< -o $$@
endef

.PHONY: all checkdirs clean

all: checkdirs $(DRIVER_OBJ) build/reader build/scatter build/balance_indegree build/vertex_metrics build/neurograph_import

ifdef PARMETIS
build/neurograph_parts: build/driver/neurograph_parts.o build/libngh5.graph.a build/libngh5.io.a build/libngh5.io.hdf5.a build/libngh5.mpi.a
	$(LD) -o $@ $^ -L$(HDF5_LIBDIR) -L$(PARMETIS_LIBDIR) -l$(HDF5_LIB)  -lparmetis -lmetis $(LINK_MPI)
endif

build/reader: build/driver/neurograph_reader.o build/libngh5.graph.a build/libngh5.io.a build/libngh5.io.hdf5.a
	$(LD) -o $@ $^ -L$(HDF5_LIBDIR) -l$(HDF5_LIB) $(LINK_MPI) $(LINK_STDCPLUS)

build/scatter: build/driver/neurograph_scatter.o build/libngh5.graph.a build/libngh5.io.a build/libngh5.io.hdf5.a
	$(LD) -o $@ $^ -L$(HDF5_LIBDIR) -l$(HDF5_LIB) $(LINK_MPI) $(LINK_STDCPLUS)

build/balance_indegree: build/driver/balance_indegree.o build/libngh5.graph.a build/libngh5.io.a build/libngh5.io.hdf5.a
	$(LD) -o $@ $^ -L$(HDF5_LIBDIR) -l$(HDF5_LIB) $(LINK_MPI) $(LINK_STDCPLUS)

build/vertex_metrics: build/driver/vertex_metrics.o build/libngh5.graph.a build/libngh5.io.a build/libngh5.io.hdf5.a
	$(LD) -o $@ $^ -L$(HDF5_LIBDIR) -l$(HDF5_LIB) $(LINK_MPI) $(LINK_STDCPLUS)

build/neurograph_import: build/driver/neurograph_import.o build/libngh5.graph.a build/libngh5.io.a build/libngh5.io.hdf5.a build/libngh5.mpi.a 
	$(LD) -o $@ $^ -L$(HDF5_LIBDIR) -l$(HDF5_LIB) $(LINK_MPI) $(LINK_STDCPLUS)

build/libngh5.io.a: $(OBJ)
	$(AR) cr $@ $^

build/libngh5.io.hdf5.a: $(OBJ)
	$(AR) cr $@ $^

build/libngh5.graph.a: $(OBJ)
	$(AR) cr $@ $^

build/libngh5.mpi.a: $(OBJ)
	$(AR) cr $@ $^

checkdirs: $(BUILD_DIR) $(DRIVER_BUILD_DIR)

$(BUILD_DIR):
	@mkdir -p $@

$(DRIVER_BUILD_DIR):
	@mkdir -p $@

clean:
	@rm -rf $(BUILD_DIR) $(DRIVER_BUILD_DIR)

$(foreach bdir,$(BUILD_DIR),$(eval $(call make-goal,$(bdir))))
$(foreach bdir,$(DRIVER_BUILD_DIR),$(eval $(call make-goal,$(bdir))))
