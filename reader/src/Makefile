MPI_INCDIR      =
MPI_LIBDIR      = 
HDF5_INCDIR	= $(HOME)/.local/include
HDF5_LIBDIR	= $(HOME)/.local/lib
HDF5_LIB        = hdf5

CFLAGS		= -Wall -I$(HDF5_INCDIR) -I$(MPI_INCDIR)
LDFLAGS		= -L$(HDF5_LIBDIR) -L$(MPI_LIBDIR)

all: reader scatter

reader: reader.o graph_reader.o dbs_edge_reader.o population_reader.o attributes.o
	$(CXX) -std=c++11 -o reader $(LDFLAGS) reader.o graph_reader.o dbs_edge_reader.o population_reader.o attributes.o -l$(HDF5_LIB) -lmpi

scatter: scatter.o graph_reader.o dbs_edge_reader.o population_reader.o
	$(CXX) -std=c++11 -o scatter $(LDFLAGS) scatter.o graph_reader.o dbs_edge_reader.o population_reader.o attributes.o -l$(HDF5_LIB) -lmpi

.cc.o:
	$(CXX) -std=c++11 $(CFLAGS) -c $< -o $@

clean:
	-rm *.o reader scatter *.edges
