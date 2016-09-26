MPI_INCDIR      =
MPI_LIBDIR      = 
HDF5_INCDIR	= $(HOME)/.local/include
HDF5_LIBDIR	= $(HOME)/.local/lib
HDF5_LIB        = hdf5

CFLAGS		= -Wall -I$(HDF5_INCDIR)
LDFLAGS		= -L$(HDF5_LIBDIR)

all: reader scatter

reader: reader.o dbs_graph_reader.o population_reader.o
	$(CXX) -std=c++11 -o reader $(LDFLAGS) reader.o dbs_graph_reader.o population_reader.o -l$(HDF5_LIB)

scatter: scatter.o dbs_graph_reader.o population_reader.o
	$(CXX) -std=c++11 -o scatter $(LDFLAGS) scatter.o dbs_graph_reader.o population_reader.o -l$(HDF5_LIB)

.cc.o:
	$(CXX) -std=c++11 $(CFLAGS) -c $< -o $@

clean:
	-rm *.o reader scatter *.edges
