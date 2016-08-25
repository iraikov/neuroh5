HDF5_INCDIR	= $(HOME)/.local/include
HDF5_LIBDIR	= $(HOME)/.local/lib

MPICXX		= mpicxx -std=c++11
CFLAGS		= -Wall -I$(HDF5_INCDIR)
LDFLAGS		= -L$(HDF5_LIBDIR)

all: reader

reader: main.o dbs_graph_reader.o population_reader.o
	$(MPICXX) -o reader $(LDFLAGS) main.o dbs_graph_reader.o population_reader.o -lhdf5

.cc.o:
	$(MPICXX) $(CFLAGS) -c $< -o $@

clean:
	-rm *.o reader
