HDF5_INCDIR	= $(HOME)/.local/include
HDF5_LIBDIR	= $(HOME)/.local/lib

MPICXX		= mpicxx
CFLAGS		= -Wall -I$(HDF5_INCDIR)
LDFLAGS		= -L$(HDF5_LIBDIR)

all: reader

reader: main.o crs_graph_reader.o
	$(MPICXX) -o reader $(LDFLAGS) main.o crs_graph_reader.o -lhdf5

.cc.o:
	$(MPICXX) $(CFLAGS) -c $< -o $@

clean:
	-rm *.o reader
