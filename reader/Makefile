MPI_INCDIR      =
MPI_LIBDIR      = 
HDF5_INCDIR	= $(HOME)/.local/include
HDF5_LIBDIR	= $(HOME)/.local/lib

CFLAGS		= -Wall -I$(HDF5_INCDIR) -I$(MPI_INCDIR)
LDFLAGS		= -L$(HDF5_LIBDIR) -L$(MPI_LIBDIR)

all: reader

reader: main.o
	$(CXX) -o reader $(LDFLAGS) main.o -lhdf5 -lmpich

.cc.o:
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	-rm *.o reader
