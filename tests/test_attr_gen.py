import sys
from mpi4py import MPI
from neuroh5.io import read_population_ranges, read_population_names, scatter_read_cell_attributes, NeuroH5CellAttrGen
import numpy as np
import click

def list_find (f, lst):
    i=0
    for x in lst:
        if f(x):
            return i
        else:
            i=i+1
    return None

script_name = 'test_attr_gen.py'

@click.command()
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-namespace", type=str, default='Coordinates')
@click.option("--io-size", type=int, default=-1)
@click.option("--cache-size", type=int, default=10)
def main(coords_path, coords_namespace, io_size, cache_size):

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    print ('Allocated %i ranks' % size)

    population_ranges = read_population_ranges(coords_path)[0]
    print (population_ranges)
    
    soma_coords = {}
    for population in population_ranges.keys():

        attr_iter = NeuroH5CellAttrGen(coords_path, population, namespace=coords_namespace, \
                                        comm=comm, io_size=io_size, cache_size=cache_size)

        i = 0
        for cell_gid, coords_dict in attr_iter:

            if cell_gid is not None:
                print('coords_dict: ', coords_dict)
                cell_u = coords_dict['U Coordinate']
                cell_v = coords_dict['V Coordinate']
                
                print ('Rank %i: gid = %i u = %f v = %f' % (rank, cell_gid, cell_u, cell_v))
                if i > 10:
                    break
                i = i+1
                
    if rank == 0:
        import h5py
        count = 0
        f = h5py.File(coords_path, 'r+')
        if 'test' in f:
            count = f['test'][()]
            del(f['test'])
        f['test'] = count+1
    comm.barrier()
    

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])

