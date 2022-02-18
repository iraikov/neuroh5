import sys, os
from mpi4py import MPI
from neuroh5.io import read_population_ranges, read_population_names, bcast_cell_attributes
import numpy as np
import click

# import mkl

def list_find (f, lst):
    i=0
    for x in lst:
        if f(x):
            return i
        else:
            i=i+1
    return None

script_name = 'test_bcast_coords.py'

@click.command()
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-namespace", type=str, default='Coordinates')
def main(coords_path, coords_namespace):

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    print ('Allocated %i ranks' % size)
    sys.stdout.flush()

    population_ranges = read_population_ranges(coords_path)[0]
    
    soma_coords = {}
    for population in ['GC']:


        attr_iter = bcast_cell_attributes(coords_path, population, namespace=coords_namespace, 
                                          root=0, mask=set(['U Coordinate', 'V Coordinate', 'L Coordinate']),
                                          comm=comm)
        
        for cell_gid, coords_dict in attr_iter:

            cell_u = coords_dict['U Coordinate']
            cell_v = coords_dict['V Coordinate']
                
            print ('Rank %i: gid = %i u = %f v = %f' % (rank, cell_gid, cell_u, cell_v))
    

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])
