import sys
from mpi4py import MPI
from neuroh5.io import read_population_ranges, read_population_names, read_cell_attributes
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

script_name = 'test_read_coords.py'

@click.command()
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-namespace", type=str, default='Coordinates')
def main(coords_path, coords_namespace):

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    print ('Allocated %i ranks' % size)

    population_ranges = read_population_ranges(coords_path)[0]
    print (population_ranges)
    
    soma_coords = {}
    populations = ['GC']
    for population in population_ranges.keys():

        print('Population %s' % population)
        it = read_cell_attributes(coords_path, population, namespace=coords_namespace)

        print('it = %s' % str(it))
        for cell_gid, coords_dict in it:

            print(hasattr(coords_dict, 'U Coordinate'))
            cell_u = getattr(coords_dict, 'U Coordinate')
            cell_u = getattr(coords_dict, 'U Coordinate')
            cell_v = getattr(coords_dict, 'V Coordinate')
                
            print ('Rank %i: gid = %i u = %f v = %f' % (rank, cell_gid, cell_u, cell_v))


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])

