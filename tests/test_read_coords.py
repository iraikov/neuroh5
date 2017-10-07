import sys
from mpi4py import MPI
from neuroh5.io import read_population_ranges, read_population_names, NeuroH5CellAttrGen
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
@click.option("--coords-namespace", type=str, default='Sorted Coordinates')
@click.option("--io-size", type=int, default=-1)
def main(coords_path, coords_namespace, io_size):

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    print 'Allocated %i ranks' % size

    population_ranges = read_population_ranges(comm, coords_path)[0]

    print population_ranges
    
    soma_coords = {}
    for population in population_ranges.keys():
        (population_start, _) = population_ranges[population]

        for cell_gid, attr_dict in NeuroH5CellAttrGen(comm, coords_path, population, io_size=io_size,
                                                       namespace=coords_namespace):

            if cell_gid is None:
                print 'Rank %i cell gid is None' % rank
            else:
                coords_dict = attr_dict[coords_namespace]

                cell_u = coords_dict['U Coordinate']
                cell_v = coords_dict['V Coordinate']
                
                print 'Rank %i: gid = %i u = %f v = %f' % (rank, cell_gid, cell_u, cell_v)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])

