import sys, pprint
from mpi4py import MPI
from neuroh5.io import read_population_ranges, read_population_names, scatter_read_cell_attributes
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

script_name = 'test_scatter_read_syn_attrs.py'

@click.command()
@click.option("--syn-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--syn-namespace", type=str, default='Synapse Attributes')
@click.option("--io-size", type=int, default=-1)
def main(syn_path, syn_namespace, io_size):

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    print ('Allocated %i ranks' % size)

    population_ranges = read_population_ranges(syn_path)[0]
    print (population_ranges)
    
    for population in population_ranges.keys():

        attr_dict = scatter_read_cell_attributes(syn_path, population, namespaces=[syn_namespace], io_size=io_size)
        attr_iter = attr_dict[syn_namespace]
        
        for cell_gid, attr_dict in attr_iter:

            print ('Rank %i: gid = %i syn attrs:' % (rank, cell_gid))
            pprint.pprint(attr_dict)
    

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])

