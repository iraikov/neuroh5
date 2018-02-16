"""
This checks that attributes retrieved by neuroh5.io general read attribute methods match those returned by specific
selective read attribute methods.

forest_path: contains tree attributes
cell_attr_path: contains cell attributes
connections_path: contains edge attributes
"""
from neuroh5 import h5py_io_utils
from mpi4py import MPI
from neuroh5.io import read_population_ranges, NeuroH5TreeGen, NeuroH5CellAttrGen, NeuroH5ProjectionGen
import numpy as np
import click


@click.command()
@click.option("--forest-path", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default=None)
@click.option("--cell-attr-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--connections-path", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default=None)
@click.option("--cell-attr-namespace", type=str, default='Synapse Attributes')
@click.option("--cell-attr", type=str, default='syn_locs')
@click.option("--destination", type=str, default='GC')
@click.option("--source", type=str, default='MPP')
@click.option("--io-size", type=int, default=-1)
@click.option("--cache-size", type=int, default=50)
def main(forest_path, cell_attr_path, connections_path, cell_attr_namespace, cell_attr, destination, source, io_size,
         cache_size):
    """

    :param forest_path: str (path)
    :param cell_attr_path: str (path)
    :param connections_path: str (path)
    :param cell_attr_namespace: str
    :param cell_attr: str
    :param destination: str
    :param source: str
    :param io_size: int
    :param cache_size: int
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        print '%s: %i ranks have been allocated' % (os.path.basename(__file__), comm.size)
    sys.stdout.flush()

    pop_ranges, pop_size = read_population_ranges(cell_attr_path, comm=comm)
    destination_gid_offset = pop_ranges[destination][0]
    source_gid_offset = pop_ranges[source][0]
    maxiter = 10
    cell_attr_matched = 0
    cell_attr_processed = 0
    edge_attr_matched = 0
    edge_attr_processed = 0
    tree_attr_matched = 0
    tree_attr_processed = 0

    cell_attr_gen = NeuroH5CellAttrGen(cell_attr_path, destination, comm=comm, io_size=io_size, cache_size=cache_size,
                                             namespace=cell_attr_namespace)
    index_map = get_cell_attributes_index_map(comm, cell_attr_path, destination, cell_attr_namespace)
    for itercount, (target_gid, attr_dict) in enumerate(cell_attr_gen):
        print 'Rank: %i receieved target_gid: %s from the cell attribute generator.' % (rank, str(target_gid))
        attr_dict2 = select_cell_attributes(target_gid, comm, cell_attr_path, index_map, destination,
                                                cell_attr_namespace, population_offset=destination_gid_offset)
        if np.all(attr_dict[cell_attr][:] == attr_dict2[cell_attr][:]):
            print 'Rank: %i; cell attributes match!' % rank
            cell_attr_matched += 1
        else:
            print 'Rank: %i; cell attributes do not match.' % rank
        comm.barrier()
        cell_attr_processed += 1
        if itercount > maxiter:
            break
    cell_attr_matched = comm.gather(cell_attr_matched, root=0)
    cell_attr_processed = comm.gather(cell_attr_processed, root=0)

    if connections_path is not None:
        edge_attr_gen = NeuroH5ProjectionGen(connections_path, source, destination, comm=comm, io_size=io_size,
                                                               cache_size=cache_size, namespaces=['Synapses'])
        index_map = get_edge_attributes_index_map(comm, connections_path, source, destination)
        processed = 0
        for itercount, (target_gid, attr_package) in enumerate(edge_attr_gen):
            print 'Rank: %i receieved target_gid: %s from the edge attribute generator.' % (rank, str(target_gid))
            source_indexes, attr_dict = attr_package
            syn_ids = attr_dict['Synapses'][0]
            source_indexes2, attr_dict2 = select_edge_attributes(target_gid, comm, connections_path, index_map,
                                                                 source, destination, namespaces=['Synapses'],
                                                                 source_offset=source_gid_offset,
                                                                 destination_offset=destination_gid_offset)
            syn_ids2 = attr_dict2['Synapses'][0]
            if np.all(syn_ids == syn_ids2) and np.all(source_indexes == source_indexes2):
                print 'Rank: %i; edge attributes match!' % rank
                edge_attr_matched += 1
            else:
                print 'Rank: %i; attributes do not match.' % rank
            comm.barrier()
            edge_attr_processed += 1
            if itercount > maxiter:
                break
        edge_attr_matched = comm.gather(edge_attr_matched, root=0)
        edge_attr_processed = comm.gather(edge_attr_processed, root=0)

    if forest_path is not None:
        tree_attr_gen = NeuroH5TreeGen(forest_path, destination, comm=comm, io_size=io_size)
        for itercount, (target_gid, attr_dict) in enumerate(tree_attr_gen):
            print 'Rank: %i receieved target_gid: %s from the tree attribute generator.' % (rank, str(target_gid))
            attr_dict2 = select_tree_attributes(target_gid, comm, forest_path, destination)
            if (attr_dict.keys() == attr_dict2.keys()) and all(attr_dict['layer'] == attr_dict2['layer']):
                print 'Rank: %i; tree attributes match!' % rank
                tree_attr_matched += 1
            else:
                print 'Rank: %i; tree attributes do not match.' % rank
            comm.barrier()
            tree_attr_processed += 1
            if itercount > maxiter:
                break
        tree_attr_matched = comm.gather(tree_attr_matched, root=0)
        tree_attr_processed = comm.gather(tree_attr_processed, root=0)

    if comm.rank == 0:
        print '%i / %i processed gids had matching cell attributes returned by both read methods' % \
              (np.sum(cell_attr_matched), np.sum(cell_attr_processed))
        print '%i / %i processed gids had matching edge attributes returned by both read methods' % \
              (np.sum(edge_attr_matched), np.sum(edge_attr_processed))
        print '%i / %i processed gids had matching tree attributes returned by both read methods' % \
              (np.sum(tree_attr_matched), np.sum(tree_attr_processed))


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1,sys.argv)+1):])
