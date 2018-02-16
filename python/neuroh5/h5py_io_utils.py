import sys, os
from mpi4py import MPI  # Must come before importing NEURON and/or h5py
import h5py
import numpy as np
from neuroh5.io import read_tree_selection


def list_find(criterion, items):
    for i, item in enumerate(items):
        if criterion(item):
            return i
    return None


def get_cell_attributes_index_map(comm, file_path, population, namespace):
    """

    :param comm: MPI communicator
    :param file_path: str (path to neuroh5 file)
    :param population: str
    :param namespace: str
    :return: dict
    """
    index_name = 'Cell Index'
    index_map = {}
    with h5py.File(file_path, 'r', driver='mpio', comm=comm) as f:
        if 'Populations' not in f or population not in f['Populations'] or \
                namespace not in f['Populations'][population]:
            raise KeyError('Population: %s; namespace: %s; not found in file_path: %s' %
                           (population, namespace, file_path))
        for attribute, group in f['Populations'][population][namespace].iteritems():
            dataset = group[index_name]
            with dataset.collective:
                index_map[attribute] = dict(zip(dataset[:], xrange(dataset.shape[0])))
    return index_map


def select_cell_attributes(gid, comm, file_path, index_map, population, namespace, population_offset=0):
    """

    :param gid: int
    :param comm: MPI communicator
    :param file_path: str (path to neuroh5 file)
    :param index_map: dict of int: int; maps gid to pointer index
    :param population: str
    :param namespace: str
    :param population_offset: int
    :return: dict
    """
    pointer_name = 'Attribute Pointer'
    value_name = 'Attribute Value'
    in_dataset = True
    attr_dict = {}
    with h5py.File(file_path, 'r', driver='mpio', comm=comm) as f:
        if 'Populations' not in f or population not in f['Populations'] or \
                namespace not in f['Populations'][population]:
            raise KeyError('Population: %s; namespace: %s; not found in file_path: %s' %
                           (population, namespace, file_path))
        group = f['Populations'][population][namespace]
        for attribute in group:
            if attribute not in index_map:
                raise KeyError('Invalid index_map; population: %s; namespace: %s; attribute: %s' %
                               (population, namespace, attribute))
            if not in_dataset or gid is None:
                index = 0
                in_dataset = False
            else:
                in_dataset = True
                try:
                    index = index_map[attribute][gid - population_offset]
                except KeyError:
                    index = 0
                    in_dataset = False
            pointer_dataset = group[attribute][pointer_name]
            with pointer_dataset.collective:
                start = pointer_dataset[index]
                end = pointer_dataset[index + 1]
            value_dataset = group[attribute][value_name]
            with value_dataset.collective:
                attr_dict[attribute] = value_dataset[start:end]
    if in_dataset:
        return attr_dict
    else:
        return None


def get_edge_attributes_index_map(comm, file_path, source, destination):
    """

    :param comm: MPI communicator
    :param file_path: str (path to neuroh5 file)
    :param source: str
    :param destination: str
    :return: dict
    """
    index_namespace = 'Edges'
    index_name = 'Destination Block Index'
    index_map = {}
    with h5py.File(file_path, 'r', driver='mpio', comm=comm) as f:
        if 'Projections' not in f or destination not in f['Projections'] or \
                source not in f['Projections'][destination]:
            raise KeyError('Projection from source: %s --> destination: %s not found in file_path: %s' %
                           (source, destination, file_path))
        if index_namespace not in f['Projections'][destination][source]:
            raise KeyError('Projection from source: %s --> destination: %s; namespace: %s not found in file_path: '
                           '%s' % (source, destination, index_namespace, file_path))
        dataset = f['Projections'][destination][source][index_namespace]['Destination Block Pointer']
        with dataset.collective:
            if np.any(np.diff(dataset[:])[:-1] > 1):
                raise NotImplementedError('get_edge_attributes_gid_index_map: block size > 1 not yet implemented')
        dataset = f['Projections'][destination][source][index_namespace][index_name]
        with dataset.collective:
            index_map = dict(zip(dataset[:], xrange(dataset.shape[0])))
    return index_map


def select_edge_attributes(gid, comm, file_path, index_map, source, destination, namespaces, source_offset=0,
                           destination_offset=0):
    """

    :param gid: int
    :param comm: MPI communicator
    :param file_path: str (path to neuroh5 file)
    :param index_map: dict of int: int; maps gid to pointer index
    :param source: str
    :param destination: str
    :param namespaces: list of str
    :param source_offset: int
    :param destination_offset: int
    :return: tuple: (array, dict)
    """
    pointer_name = 'Destination Pointer'
    valid_namespaces = {'Synapses': 'syn_id', 'Connections': 'distance'}
    for namespace in namespaces:
        if namespace not in valid_namespaces:
            raise KeyError('Projection from source: %s --> destination: %s; invalid namespace: %s; file_path: %s' %
                           (source, destination, namespace, file_path))
    in_dataset = True
    attr_dict = {}
    with h5py.File(file_path, 'r', driver='mpio', comm=comm) as f:
        if 'Projections' not in f or destination not in f['Projections'] or \
                source not in f['Projections'][destination]:
            raise KeyError('Projection from source: %s --> destination: %s not found in file_path: %s' %
                           (source, destination, file_path))
        if not in_dataset or gid is None:
            index = 0
            in_dataset = False
        else:
            in_dataset = True
            try:
                index = index_map[gid - destination_offset]
            except KeyError:
                index = 0
                in_dataset = False
        pointer_dataset = f['Projections'][destination][source]['Edges'][pointer_name]
        with pointer_dataset.collective:
            start = pointer_dataset[index]
            end = pointer_dataset[index + 1]
        source_index_dataset = f['Projections'][destination][source]['Edges']['Source Index']
        with source_index_dataset.collective:
            source_indexes = source_index_dataset[start:end] + source_offset
        for namespace in namespaces:
            attribute = valid_namespaces[namespace]
            value_dataset = f['Projections'][destination][source][namespace][attribute]
            attr_dict[namespace] = []
            with value_dataset.collective:
                attr_dict[namespace].append(value_dataset[start:end])
    if in_dataset:
        return source_indexes, attr_dict
    else:
        return None, None


def select_tree_attributes(gid, comm, file_path, population):
    """

    :param gid: int
    :param comm: MPI communicator
    :param file_path: str (path to neuroh5 file)
    :param population: str
    :return: dict
    """
    try:
        tree_attr_iter, num_cells = read_tree_selection(file_path, population, [gid], comm=comm)
    except Exception:
        raise Exception('Something went wrong with read_tree_selection for population: %s; gid: %s, file_path: %s' %
                        (population, gid, file_path))
    gid, attr_dict = tree_attr_iter.next()
    return attr_dict


def create_new_neuroh5_file(template_path, output_path):
    """
    A blank neuroH5 file capable of being the target of new append operations only requires the existence of a
    top level 'H5Types' group. This method creates a new file by copying 'H5Types' from a template file.
    :param template_path: str (path)
    """
    if not os.path.isfile(template_path):
        raise IOError('Invalid path to neuroH5 template file: %s' % template_path)
    with h5py.File(template_path, 'r') as source:
        if 'H5Types' not in source:
            raise KeyError('Invalid neuroH5 template file: %s' % template_path)
        with h5py.File(output_path, 'w') as target:
            target.copy(source['H5Types'], target, name='H5Types')
            print 'Created new neuroH5 file: %s' % output_path


def gid_in_population_list(gid, population_list, population_range_dict):
    """
    If the gid is in the gid range of a population, the name of the population is returned. Otherwise, None.
    :param gid: int
    :param population_list: list of str
    :param population_range_dict: dict of tuple of int
    :return: str or None
    """
    for population in population_list:
        if population_range_dict[population][0] <= gid < population_range_dict[population][0] + \
                population_range_dict[population][1]:
            return population
    return None


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.rank
    file_path = 'datasets/Full_Scale_Control/neuroh5_example_file.h5'
    population = 'GC'
    namespace = 'Synapse Attributes'
    index_map = get_cell_attributes_index_map(comm, file_path, population, namespace)
    gid = index_map.itervalues().next().keys()[rank]
    attr_dict = select_cell_attributes(gid, comm, file_path, index_map, population, namespace, population_offset=0)
    print 'Rank: %i, gid: %i, num_syns: %i' % (rank, gid, len(attr_dict['syn_ids']))
    sys.stdout.flush()
