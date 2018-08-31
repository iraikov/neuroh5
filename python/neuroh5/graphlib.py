""" Module for parallel graph analysis operations. """

import sys, os
import os.path
import click
import itertools, functools
from collections import defaultdict
from array import array
import numpy as np
from mpi4py import MPI 
from neuroh5.io import read_population_ranges, scatter_read_graph, read_graph

#from networkit import *
#from _NetworKit import GraphEvent, GraphUpdater

def ifilternone(iterable):
    for x in iterable:
        if not (x is None):
            yield x
            
def flatten(iterables):
    return (elem for iterable in ifilternone(iterables) for elem in iterable)

def query_alltoall (comm, f, query_rank_node_dict):
    sendbuf=[]

    for rank in range(0,comm.Get_size()):
        if rank in query_rank_node_dict:
            sendbuf.append(query_rank_node_dict[rank])
        else:
            sendbuf.append(None)

    query_input = flatten(comm.alltoall(sendobj=sendbuf))
    
    query_local_result = functools.reduce(f, query_input, {})

    sendbuf = []
    for rank in range(0,comm.Get_size()):
        if rank in query_local_result:
            sendbuf.append(query_local_result[rank])
        else:
            sendbuf.append(None)

    query_result = comm.alltoall(sendobj=sendbuf)

    return query_result


def make_node_rank_map (comm, filepath, iosize):

    size = comm.Get_size()
    
    (_, n_nodes) = read_population_ranges(filepath, comm=comm)

    node_rank_map = {}
    for i in range(0, n_nodes):
        node_rank_map[i] = i % size
    
    return (node_rank_map, n_nodes)


def read_neighbors (comm, filepath, iosize, node_ranks):

    def neighbors_default():
        return {'src': array('L'), 'dst': array('L')}
    
    neighbors_dict = defaultdict(neighbors_default)

    (graph, _) = scatter_read_graph (filepath, io_size=iosize, map_type=0, 
                                     node_rank_map=node_ranks, comm=comm)
    
    ## determine neighbors of vertex based on incoming edges
    for post, prj in graph.items():
        for pre, edge_iter in prj.items():
            for (n, edges) in edge_iter:
                neighbors_dict[n]['src'].extend(edges[0])
                
    ## obtain outgoing edges
    (graph, _) = scatter_read_graph (filepath, io_size=iosize, map_type=1, 
                                     node_rank_map=node_ranks, comm=comm)

    ## determine neighbors of vertex based on outgoing edges
    for pre, prj in graph.items():
        for post, edge_iter in prj.items():
            for (n, edges) in edge_iter:
                neighbors_dict[n]['dst'].extend(edges[0])
            
    return neighbors_dict


def neighbor_degrees (comm, neighbors_dict, node_ranks, verbose=False):

    rank = comm.Get_rank()

    degree_dict = {}

    min_total_degree=sys.maxsize
    max_total_degree=0
    min_in_degree=sys.maxsize
    max_in_degree=0
    min_out_degree=sys.maxsize
    max_out_degree=0
    
    max_in_degree_node_id=0
    min_in_degree_node_id=0
    max_out_degree_node_id=0
    min_out_degree_node_id=0
    max_total_degree_node_id=0
    min_total_degree_node_id=0

    for (v,ns) in neighbors_dict.items():
        in_degree  = len(ns['src'])
        out_degree = len(ns['dst'])
        total_degree = in_degree+out_degree
        if max_total_degree < total_degree:
            max_total_degree_node_id = v
        if min_total_degree > total_degree:
            min_total_degree_node_id = v
        if max_in_degree < in_degree:
            max_in_degree_node_id = v
        if min_in_degree > in_degree:
            min_in_degree_node_id = v
        if max_out_degree < out_degree:
            max_out_degree_node_id = v
        if min_out_degree > in_degree:
            min_out_degree_node_id = v
        min_total_degree = min(min_total_degree, total_degree)
        max_total_degree = max(max_total_degree, total_degree)
        min_in_degree  = min(min_in_degree, in_degree)
        max_in_degree  = max(max_in_degree, in_degree)
        min_out_degree = min(min_out_degree, out_degree)
        max_out_degree = max(max_out_degree, out_degree)
        degree_dict[v] = {'total': in_degree+out_degree, 'in': in_degree, 'out': out_degree}

    (global_min_total_degree, global_min_total_degree_node_id) = comm.allreduce(sendobj=(min_total_degree,min_total_degree_node_id), op=MPI.MINLOC)
    (global_max_total_degree, global_max_total_degree_node_id) = comm.allreduce(sendobj=(max_total_degree,max_total_degree_node_id), op=MPI.MAXLOC)
    (global_min_in_degree, global_min_in_degree_node_id)       = comm.allreduce(sendobj=(min_in_degree,min_in_degree_node_id), op=MPI.MINLOC)
    (global_max_in_degree, global_max_in_degree_node_id)       = comm.allreduce(sendobj=(max_in_degree,max_in_degree_node_id), op=MPI.MAXLOC)
    (global_min_out_degree, global_min_out_degree_node_id)     = comm.allreduce(sendobj=(min_out_degree,min_out_degree_node_id), op=MPI.MINLOC)
    (global_max_out_degree, global_max_out_degree_node_id)     = comm.allreduce(sendobj=(max_out_degree,max_out_degree_node_id), op=MPI.MAXLOC)
    if rank == 0 and verbose:
        print(('neighbor_degrees: max degrees: total=%d (%d) in=%d (%d) out=%d (%d)' % (global_max_total_degree, global_max_total_degree_node_id,
                                                                                       global_max_in_degree, global_max_in_degree_node_id,
                                                                                       global_max_out_degree, global_max_out_degree_node_id)))
        print(('neighbor_degrees: min degrees: total=%d (%d) in=%d (%d) out=%d (%d)' % (global_min_total_degree, global_min_total_degree_node_id,
                                                                                       global_min_in_degree, global_min_in_degree_node_id,
                                                                                       global_min_out_degree, global_min_out_degree_node_id)))

    neighbor_index=0
    while True:
        ## For i-th neighbor, query the owning rank for its degree
        ith_neighbors=[]

        if rank == 0 and verbose:
            print(('neighbor_degrees: rank %d: neighbor_index = %d' % (rank, neighbor_index)))
        
        for (v,ns) in neighbors_dict.items():
            if 'src' in ns:
                len_src = len(ns['src'])
            else:
                len_src = 0
            if 'dst' in ns:
                len_dst = len(ns['dst'])
            else:
                len_dst = 0
                
            if neighbor_index < len_src:
               n = ns['src'][neighbor_index]
               ith_neighbors.append(n)
            elif neighbor_index < (len_dst + len_src):
               n = ns['dst'][neighbor_index - len_src]
               ith_neighbors.append(n)

        rank_neighbor_dict = defaultdict(list)
        if len(ith_neighbors) > 0:
            for n in ith_neighbors:
                rank = node_ranks[n]
                rank_neighbor_dict[rank].append(n)
        
        ## Stop if all ranks have exhausted their lists of neighbors
        sum_len_ith_neighbors = comm.allreduce(sendobj=len(ith_neighbors), op=MPI.SUM)
        if sum_len_ith_neighbors == 0:
           break
        
        if rank == 0 and verbose:
            print(('neighbor_degrees: rank %d: len of neighbors with index %d = %d' % (rank, neighbor_index, sum_len_ith_neighbors)))

        def f (rank_degree_dict, v):
            rank = node_ranks[v]
            if rank in rank_degree_dict:
                rank_degree_dict[rank].append((v, degree_dict[v]))
            else:
                rank_degree_dict[rank] = [(v, degree_dict[v])]
            return rank_degree_dict

        query_degrees = query_alltoall(comm, f, rank_neighbor_dict)

        for degree_dict_list in query_degrees:
            if degree_dict_list is not None:
                for (v,d) in degree_dict_list:
                    degree_dict[v] = d
        
        neighbor_index += 1
        
            
    return degree_dict
        

def clustering_coefficient (comm, n_nodes, neighbors_dict, degree_dict, node_ranks, verbose=False):
    rank = comm.Get_rank()

    cc_dict = {}
    k_dict = {}
    for (v, d) in degree_dict.items():
        degree = d['total']
        if degree > 1:
            k_dict[v] = degree * (degree-1)
        else:
            k_dict[v] = degree
        
    neighbor_index=0
    while True:
        if rank == 0 and verbose:
            print(('clustering_coefficient: rank %d: neighbor_index = %d' % (rank, neighbor_index)))

        ## For i-th neighbor, query the owning rank for its neighbors
        ith_neighbors=[]
        ith_neighbors_dict={}

        for (v,ns) in neighbors_dict.items():
            if 'src' in ns:
                len_src = len(ns['src'])
            else:
                len_src = 0
            if 'dst' in ns:
                len_dst = len(ns['dst'])
            else:
                len_dst = 0
                
            if neighbor_index < len_src:
               ith_neighbors.append(ns['src'][neighbor_index])
               ith_neighbors_dict[ns['src'][neighbor_index]] = v
            elif neighbor_index < (len_dst - len_src):
               ith_neighbors.append(ns['dst'][neighbor_index - len_src])
               ith_neighbors_dict[ns['dst'][neighbor_index - len_src]] = v
                
        ## Stop if all ranks have exhausted their lists of neighbors
        sum_len_ith_neighbors = comm.allreduce(sendobj=len(ith_neighbors), op=MPI.SUM)
        if rank == 0 and verbose:
            print(('clustering_coefficient: rank %d: sum ith neighbors = %d' % (rank, sum_len_ith_neighbors)))
        if sum_len_ith_neighbors == 0:
           break

        def f (rank_ngbs_dict, v):
            rank = node_ranks[v]
            if rank in rank_ngbs_dict:
                rank_ngbs_dict[rank].append((v, neighbors_dict[v]))
            else:
                rank_ngbs_dict[rank] = [(v, neighbors_dict[v])]
            return rank_ngbs_dict

        rank_neighbor_dict = {}
        for n in ith_neighbors:
            rank = node_ranks[n]
            if rank in rank_neighbor_dict:
                rank_neighbor_dict[rank].append(n)
            else:
                rank_neighbor_dict[rank] = [n]
                        
        query_neighbors = flatten(query_alltoall(comm, f, rank_neighbor_dict))

        for (c, ngbs) in query_neighbors:
            if c in cc_dict:
                if 'dst' in neighbors_dict[c]:
                    dst_set = set(neighbors_dict[c]['dst'])
                else:
                    dst_set = set([])
                if 'src' in neighbors_dict[c]:
                    src_set = set(neighbors_dict[c]['src'])
                else:
                    src_set = set([])
                s = set(ngbs).intersection(dst_set).intersection(src_set)
                cc_dict[c] += len(s)
            else:
                cc_dict[c] = len(set(ngbs))
               
        neighbor_index += 1

    wcc = 0.0
    for (c,cc) in cc_dict.items():
        wcc += float(cc) / float(k_dict[c])
        
    sum_wcc = comm.allreduce(sendobj=wcc)

    return float(sum_wcc) / float(n_nodes)


def load_graph_networkit(comm, input_file):

    (_, n_nodes) = read_population_ranges(input_file, comm=comm)
    nhg = read_graph(input_file, comm=comm)
    g = Graph(n_nodes, False, True)

    for (presyn, prjs) in list(nhg.items()):
        for (postsyn, edges) in list(prjs.items()):
            sources = edges[0]
            destinations = edges[1]
            for (src,dst) in zip(sources,destinations):
                g.addEdge(src,dst)

    return g

