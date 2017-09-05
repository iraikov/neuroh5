""" Module for parallel graph analysis operations. """

import sys, os
import os.path
import click
import itertools, functools
from collections import defaultdict
import numpy as np
from mpi4py import MPI 
from neuroh5.io import read_population_ranges, scatter_read_graph, bcast_graph, read_graph

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

    for rank in xrange(0,comm.Get_size()):
        if query_rank_node_dict.has_key(rank):
            sendbuf.append(query_rank_node_dict[rank])
        else:
            sendbuf.append(None)

    query_input = flatten(comm.alltoall(sendobj=sendbuf))
    
    query_local_result = functools.reduce(f, query_input, {})

    sendbuf = []
    for rank in xrange(0,comm.Get_size()):
        if query_local_result.has_key(rank):
            sendbuf.append(query_local_result[rank])
        else:
            sendbuf.append(None)

    query_result = comm.alltoall(sendobj=sendbuf)

    return query_result


def make_node_rank_map (comm, filepath, iosize):

    size = comm.Get_size()
    
    (_, n_nodes) = read_population_ranges(comm,filepath)

    node_rank_map = {}
    for i in xrange(0, n_nodes):
        node_rank_map[i] = i % size
    
    return (node_rank_map, n_nodes)


def read_neighbors (comm, filepath, iosize, node_ranks):
    
    neighbors_dict = {}

    graph = scatter_read_graph (comm, filepath, io_size=iosize, map_type=0, attributes=False,
                                node_rank_map=node_ranks)
    
    ## determine neighbors of vertex based on incoming edges
    for post in graph.keys():
        for pre in graph[post].keys():
            prj = graph[post][pre]
            for n in prj.keys():
                edges = prj[n]
                neighbors_dict[n] = {'src': edges[0]}
                
    ## obtain outgoing edges
    graph = scatter_read_graph (comm, filepath, io_size=iosize, map_type=1, attributes=False,
                                node_rank_map=node_ranks)

    ## determine neighbors of vertex based on outgoing edges
    for pre in graph.keys():
        for post in graph[pre].keys():
            prj = graph[pre][post]
            for n in prj.keys():
                edges = prj[n]
                if neighbors_dict.has_key(n):
                    neighbors_dict[n]['dst'] = edges[0]
                else:
                    neighbors_dict[n] = {'dst': edges[0]}
            
    return neighbors_dict


def neighbor_degrees (comm, neighbors_dict, node_ranks):

    rank = comm.Get_rank()

    degree_dict = {}

    min_total_degree=sys.maxint
    max_total_degree=0
    min_in_degree=sys.maxint
    max_in_degree=0
    min_out_degree=sys.maxint
    max_out_degree=0

    for (v,ns) in neighbors_dict.iteritems():
        if ns.has_key('src'):
            in_degree = np.size(ns['src'])
        else:
            in_degree = 0
        if ns.has_key('dst'):
            out_degree = np.size(ns['dst'])
        else:
            out_degree = 0
        min_total_degree = min(min_total_degree, in_degree+out_degree)
        max_total_degree = max(max_total_degree, in_degree+out_degree)
        min_in_degree  = min(min_in_degree, in_degree)
        max_in_degree  = max(max_in_degree, in_degree)
        min_out_degree = min(min_out_degree, out_degree)
        max_out_degree = max(max_out_degree, out_degree)
        degree_dict[v] = {'total': in_degree+out_degree, 'in': in_degree, 'out': out_degree}

    global_min_total_degree = comm.allreduce(sendobj=min_total_degree, op=MPI.MIN)
    global_max_total_degree = comm.allreduce(sendobj=max_total_degree, op=MPI.MAX)
    global_min_in_degree    = comm.allreduce(sendobj=min_in_degree, op=MPI.MIN)
    global_max_in_degree    = comm.allreduce(sendobj=max_in_degree, op=MPI.MAX)
    global_min_out_degree   = comm.allreduce(sendobj=min_out_degree, op=MPI.MIN)
    global_max_out_degree   = comm.allreduce(sendobj=max_out_degree, op=MPI.MAX)
    if rank == 0:
        print 'neighbor_degrees: max degrees: total=%d in=%d out=%d' % (global_max_total_degree, global_max_in_degree, global_max_out_degree)
        print 'neighbor_degrees: min degrees: total=%d in=%d out=%d' % (global_min_total_degree, global_min_in_degree, global_min_out_degree)

    neighbor_index=0
    while True:
        ## For i-th neighbor, query the owning rank for its degree
        ith_neighbors=[]

        if rank == 0:
            print 'neighbor_degrees: rank %d: neighbor_index = %d' % (rank, neighbor_index)
        
        for (v,ns) in neighbors_dict.iteritems():
            if ns.has_key('src'):
                len_src = np.size(ns['src'])
            else:
                len_src = 0
            if ns.has_key('dst'):
                len_dst = np.size(ns['dst'])
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
        
        if rank == 0:
            print 'neighbor_degrees: rank %d: len of neighbors with index %d = %d' % (rank, neighbor_index, sum_len_ith_neighbors)

        def f (rank_degree_dict, v):
            rank = node_ranks[v]
            if rank_degree_dict.has_key(rank):
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
        

def clustering_coefficient (comm, n_nodes, neighbors_dict, degree_dict, node_ranks):
    rank = comm.Get_rank()

    cc_dict = {}
    k_dict = {}
    for (v, d) in degree_dict.iteritems():
        degree = d['total']
        if degree > 1:
            k_dict[v] = degree * (degree-1)
        else:
            k_dict[v] = degree
        
    neighbor_index=0
    while True:
        if rank == 0:
            print 'clustering_coefficient: rank %d: neighbor_index = %d' % (rank, neighbor_index)

        ## For i-th neighbor, query the owning rank for its neighbors
        ith_neighbors=[]
        ith_neighbors_dict={}

        for (v,ns) in neighbors_dict.iteritems():
            if ns.has_key('src'):
                len_src = np.size(ns['src'])
            else:
                len_src = 0
            if ns.has_key('dst'):
                len_dst = np.size(ns['dst'])
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
        if rank == 0:
            print 'clustering_coefficient: rank %d: sum ith neighbors = %d' % (rank, sum_len_ith_neighbors)
        if sum_len_ith_neighbors == 0:
           break

        def f (rank_ngbs_dict, v):
            rank = node_ranks[v]
            if rank_ngbs_dict.has_key(rank):
                rank_ngbs_dict[rank].append((v, neighbors_dict[v]))
            else:
                rank_ngbs_dict[rank] = [(v, neighbors_dict[v])]
            return rank_ngbs_dict

        rank_neighbor_dict = {}
        for n in ith_neighbors:
            rank = node_ranks[n]
            if rank_neighbor_dict.has_key(rank):
                rank_neighbor_dict[rank].append(n)
            else:
                rank_neighbor_dict[rank] = [n]
                        
        query_neighbors = flatten(query_alltoall(comm, f, rank_neighbor_dict))

        for (c, ngbs) in query_neighbors:
            if cc_dict.has_key(c):
                if neighbors_dict[c].has_key('dst'):
                    dst_set = set(neighbors_dict[c]['dst'])
                else:
                    dst_set = set([])
                if neighbors_dict[c].has_key('src'):
                    src_set = set(neighbors_dict[c]['src'])
                else:
                    src_set = set([])
                s = set(ngbs).intersection(dst_set).intersection(src_set)
                cc_dict[c] += len(s)
            else:
                cc_dict[c] = len(set(ngbs))
               
        neighbor_index += 1

    wcc = 0.0
    for (c,cc) in cc_dict.iteritems():
        wcc += float(cc) / float(k_dict[c])
        
    sum_wcc = comm.allreduce(sendobj=wcc)

    return float(sum_wcc) / float(n_nodes)


def load_graph_networkit(comm, input_file):

    (_, n_nodes) = read_population_ranges(comm, input_file)
    nhg = read_graph(comm, input_file)
    g = Graph(n_nodes, False, True)

    for (presyn, prjs) in nhg.items():
        for (postsyn, edges) in prjs.items():
            sources = edges[0]
            destinations = edges[1]
            for (src,dst) in zip(sources,destinations):
                g.addEdge(src,dst)

    return g

