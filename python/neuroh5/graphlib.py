""" Module for parallel graph analysis operations. """


import sys, os
import os.path
import click
import itertools, functools
import numpy as np
from mpi4py import MPI # Must come before importing NEURON
from neuron import h
from neurograph.io import scatter_graph, bcast_graph

def load_graph (comm, filepath, iosize, map_type=0, node_ranks):
    ## obtain incoming edges
    if node_ranks is None:
        graph = scatter_graph(MPI._addressof(comm),filepath,iosize,map_type=map_type)
    else:
        graph = scatter_graph(MPI._addressof(comm),filepath,iosize,node_rank_map=node_ranks,map_type=map_type)

    n_nodes=0
    for prj in graph:
        n_nodes += len(prj)
        
    sum_n_nodes = comm.Allreduce(len(n_nodes), op=MPI.SUM)
        
    return (graph, n_nodes)

def query_alltoall (comm, f, query_rank_node_dict):
    sendbuf=[]
    ## send out the vertices whose degrees we wish to query
    if len(query_rank_node_dict) > 0:
        for rank in xrange(0,comm.Size()):
            sendbuf.append(query_rank_node_dict[rank])
    else:
        for rank in xrange(0,comm.Size()):
            sendbuf.append(None)

    query_nodes = comm.alltoall(sendbuf)

    query_local_result = functools.reduce(f, query_nodes, {})

    sendbuf = []
    for rank in xrange(0,comm.Size()):
        if query_local_result.has_key(rank):
            sendbuf.append(rank_degree_dict[rank])
        else:
            sendbuf.append(None)

    query_result = comm.alltoall(sendbuf)

    return query_result
    

def neighbors (comm, filepath, iosize, node_ranks):

    neighbors_dict = {}

    (graph, a) = loadGraph (comm, filepath, iosize, node_ranks=node_ranks)
    
    ## determine neighbors of vertex based on incoming edges
    for (name, prj) in graph.iteritems():
        for dst in prj:
            edges   = prj[dst]
            srcs    = edges[0]
            neighbors_dict[dst] = {'src': srcs}
            del prj[dst]

    ## obtain outgoing edges
    (graph, a) = loadGraph (comm, filepath, iosize, map_type=1, node_ranks)

    ## determine neighbors of vertex based on outgoing edges
    for (name, prj) in graph.iteritems():
        for src in prj:
            edges   = prj[src]
            dsts    = edges[0]
            if neighbors_dict.has_key(src):
                neighbors_dict[src]['dst'] = dsts
            else:
                neighbors_dict[src] = {'dst': dsts}
            del prj[dst]
            
    return neighbors_dict


def prj_neighbors (comm, filepath, iosize, node_ranks):

    neighbors_dict = {}

    (graph, a) = loadGraph (comm, filepath, iosize, node_ranks=node_ranks)
    
    ## determine neighbors of vertex based on incoming edges
    for (name, prj) in graph.iteritems():
        neighbors_dict_prj = {}
        for dst in prj:
            edges   = prj[dst]
            srcs    = edges[0]
            neighbors_dict_prj[dst] = {'src': srcs}
            del prj[dst]
        neighbors_dict[name] = neighbors_dict_prj

    ## obtain outgoing edges
    (graph, a) = loadGraph (comm, filepath, iosize, map_type=1, node_ranks)

    ## determine neighbors of vertex based on outgoing edges
    for (name, prj) in graph.iteritems():
        neighbors_dict_prj = neighbors_dict[name]
        for src in prj:
            edges   = prj[src]
            dsts    = edges[0]
            if neighbors_dict_prj.has_key(src):
                neighbors_dict_prj[src]['dst'] = dsts
            else:
                neighbors_dict_prj[src]['dst'] = dsts
            del prj[dst]
            
    return neighbors_dict


def neighbor_degrees (comm, neighbors_dict, node_ranks):

    degree_dict = {}
    
    for (v,ns) in neighbors_dict.iteritems():
        in_degree = np.size(ns['src'])
        out_degree = np.size(ns['dst'])
        degree_dict[v] = {'total': in_degree+out_degree, 'in': in_degree, 'out': out_degree}

    neighbor_index=0
    while True:
        ## For i-th neighbor, query the owning rank for its degree
        ith_neighbors=[]
        
        for (v,ns) in neighbors_dict.iteritems():
            if neighbor_index < np.size(ns):
                ith_neighbors.append(ns[neighbor_index])

        rank_neighbor_dict = {}
        if len(ith_neighbors) > 0:
            for n in ith_neighbors:
                rank = node_ranks[n]
                if rank_neighbor_dict.has_key(rank):
                    rank_neighbor_dict[rank].append(n)
                else:
                    rank_neighbor_dict[rank] = [n]
        
        def f (ns, rank_degree_dict):
            if ns is not None:
                for v in ns:
                    rank = node_ranks[v]
                    if rank_degree_dict.has_key(rank):
                        rank_degree_dict[rank].append((v, degree_dict[v])
                    else:
                        rank_degree_dict[rank] = [(v, degree_dict[v])]
                        
        query_degrees = query_alltoall(comm, f, rank_neighbor_dict)
                                                
        for (v,d) in query_degrees:
            degree_dict[v] = d
        
        ## Stop if all ranks have exhausted their lists of neighbors
        sum_len_ith_neighbors = comm.Allreduce(len(ith_neighbors), op=MPI.SUM)
        if sum_len_ith_neighbors == 0:
           break
        
        neighbor_index += 1
        
            
    return degree_dict
            
        

def clustering_coefficient (comm, n_nodes, neighbors_dict, degree_dict, node_ranks):

    k_dict = {}
    for (v, ns) in neighbors_dict.iteritems():
        ## sort neighbors by their degree, largest first
        ns.sort(key=lambda x: degree_dict[x]['total'], reverse=True)
        k_dict[v] = len(ns) * (len(ns)-1)
        
    neighbor_index=0
    while True:
        ## For i-th neighbor, query the owning rank for its neighbors
        ith_neighbors=[]
        ith_neighbors_dict={}
        
        for (v,ns) in neighbors_dict.iteritems():
            if neighbor_index < np.size(ns):
                ith_neighbors.append(ns[neighbor_index])
                ith_neighbors_dict[ns[neighbor_index]] = v
                
        if len(ith_neighbors) > 0:

            rank_neighbor_dict = {}
            for n in ith_neighbors:
                rank = node_ranks[n]
                if rank_neighbor_dict.has_key(rank):
                    rank_neighbor_dict[rank].append(n)
                else:
                    rank_neighbor_dict[rank] = [n]

        def f (ns, rank_ngbs_dict):
            if ns is not None:
                for v in ns:
                    rank = node_ranks[v]
                    if rank_ngbs_dict.has_key(rank):
                        rank_ngbs_dict[rank].append((v, neighbors_dict[v])
                    else:
                        rank_ngbs_dict[rank] = [(v, neighbors_dict[v])]
                        
        query_neighbors = query_alltoall(comm, f, rank_neighbor_dict)

        for (v,ngbs) in query_neighbors:
            c = ith_neighbors_dict[v]
            if cc_dict.has_key(c):
               s = set(ngbs).intersection(set(neighbors_dict[c]['out'])).intersection(set(neighbors_dict[c]['in']))
               cc_dict[c] += len(s)
            else:
               cc_dict[c] = len(set(ngbs))
               
        ## Stop if all ranks have exhausted their lists of neighbors
        sum_len_ith_neighbors = comm.Allreduce(len(ith_neighbors), op=MPI.SUM)
        if sum_len_ith_neighbors == 0:
           break
        
        neighbor_index += 1

    wcc = 0
    for (c,cc) in cc_dict.iteritems():
        wcc += cc / k_dict[c]
        
    sum_wcc = comm.Allreduce(wcc, op=MPI.SUM)

    return sum_wcc / n_nodes

    



def dfs (comm, neighbors_dict, node_ranks):

    path_len_dict = {}

    path_len = 1
    path_len_dict[path_len] = 0
    for (v,ns) in neighbors_dict.iteritems():
        path_len_dict[path_len] += np.size(ns)

    visited = set([])
    neighbor_index=0
    frontier = [neighbors_dict]
    neighbor_index_stack = [neighbor_index]
    
    while True:

        ## For i-th neighbor, query the owning rank for its neighbors
        ith_neighbors=[]
        
        for (v,ns) in frontier[-1].iteritems():
            if neighbor_index < np.size(ns):
                ith_neighbors.append(ns[neighbor_index])


        if len(ith_neighbors) == 0:
            rank_neighbor_dict = {}
            for n in ith_neighbors:
                rank = node_ranks[n]
                if rank_neighbor_dict.has_key(rank):
                    rank_neighbor_dict[rank].append(n)
                else:
                    rank_neighbor_dict[rank] = [n]

        def f (ns, rank_ngbs_dict):
            if ns is not None:
                ndict = frontier[-1]
                for v in ns:
                    rank = node_ranks[v]
                    if rank_ngbs_dict.has_key(rank):
                        rank_ngbs_dict[rank].append((v, filter(lambda x: x is not in visited, ndict[v]['out']))
                    else:
                        rank_ngbs_dict[rank] = [(v, filter(lambda x: x is not in visited, ndict[v]['out']))]
                        
        query_neighbors = query_alltoall(comm, f, rank_neighbor_dict)

                    
        path_len += 1
        path_len_dict[path_len] = 0

        for (v,out_ngbs) in query_neighbors:
            visited.update(out_ngbs)
            path_len_dict[path_len] += len(out_ngbs)

        frontier.append(query_neighbors)
        neighbor_index_stack.append(neighbor_index)
        
        ## Stop if all ranks have exhausted their lists of neighbors
        sum_len_ith_neighbors = comm.Allreduce(len(ith_neighbors), op=MPI.SUM)
        if sum_len_ith_neighbors == 0:
           if len(frontier) > 1:
              frontier.pop()
              neighbor_index = neighbor_index_stack.pop() + 1
           else:
              break
        else:
           
           neighbor_index += 1
        
        
            
    return degree_dict
