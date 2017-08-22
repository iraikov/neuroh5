""" Module for parallel graph analysis operations. """

import sys, os
import os.path
import click
import itertools, functools
import numpy as np
from mpi4py import MPI 
from neuroh5.io import scatter_read_graph, bcast_graph

#from networkit import *
#from _NetworKit import GraphEvent, GraphUpdater


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


def load_graph (comm, filepath, iosize, map_type=0, node_ranks=None):
    ## obtain incoming edges
    if node_ranks is None:
        graph = scatter_read_graph(comm,filepath,io_size=iosize,attributes=False,map_type=map_type)
    else:
        graph = scatter_read_graph(comm,filepath,io_size=iosize,node_rank_map=node_ranks,attributes=False,map_type=map_type)

    n_nodes=0
    for post in graph.keys():
        for pre in graph[post].keys():
            prj = graph[post][pre]
            for n in prj.keys():
                n_nodes += 1
        
    sum_n_nodes = comm.allreduce(sendobj=n_nodes, op=MPI.SUM)
        
    return (graph, sum_n_nodes)

    

def load_neighbors (comm, filepath, iosize, node_ranks=None):

    neighbors_dict = {}

    (graph, n_nodes) = load_graph (comm, filepath, iosize, node_ranks=node_ranks)
    
    ## determine neighbors of vertex based on incoming edges
    for post in graph.keys():
        for pre in graph[post].keys():
            prj = graph[post][pre]
            for n in prj.keys():
                edges = prj[n]
                neighbors_dict[n] = {'src': edges[0]}
                
    ## obtain outgoing edges
    (graph, _) = load_graph (comm, filepath, iosize, map_type=1, node_ranks=node_ranks)

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
            
    return (neighbors_dict, n_nodes)



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
                        rank_degree_dict[rank].append((v, degree_dict[v]))
                    else:
                        rank_degree_dict[rank] = [(v, degree_dict[v])]
                        
        query_degrees = query_alltoall(comm, f, rank_neighbor_dict)
                                                
        for (v,d) in query_degrees:
            degree_dict[v] = d
        
        ## Stop if all ranks have exhausted their lists of neighbors
        sum_len_ith_neighbors = comm.allreduce(sendobj=len(ith_neighbors), op=MPI.SUM)
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
                        rank_ngbs_dict[rank].append((v, neighbors_dict[v]))
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
        sum_len_ith_neighbors = comm.allreduce(sendobj=len(ith_neighbors), op=MPI.SUM)
        if sum_len_ith_neighbors == 0:
           break
        
        neighbor_index += 1

    wcc = 0
    for (c,cc) in cc_dict.iteritems():
        wcc += cc / k_dict[c]
        
    sum_wcc = comm.allreduce(sendobj=wcc, op=MPI.SUM)

    return sum_wcc / n_nodes


def load_graph_networkit(comm, input_file):

    (nhg, n_nodes) = load_graph(comm, input_file)
    g = Graph(n_nodes, False, True)

    for (presyn, prjs) in nhg.items():
        for (postsyn, edges) in prjs.items():
            sources = edges[0]
            destinations = edges[1]
            for (src,dst) in zip(sources,destinations):
                g.addEdge(src,dst)

    return g


def profile(g):
    pf = profiling.Profile.create(g, preset="minimal")
    pf.output("HTML",".")


def degree_centrality (g):
    dd = sorted(centrality.DegreeCentrality(g).run().scores(), reverse=True)
    return dd

#import matplotlib.pyplot as plt

#def plot_degree_centrality (dd):    
#    plt.xscale("log")
#    plt.xlabel("degree")
#    plt.yscale("log")
#    plt.ylabel("number of nodes")
#    plt.plot(dd)
#    plt.show()

#cc = components.StronglyConnectedComponents(g)
#cc.run()

#print("number of components: ", cc.numberOfComponents())

