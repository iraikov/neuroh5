# neurographdf5

An HDF5-based library for storage of large-scale graphs and parallel operations on them.

## Introduction

The neurograph library implements an HDF5-based format for storing
connectivity information of large neural networks, perform parallel
graph partitioning and analysis.

neurograph assumes that synaptic connectivity between neurons in
neuronal network models is represented as directed graphs stored as
adjacency lists, where the vertices represent the neurons in the
network and are identified by unsigned integers called unique global
identifiers (gid). 

## Basic concepts and terminology

Connectivity between neurons is described in terms of vertices and
directed edges, where each vertex has an integer id associated with
it, which corresponds to the id of the respective neuron in the
network simulation. Each edge is identified by source and destination
vertex, and a number of additional attributes, such as distance and
synaptic weight. In addition, the vertices are organized in
populations of neurons, where each population is comprised of the set
of neurons that belong to the same biological type of neuron (such as
granule cell or basket cell). Connectivity is organized in
projections, where a projection is the set of connections between two
populations, or withing the same population. A projection is
identified by its source and destination populations, and all edges
between those two populations.

## Graph representation

An adjacency matrix is a square matrix where the elements of the
matrix indicate whether pairs of vertices are connected or not in the
graph. A sparse adjacency matrix representation explicitly stores only
the source or only the destination vertices, and uses range data
structures to indicate which vertices are connected. For example, in
one type of sparse format the source dataset will contain only how
many destination vertices are associated with a given source, but will
not explicitly store the source vertex ids.

Our initial implementation of an HDF5-based graph representation
includes two types of representation. The first one is a direct
edge-list-type representation where source and destination indices are
explicitly represented as HDF5 datasets. The second one is what we
refer to as Destination Block Sparse format, which is a type of sparse
adjacency matrix representation where the connectivity is additionally
divided into blocks of contiguous indices to account for potential
gaps in connectivity (i.e. ranges of indices that are not
connected). In the next section, we present the Destination Block
Sparse format, and present details of the implementation and initial
performance metrics.

## Destination Block Sparse connectivity format

In the Destination Block Sparse format the destination indices are
stored in blocks (of destinations). The following invariants hold:

1. The destination indices in a block are contiguous. 
2. The number of destinations per block may vary from block to block.

The Destination Block Sparse format consists of the following datasets:

- Source Index : This array holds the indices of all source vertices in the projection. It's length is equal to the number of edges in the projection.
- Destination Index : This array holds the first destination index in each block. Its length is equal to the number of blocks.
- Destination Block Pointer : This array holds offsets into the Destination Pointer array. Its length is equal to the number of blocks plus one. The number of destinations in block i equals:
  Destination Block Pointer[i + 1] – Destination Block Pointer[i]
The destination index of destination j in block i is Destination Index[i] + j.
- Destination Pointer : This array holds offsets into the Source Index and edge attribute datasets. Its length is equal to the sum of the destination counts in all blocks plus one. For each destination block, Destination Pointer stores one offset per destination in the block. The number of source entries for destination j of block i equals:

  Destination Pointer[Destination Block Pointer[i] + j + 1] - estination Pointer[Destination Block Pointer[i] + j]

## Edge Attributes

Several datasets are defined that hold the non-zero edge attributes of
a projection. Each edge attribute dataset is of the same length as the
Source Index datasets (i.e. the number of edges in the projection).

