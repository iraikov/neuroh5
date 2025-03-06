# neuroh5

A parallel HDF5-based library for storage and processing of large-scale graphs and neural cell model attributes.

## Introduction

The neuroh5 library implements an HDF5-based format for storing
neuronal morphology information, synaptic and connectivity information
of large neural networks, and perform parallel graph partitioning and
analysis.

neuroh5 assumes that synaptic connectivity between neurons in
neuronal network models is represented as directed graphs stored as
adjacency lists, where the vertices represent the neurons in the
network and are identified by unsigned integers called unique global
identifiers (gid). 

## Installation

Building and installing NeuroH5 

NeuroH5 requires parallel HDF5, MPI, cmake. The Python module requires Python 3 and numpy.

To build the NeuroH5 C++ library and applications:

```
git clone https://github.com/iraikov/neuroh5.git
cd neuroh5
cmake .
make 
```


To build the python module:

```
git clone https://github.com/iraikov/neuroh5.git
cd neuroh5
CMAKE_BUILD_PARALLEL_LEVEL=8 \
  CMAKE_MPI_C_COMPILER=$(which mpicc) \
  CMAKE_MPI_CXX_COMPILER=$(which mpicxx) \
  pip install .
```


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

## Destination Block Sparse (DBS) Format

The Destination Block Sparse (DBS) format is a memory-efficient graph
representation designed for parallel processing of large-scale neural
network connectivity. This format optimizes for the common case where
each destination (target neuron) connects to a relatively small subset
of sources (input neurons).

### Core Data Structures

The DBS format consists of four primary arrays:

1. **Source Index Array (`src_idx`)**: 
   - Contains the indices of all source vertices in the projection
   - Length equals the total number of edges (connections) in the projection
   - Stores the actual connectivity information

2. **Destination Block Pointer Array (`dst_blk_ptr`)**: 
   - Contains offsets into the Destination Pointer array
   - Length equals the number of blocks plus one (includes a sentinel value)
   - The difference between consecutive elements indicates the number of destinations in each block

3. **Destination Index Array (`dst_idx`)**: 
   - Contains the first destination index in each block
   - Length equals the number of blocks
   - Destinations within a block have contiguous indices

4. **Destination Pointer Array (`dst_ptr`)**: 
   - Contains offsets into the Source Index array
   - Length equals the total number of destinations plus one (includes a sentinel value)
   - Indicates where each destination's source connections begin and end

### Key Properties and Relationships

- **Block Structure**: Destinations are organized into blocks where each block contains contiguous destination indices
- **Variable Block Size**: The number of destinations per block can vary
- **Contiguous Destinations**: All destinations within a block have contiguous indices
- **Efficient Edge Lookup**: To find all sources connected to a specific destination:
   1. Locate the block containing the destination
   2. Calculate the destination's offset within the block
   3. Use the offset to find the appropriate pointers in `dst_ptr`
   4. Access the source indices from `src_idx`

### Formal Relationships

For a given block index `i`:
- Number of destinations in block `i` = `dst_blk_ptr[i+1] - dst_blk_ptr[i]`
- Destination index of the j-th destination in block `i` = `dst_idx[i] + j`
- For the j-th destination in block `i`:
  - Offset into `dst_ptr` = `dst_blk_ptr[i] + j`
  - Source index range starts at `src_idx[dst_ptr[dst_blk_ptr[i] + j]]`
  - Source index range ends at `src_idx[dst_ptr[dst_blk_ptr[i] + j + 1] - 1]`
  - Number of sources = `dst_ptr[dst_blk_ptr[i] + j + 1] - dst_ptr[dst_blk_ptr[i] + j]`

## Benefits for Parallel Processing

This format is particularly well-suited for parallel processing because:
1. It clusters related destinations into blocks, improving cache locality
2. It allows for balanced distribution of computational load across processors
3. It minimizes communication overhead when distributing graph data
4. It provides efficient access patterns for both forward and backward traversals

The format achieves memory efficiency by using index arrays and offset
pointers rather than storing a full adjacency matrix, making it ideal
for sparse connectivity patterns typical in neural networks.

## Edge Attributes

Several datasets are defined that hold the non-zero edge attributes of
a projection. Each edge attribute dataset is of the same length as the
Source Index datasets (i.e. the number of edges in the projection).

