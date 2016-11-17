// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file ngh5types.hh
///
///  Type definitions for the fundamental datatypes used in the graph storage
///  format.
///
///  Copyright (C) 2016 Project Neurograph.
//==============================================================================
#ifndef NGH5TYPES_HH
#define NGH5TYPES_HH

#include "edge_attr.hh"

#include "hdf5.h"

#include <map>
#include <tuple>
#include <vector>

// Block offset type
typedef uint64_t DST_BLK_PTR_T;

// In-memory HDF5 datatype of destination block pointers
#define DST_BLK_PTR_H5_NATIVE_T H5T_NATIVE_UINT64

// In-file HDF5 datatype of destination block pointers
#define DST_BLK_PTR_H5_FILE_T   H5T_STD_U64LE

// DBS offset type
typedef uint64_t DST_PTR_T;

// In-memory HDF5 datatype of destination pointers
#define DST_PTR_H5_NATIVE_T H5T_NATIVE_UINT64

// In-file HDF5 datatype of destination pointers
#define DST_PTR_H5_FILE_T   H5T_STD_U64LE

// DBS node index type
typedef unsigned int NODE_IDX_T;

// In-memory HDF5 datatype of node indexes
#define NODE_IDX_H5_NATIVE_T H5T_NATIVE_UINT32

// In-file HDF5 datatype of node indexes
#define NODE_IDX_H5_FILE_T   H5T_STD_U32LE

// MPI type of node indexes
#define NODE_IDX_MPI_T       MPI_UINT32_T


namespace ngh5
{

  // Population types

  typedef uint16_t pop_t;

  // population combination type
  typedef struct
  {
    uint16_t src;
    uint16_t dst;
  }
    pop_comb_t;

  // population range type
  typedef struct
  {
    uint64_t start;
    uint32_t count;
    uint16_t pop;
  }
    pop_range_t;

  //
  // ???
  //

  typedef std::map<NODE_IDX_T,std::pair<uint32_t,pop_t> > pop_range_map_t;

  typedef pop_range_map_t::const_iterator pop_range_iter_t;

  // Type for mapping nodes and edges in the graph to MPI ranks

  typedef uint32_t rank_t;

  typedef std::tuple< std::vector<NODE_IDX_T>, // source vector
                      EdgeAttr  // edge attribute vector,
                      > edge_tuple_t;

  typedef std::map<NODE_IDX_T, edge_tuple_t> edge_map_t;

  typedef edge_map_t::const_iterator edge_map_iter_t;

  typedef std::map<rank_t, edge_map_t> rank_edge_map_t;

  typedef rank_edge_map_t::const_iterator rank_edge_map_iter_t;

  typedef std::tuple< std::vector<NODE_IDX_T>, // source vector
                      std::vector<NODE_IDX_T>, // destination vector
                      EdgeAttr  // edge attribute vector
                      > prj_tuple_t;

}

#endif
