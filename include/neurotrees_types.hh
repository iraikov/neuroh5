// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file neurotrees_types.hh
///
///  Type definitions for the fundamental datatypes used in the neurotrees storage
///  format.
///
///  Copyright (C) 2016 Project Neurotrees.
//==============================================================================
#ifndef NEUROTREES_TYPES_HH
#define NEUROTREES_TYPES_HH

#include <hdf5.h>

#include <utility>
#include <string>
#include <map>
#include <tuple>
#include <vector>

#include "ngraph.hh"

#define MAX_ATTR_NAME_LEN 128

namespace neurotrees
{

  typedef float      COORD_T;
  typedef float      REALVAL_T;
  typedef int8_t     SWC_TYPE_T;
  typedef uint16_t   LAYER_IDX_T;
  typedef uint16_t   SECTION_IDX_T;
  typedef uint32_t   NODE_IDX_T;
  typedef int32_t    PARENT_NODE_IDX_T;
  typedef uint64_t   ATTR_PTR_T;
  typedef uint64_t   SEC_PTR_T;
  typedef uint64_t   TOPO_PTR_T;
  typedef uint32_t   CELL_IDX_T;

#define MPI_CELL_IDX_T MPI_UINT32_T
#define MPI_COORD_T MPI_FLOAT
#define MPI_REALVAL_T MPI_FLOAT
#define MPI_SWC_TYPE_T MPI_INT8_T
#define MPI_LAYER_IDX_T MPI_UINT16_T
#define MPI_SECTION_IDX_T MPI_UINT16_T
#define MPI_NODE_IDX_T MPI_UINT32_T
#define MPI_PARENT_NODE_IDX_T MPI_INT32_T

#define MPI_ATTR_PTR_T MPI_UINT64_T
  
  const static std::vector< std::pair<SWC_TYPE_T, std::string> > swc_type_enumeration {
    {(SWC_TYPE_T)0, "SWC_UNDEFINED"},
    {(SWC_TYPE_T)1, "SWC_SOMA"},
    {(SWC_TYPE_T)2, "SWC_AXON"},
    {(SWC_TYPE_T)3, "SWC_BASAL_DENDRITE"},
    {(SWC_TYPE_T)4, "SWC_APICAL_DENDRITE"},
    {(SWC_TYPE_T)5, "SWC_CUSTOM"}
  };

  using namespace NGraph;

  typedef uint16_t pop_t;
  
  // population range type
  typedef struct
  {
    uint64_t start;
    uint32_t count;
    uint16_t pop;
  } pop_range_t;

  
  typedef std::tuple< CELL_IDX_T,   // Tree gid
                      std::vector<SECTION_IDX_T>,   // Section id sources
                      std::vector<SECTION_IDX_T>,   // Section id destinations
                      std::vector<SECTION_IDX_T>,   // Mapping of node ids to section ids
                      std::vector<COORD_T>,     // X coordinates of nodes
                      std::vector<COORD_T>,     // Y coordinates of nodes
                      std::vector<COORD_T>,     // Z coordinates of nodes
                      std::vector<REALVAL_T>,   // Radius
                      std::vector<LAYER_IDX_T>,  // Layer
                      std::vector<PARENT_NODE_IDX_T>,  // Parent
                      std::vector<SWC_TYPE_T>  // SWC type
                      > neurotree_t;

  typedef std::map<Graph::vertex, std::vector<Graph::vertex> > contraction_map_t;

}

#endif
