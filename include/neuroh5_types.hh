// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file neuroh5_types.hh
///
///  Type definitions for the fundamental datatypes used in the API
///
///  Copyright (C) 2016-2021 Project NeuroH5.
//==============================================================================
#ifndef NEUROH5_TYPES_HH
#define NEUROH5_TYPES_HH

#include <cstdint>
#include <limits.h>
#include <utility>
#include <string>
#include <map>
#include <tuple>
#include <vector>

#include <mpi.h>
#include <hdf5.h>

// type support
#include "cereal/types/vector.hpp"
#include "cereal/types/tuple.hpp"
#include "cereal/types/set.hpp"
#include "cereal/types/map.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/complex.hpp"
#include "cereal/types/memory.hpp"

#include "ngraph.hh"
#include "attr_val.hh"
#include "attr_index.hh"
#include "compact_optional.hh"
#include "optional_value.hh"

#define MAX_ATTR_NAME_LEN 128

#if H5_VERSION_GE(1,10,5)
#define H5_HAS_PARALLEL_DEFLATE 
#endif

using namespace NGraph;

namespace neuroh5
{

#if SIZE_MAX == UCHAR_MAX
   #define MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
   #define MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
   #define MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
   #define MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
   #define MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
   #error "unknown SIZE_T"
#endif

  // node index type
  typedef unsigned int NODE_IDX_T;
  
  // cell index type
  typedef NODE_IDX_T CELL_IDX_T;
  
  // population index type
  typedef uint16_t POP_IDX_T;
  
  // MPI type of node indexes
  #define NODE_IDX_MPI_T MPI_UINT32_T
  
  // DBS offset type
  typedef uint64_t DST_PTR_T;
  
  // Block offset type
  typedef uint64_t DST_BLK_PTR_T;
        
  enum AttrType
    {
      NoneVal, SIntVal, UIntVal, FloatVal, EnumVal
    };

  struct AttrKind
  {
    AttrType type=NoneVal;
    size_t size=0;
    AttrKind() {};
    AttrKind (const AttrType& p_type, const size_t& p_size ) : type(p_type), size(p_size) {}
    AttrKind (AttrType& p_type, size_t& p_size ) : type(p_type), size(p_size) {}
    // This method lets cereal know which data members to serialize
    template<class Archive>
    void serialize(Archive & archive)
    {
      archive(type, size);
    }
    
  };
  
  enum EdgeMapType
    {
      EdgeMapDst,
      EdgeMapSrc
    };

  enum CellIndex
    {
      IndexOwner,
      IndexShared,
      IndexNone
    };

  enum CellPtrType
    {
      PtrOwner,
      PtrShared,
      PtrNone
    };

  struct CellPtr
  {
    const CellPtrType type;
    const data::optional_string shared_ptr_name;
    CellPtr (const CellPtrType& p_type, const data::optional_string& p_shared_ptr_name) : type(p_type), shared_ptr_name (p_shared_ptr_name) {}
    CellPtr (const CellPtrType& p_type) : type(p_type) {}

  };

  typedef float      COORD_T;
  typedef float      REALVAL_T;
  typedef int8_t     SWC_TYPE_T;
  typedef int8_t     LAYER_IDX_T;
  typedef uint16_t   SECTION_IDX_T;
  typedef uint32_t   NODE_IDX_T;
  typedef int32_t    PARENT_NODE_IDX_T;
  typedef uint64_t   ATTR_PTR_T;
  typedef uint64_t   SEC_PTR_T;
  typedef uint64_t   TOPO_PTR_T;
  typedef uint32_t   CELL_IDX_T;
  typedef uint32_t   NODE_IDX_T;

  #define MPI_CELL_IDX_T    MPI_UINT32_T 
  #define MPI_COORD_T       MPI_FLOAT 
  #define MPI_REALVAL_T     MPI_FLOAT 
  #define MPI_SWC_TYPE_T    MPI_INT8_T 
  #define MPI_LAYER_IDX_T   MPI_INT8_T 
  #define MPI_SECTION_IDX_T MPI_UINT16_T 
  #define MPI_NODE_IDX_T    MPI_UINT32_T 
  #define MPI_PARENT_NODE_IDX_T MPI_INT32_T 
  #define MPI_ATTR_PTR_T    MPI_UINT64_T 

  // population type
  typedef uint16_t pop_t;
  
  // population range type
  typedef struct
  {
    uint64_t start;
    uint32_t count;
    uint16_t pop;
  } pop_range_t;

  
  typedef std::tuple< CELL_IDX_T,   // Tree id
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

  // population combination type
  typedef struct
  {
    uint16_t src;
    uint16_t dst;
  }
    pop_comb_t;


  typedef std::map<pop_t, string> pop_label_map_t;
  typedef std::map<pop_t, pop_range_t > pop_range_map_t;

  typedef std::map<NODE_IDX_T, std::pair<uint32_t, pop_t> > pop_search_range_map_t;
  
  typedef pop_search_range_map_t::const_iterator pop_search_range_iter_t;

  // Type for mapping nodes and edges in the graph to MPI ranks
  
  typedef uint32_t rank_t;
  
  typedef std::tuple< std::vector<NODE_IDX_T>, // adjacency vector
                      std::vector< data::AttrVal >  // edge attribute vector,
                      > edge_tuple_t;

  typedef std::map<NODE_IDX_T, edge_tuple_t> edge_map_t;
  
  typedef edge_map_t::const_iterator edge_map_iter_t;
  
  typedef std::map<rank_t, edge_map_t> rank_edge_map_t;
  
  typedef rank_edge_map_t::const_iterator rank_edge_map_iter_t;

  typedef map<CELL_IDX_T, set<rank_t> > node_rank_map_t;
  
// In-memory HDF5 datatype of attribute pointers
#define ATTR_PTR_H5_NATIVE_T H5T_NATIVE_UINT64

// In-file HDF5 datatype of attribute pointers
#define ATTR_PTR_H5_FILE_T   H5T_STD_U64LE

// In-memory HDF5 datatype of section pointers
#define SEC_PTR_H5_NATIVE_T H5T_NATIVE_UINT64

// In-file HDF5 datatype of section pointers
#define SEC_PTR_H5_FILE_T   H5T_STD_U64LE

// In-memory HDF5 datatype of topology pointers
#define TOPO_PTR_H5_NATIVE_T H5T_NATIVE_UINT64

// In-file HDF5 datatype of topology pointers
#define TOPO_PTR_H5_FILE_T   H5T_STD_U64LE

// In-memory HDF5 datatype of node indexes
#define NODE_IDX_H5_NATIVE_T H5T_NATIVE_UINT32

// In-file HDF5 datatype of node indexes
#define NODE_IDX_H5_FILE_T   H5T_STD_U32LE

// In-memory HDF5 datatype of parent node indexes
#define PARENT_NODE_IDX_H5_NATIVE_T H5T_NATIVE_INT32

// In-file HDF5 datatype of parent node indexes
#define PARENT_NODE_IDX_H5_FILE_T   H5T_STD_I32LE

// In-memory HDF5 datatype of coordinates
#define COORD_H5_NATIVE_T H5T_NATIVE_FLOAT

// In-file HDF5 datatype of coordinates
#define COORD_H5_FILE_T   H5T_IEEE_F32LE

// In-memory HDF5 datatype of real-valued attributes
#define REAL_H5_NATIVE_T H5T_NATIVE_FLOAT

// In-file HDF5 datatype of real-valued attributes
#define REAL_H5_FILE_T   H5T_IEEE_F32LE

// In-memory HDF5 datatype of layers
#define LAYER_IDX_H5_NATIVE_T H5T_NATIVE_INT8

// In-file HDF5 datatype of layers
#define LAYER_IDX_H5_FILE_T   H5T_STD_I8LE

// In-memory HDF5 datatype of sections
#define SECTION_IDX_H5_NATIVE_T H5T_NATIVE_UINT16

// In-file HDF5 datatype of sections
#define SECTION_IDX_H5_FILE_T   H5T_STD_U16LE

// In-memory HDF5 datatype of SWC types
#define SWC_TYPE_H5_NATIVE_T H5T_NATIVE_INT8

// In-file HDF5 datatype of SWC types
#define SWC_TYPE_H5_FILE_T   H5T_STD_I8LE

// In-memory HDF5 datatype of cell index
#define CELL_IDX_H5_NATIVE_T H5T_NATIVE_UINT32

// In-file HDF5 datatype of cell index
#define CELL_IDX_H5_FILE_T   H5T_STD_U32LE

// In-memory HDF5 datatype of destination block pointers
#define DST_BLK_PTR_H5_NATIVE_T H5T_NATIVE_UINT64

// In-file HDF5 datatype of destination block pointers
#define DST_BLK_PTR_H5_FILE_T   H5T_STD_U64LE

// In-memory HDF5 datatype of destination pointers
#define DST_PTR_H5_NATIVE_T H5T_NATIVE_UINT64

// In-file HDF5 datatype of destination pointers
#define DST_PTR_H5_FILE_T   H5T_STD_U64LE

  
  
}


#endif
