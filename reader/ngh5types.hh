 #ifndef NGH5TYPES_HH
#define NGH5TYPES_HH

#include <map>
#include <tuple>
#include <vector>
#include <iostream>
#include <utility>

#include "hdf5.h"

#include "edge_attr.hh"

// Block offset type
typedef uint64_t DST_BLK_PTR_T;

// DBS offset type
typedef uint64_t DST_PTR_T;

#define DST_PTR_H5_NATIVE_T H5T_NATIVE_UINT64
#define DST_PTR_H5_FILE_T   H5T_STD_U64LE

#define DST_BLK_PTR_H5_NATIVE_T H5T_NATIVE_UINT64
#define DST_BLK_PTR_H5_FILE_T   H5T_STD_U64LE

// DBS node index type
typedef unsigned int NODE_IDX_T;

#define NODE_IDX_H5_NATIVE_T H5T_NATIVE_UINT32
#define NODE_IDX_H5_FILE_T   H5T_STD_U32LE
#define NODE_IDX_MPI_T       MPI_UINT32_T

#define LAYER_H5_NATIVE_T H5T_NATIVE_UINT8
#define LAYER_MPI_T MPI_UINT8_T

#define SEGMENT_INDEX_H5_NATIVE_T H5T_NATIVE_UINT16
#define SEGMENT_INDEX_MPI_T MPI_UINT16_T

#define SEGMENT_POINT_INDEX_H5_NATIVE_T H5T_NATIVE_UINT16
#define SEGMENT_POINT_INDEX_MPI_T MPI_UINT16_T

#define LONG_DISTANCE_H5_NATIVE_T H5T_NATIVE_FLOAT
#define TRANS_DISTANCE_H5_NATIVE_T H5T_NATIVE_FLOAT

#define DISTANCE_H5_NATIVE_T H5T_NATIVE_FLOAT
#define DISTANCE_MPI_T MPI_FLOAT

#define SYNAPTIC_WEIGHT_H5_NATIVE_T H5T_NATIVE_FLOAT
#define SYNAPTIC_WEIGHT_MPI_T MPI_FLOAT


namespace ngh5
{


// Population types

// memory type for valid population combinations

typedef uint16_t pop_t;

typedef struct
{
  uint16_t src;
  uint16_t dst;
}
pop_comb_t;

typedef struct
{
  uint64_t start;
  uint32_t count;
  uint16_t pop;
}
pop_range_t;

typedef std::map<NODE_IDX_T,std::pair<uint32_t,pop_t> > pop_range_map_t;

typedef pop_range_map_t::const_iterator pop_range_iter_t;

// Type for mapping nodes and edges in the graph to MPI ranks

typedef uint32_t rank_t;

typedef std::tuple< std::vector<NODE_IDX_T>, // source vector
                    std::vector< EdgeAttr >  // edge attribute vector,
                   > edge_tuple_t;

typedef std::map<NODE_IDX_T, edge_tuple_t> edge_map_t;

typedef edge_map_t::const_iterator edge_map_iter_t;

typedef std::map<rank_t, edge_map_t> rank_edge_map_t;

typedef rank_edge_map_t::const_iterator rank_edge_map_iter_t;

typedef std::tuple< std::vector<NODE_IDX_T>, // source vector
                    std::vector<NODE_IDX_T>, // destination vector
                    std::vector< std::pair<std::string,hid_t> >, // edge attribute name & type
                    std::vector< EdgeAttr >  // edge attribute vector
                   > prj_tuple_t;

}

#endif
