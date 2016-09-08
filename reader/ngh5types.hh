 #ifndef NGH5TYPES_HH
#define NGH5TYPES_HH

#include <map>
#include <iostream>
#include <utility>

#include "hdf5.h"


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

typedef std::map<uint32_t, vector<NODE_IDX_T> > rank_edge_map_t;

typedef rank_edge_map_t::const_iterator rank_edge_iter_t;

#endif
