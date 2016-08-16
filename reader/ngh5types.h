#ifndef NGH5TYPES_HH
#define NGH5TYPES_HH

#include "hdf5.h"

// node index type

typedef unsigned int NODE_IDX_T;

#define NODE_IDX_H5_NATIVE_T H5T_NATIVE_UINT32
#define NODE_IDX_H5_FILE_T   H5T_STD_U32LE

// CRS offset type

typedef uint64_t ROW_PTR_T;

#define ROW_PTR_H5_NATIVE_T H5T_NATIVE_UINT64
#define ROW_PTR_H5_FILE_T   H5T_STD_U64LE

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

#endif
