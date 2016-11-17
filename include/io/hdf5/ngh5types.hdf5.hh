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

#include "hdf5.h"

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


#endif
