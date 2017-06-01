// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file hdf5_types.hh
///
///  Type definitions for the fundamental datatypes used in the NeuroH5 format.
///
///  Copyright (C) 2016-2017 Project Neurotrees.
//==============================================================================
#ifndef HDF5_TYPES_HH
#define HDF5_TYPES_HH

#include <hdf5.h>


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
#define LAYER_IDX_H5_NATIVE_T H5T_NATIVE_UINT16

// In-file HDF5 datatype of layers
#define LAYER_IDX_H5_FILE_T   H5T_STD_U16LE

// In-memory HDF5 datatype of sections
#define SECTION_IDX_H5_NATIVE_T H5T_NATIVE_UINT16

// In-file HDF5 datatype of sections
#define SECTION_IDX_H5_FILE_T   H5T_STD_U16LE

// In-memory HDF5 datatype of SWC types
#define SWC_TYPE_H5_NATIVE_T H5T_NATIVE_INT8

// In-file HDF5 datatype of SWC types
#define SWC_TYPE_H5_FILE_T   H5T_STD_I8LE

// In-memory HDF5 datatype of gid
#define CELL_IDX_H5_NATIVE_T H5T_NATIVE_UINT32

// In-file HDF5 datatype of gid
#define CELL_IDX_H5_FILE_T   H5T_STD_U32LE

#endif
