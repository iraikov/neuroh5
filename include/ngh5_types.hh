// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file ngh5_types.hh
///
///  Type definitions for the fundamental datatypes used in the API
///
///  Copyright (C) 2016 Project Neurograph.
//==============================================================================
#ifndef NGH5_TYPES_HH
#define NGH5_TYPES_HH

#include <cstdint>

// node index type
typedef unsigned int NODE_IDX_T;

// MPI type of node indexes
#define NODE_IDX_MPI_T MPI_UINT32_T

// DBS offset type
typedef uint64_t DST_PTR_T;

// Block offset type
typedef uint64_t DST_BLK_PTR_T;

#endif
