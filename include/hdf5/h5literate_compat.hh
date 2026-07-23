// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file h5literate_compat.hh
///
///  Version-conditional selection of the group iteration API.
///
///  H5Literate2() and its H5L_info2_t callback info struct were introduced in
///  HDF5 1.12.0.  H5Literate2() is VOL-aware, so it must be used when available
///  so that neuroh5 can run under a custom (non-native) VOL connector.  On
///  older HDF5 releases (e.g. the system HDF5 provided by some CI images)
///  neither symbol exists, so fall back to the original H5Literate() /
///  H5L_info_t, which are equivalent under the native VOL.
///
///  Copyright (C) 2016-2024 Project NeuroH5.
//==============================================================================
#ifndef H5LITERATE_COMPAT_HH
#define H5LITERATE_COMPAT_HH

#include <hdf5.h>

#if H5_VERSION_GE(1,12,0)
#  define NEUROH5_H5LITERATE   H5Literate2
   typedef H5L_info2_t neuroh5_h5l_info_t;
#else
#  define NEUROH5_H5LITERATE   H5Literate
   typedef H5L_info_t  neuroh5_h5l_info_t;
#endif

#endif
