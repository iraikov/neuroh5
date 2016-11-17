// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
#ifndef NGH5PATHS_HH
#define NGH5PATHS_HH

#include <string>

namespace ngh5
{
  struct H5PathNames
  {
    // Connectivity
    static const std::string CONN;

    // Destination Block Pointer
    static const std::string DST_BLK_PTR;

    // Destination Block Index
    static const std::string DST_BLK_IDX;

    // Destination Population
    static const std::string DST_POP;

    // Destination Pointer
    static const std::string DST_PTR;

    // '/'
    static const std::string H5_PATH_SEP;

    // H5Types
    static const std::string H5_TYPES;

    // Populations
    static const std::string POP;

    // Population Range
    static const std::string POP_RNG;

    // Valid population projections
    static const std::string POP_COMB;

    // Projections
    static const std::string PRJ;

    // Source Index
    static const std::string SRC_IDX;

    // Source Population
    static const std::string SRC_POP;

  };

  extern std::string ngh5_prj_path
  (
   const std::string& proj_name,
   const std::string& name
   );

  extern std::string ngh5_pop_path(const std::string& name);
};

#endif
