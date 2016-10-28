#ifndef NGH5PATHS_HH
#define NGH5PATHS_HH

// Get rid of these!

#define DST_BLK_PTR_H5_PATH "/Connectivity/Destination Block Pointer"

#define DST_IDX_H5_PATH "/Connectivity/Destination Index"

#define DST_PTR_H5_PATH "/Connectivity/Destination Pointer"

#define SRC_IDX_H5_PATH "/Connectivity/Source Index"

#define DST_POP_H5_PATH "/Destination Population"

#define SRC_POP_H5_PATH "/Source Population"

#define POP_COMB_H5_PATH "Valid population projections"

#define POP_RANGE_H5_PATH "Populations"

#define CONNECTIVITY_H5_PATH "Connectivity"

#include <string>

namespace ngh5
{
  class H5PathNames
  {
  public:

    static const std::string CONN;

    static const std::string DST_BLK_PTR;

    static const std::string DST_IDX;

    static const std::string DST_POP;

    static const std::string DST_PTR;

    static const std::string POP_RNG;

    static const std::string POP_COMB;

    static const std::string SRC_IDX;

    static const std::string SRC_POP

  }
};

#endif
