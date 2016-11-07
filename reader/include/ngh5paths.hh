#ifndef NGH5PATHS_HH
#define NGH5PATHS_HH

#include <string>

namespace ngh5
{
  struct H5PathNames
  {
    static const std::string CONN;
    
    static const std::string PRJ;

    static const std::string POP;

    static const std::string DST_BLK_PTR;

    static const std::string DST_IDX;

    static const std::string DST_POP;

    static const std::string DST_PTR;

    static const std::string POP_RNG;

    static const std::string POP_COMB;

    static const std::string SRC_IDX;

    static const std::string SRC_POP;

  };

  std::string ngh5_prj_path (const std::string& proj_name, const std::string& name);
  std::string ngh5_pop_path (const std::string& name);
};

#endif
