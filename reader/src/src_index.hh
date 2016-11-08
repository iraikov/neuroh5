// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
#ifndef SOURCE_INDEX_HH
#define SOURCE_INDEX_HH

#include "ngh5types.hh"

#include "hdf5.h"

#include <string>
#include <vector>

namespace ngh5
{

  extern herr_t source_index
  (
   MPI_Comm                      comm,
   hid_t                         file,
   const std::string&            proj_name,
   const hsize_t&                in_block,
   const DST_PTR_T&              dst_rebase,
   const std::vector<DST_PTR_T>& dst_ptr,
   std::vector<NODE_IDX_T>&      src_idx
   );

}


#endif
