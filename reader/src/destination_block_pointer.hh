// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
#ifndef DESTINATION_BLOCK_POINTER_HH
#define DESTINATION_BLOCK_POINTER_HH

#include "ngh5types.hh"

#include "hdf5.h"

#include <string>
#include <vector>

namespace ngh5
{

  extern herr_t destination_block_pointer
  (
   MPI_Comm                    comm,
   hid_t                       file,
   const std::string&          proj_name,
   const hsize_t&              start,
   const hsize_t&              block,
   std::vector<DST_BLK_PTR_T>& dst_blk_ptr,
   DST_BLK_PTR_T&              block_rebase
   );

}


#endif
