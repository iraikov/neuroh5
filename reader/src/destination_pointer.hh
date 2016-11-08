// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
#ifndef DESTINATION_POINTER_HH
#define DESTINATION_POINTER_HH

#include "ngh5types.hh"

#include "hdf5.h"

#include <string>
#include <vector>

namespace ngh5
{

  herr_t destination_pointer
  (
   MPI_Comm                          comm,
   hid_t                             file,
   const std::string&                proj_name,
   const hsize_t&                    in_block,
   const DST_BLK_PTR_T&              block_rebase,
   const std::vector<DST_BLK_PTR_T>& dst_blk_ptr,
   std::vector<DST_PTR_T>&           dst_ptr,
   DST_PTR_T&                        edge_base,
   DST_PTR_T&                        dst_rebase
   );

}


#endif
