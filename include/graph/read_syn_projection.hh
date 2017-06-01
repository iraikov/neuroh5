#ifndef READ_SYN_PROJECTION_HH
#define READ_SYN_PROJECTION_HH

#include <hdf5.h>
#include <mpi.h>

#include "ngh5_types.hh"

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

namespace neuroio
{
  namespace hdf5
  {
    herr_t read_syn_projection
    (
     MPI_Comm                   comm,
     const std::string&         file_name,
     const std::string&         prefix,
     std::vector<NODE_IDX_T>&   dst_gid,
     std::vector<DST_PTR_T>&    src_gid_ptr,
     std::vector<NODE_IDX_T>&   src_gid,
     std::vector<DST_PTR_T>&    syn_id_ptr,
     std::vector<NODE_IDX_T>&   syn_id
     );
  }
}

#endif
