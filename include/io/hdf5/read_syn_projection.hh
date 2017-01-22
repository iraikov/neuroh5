#ifndef READ_SYN_PROJECTION_HH
#define READ_SYN_PROJECTION_HH

#include "ngh5_types.hh"

#include "hdf5.h"
#include "mpi.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

namespace ngh5
{
  namespace io
  {
    namespace hdf5
    {
      herr_t read_syn_projection
      (
       MPI_Comm                   comm,
       const std::string&         file_name,
       const std::string&         prefix,
       uint64_t&                  nedges,
       vector<NODE_IDX_T>&        dst_gid,
       vector<DST_PTR_T>&         src_gid_ptr,
       vector<NODE_IDX_T>&        src_gid,
       vector<DST_PTR_T>&         syn_id_ptr,
       vector<NODE_IDX_T>&        syn_id,
       );
    }
  }
}

#endif
