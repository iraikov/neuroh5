#ifndef NUM_PROJECTION_BLOCKS_HH
#define NUM_PROJECTION_BLOCKS_HH

#include <mpi.h>
#include <hdf5.h>

#include <string>

namespace neuroh5
{
  namespace hdf5
  {

    hsize_t num_projection_blocks
    (
     MPI_Comm                   comm,
     const std::string&         file_name,
     const std::string&         src_pop_name,
     const std::string&         dst_pop_name
     );
  }
}

#endif
