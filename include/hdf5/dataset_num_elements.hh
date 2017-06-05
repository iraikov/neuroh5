#ifndef DATASET_NUM_ELEMENTS_HH
#define DATASET_NUM_ELEMENTS_HH

#include <mpi.h>
#include <hdf5.h>

#include <string>

namespace neuroh5
{
  namespace hdf5
  {
    hsize_t dataset_num_elements
    (
     MPI_Comm           comm,
     const hid_t&       loc,
     const std::string& path
     );
  }
}

#endif
