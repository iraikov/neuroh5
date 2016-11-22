#ifndef DATASET_NUM_ELEMENTS_HH
#define DATASET_NUM_ELEMENTS_HH

#include <hdf5.h>
#include <mpi.h>

#include <string>

namespace ngh5
{
  namespace io
  {
    namespace hdf5
    {
      hsize_t dataset_num_elements
      (
       MPI_Comm           comm,
       const std::string& file_name,
       const std::string& path
       );
    }
  }
}

#endif
