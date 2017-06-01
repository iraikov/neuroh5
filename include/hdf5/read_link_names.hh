#ifndef READ_LINK_NAMES_HH
#define READ_LINK_NAMES_HH

#include <hdf5.h>
#include <mpi.h>

#include <string>
#include <vector>

namespace ngh5
{
  namespace hdf5
  {
    extern void read_link_names
    (
     MPI_Comm                  comm,
     const std::string&        file_name,
     const std::string&        path,
     std::vector<std::string>& names
     );
  }
}

#endif
