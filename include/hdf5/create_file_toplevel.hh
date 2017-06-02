#ifndef CREATE_FILE_TOPLEVEL_HH
#define CREATE_FILE_TOPLEVEL_HH

#include "hdf5.h"

#include <vector>

namespace neuroh5
{
  namespace hdf5
  {
    /*****************************************************************************
     * Creates an empty tree file
     *****************************************************************************/
    
    int create_file_toplevel
    (
     MPI_Comm comm,
     const std::string& file_name
     );
  }
}

#endif
