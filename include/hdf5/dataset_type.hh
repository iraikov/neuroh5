#ifndef HDF5_READ_TEMPLATE
#define HDF5_READ_TEMPLATE

#include <hdf5.h>

#include <cassert>
#include <vector>
#include <string>


namespace neuroh5
{
  namespace hdf5
  {

    
    hid_t dataset_type
    (
     hid_t file,
     const std::string& path
     );

  }

}

#endif


