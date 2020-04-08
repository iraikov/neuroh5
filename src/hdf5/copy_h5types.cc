
#include <hdf5.h>
#include <string>

#include "path_names.hh"
#include "copy_tree_h5types.hh"
#include "throw_assert.hh"

namespace neuroh5
{
  namespace hdf5
  {
    /*****************************************************************************
     * Copy type for tree structures from src to dst file
     *****************************************************************************/
    int copy_tree_h5types
    (
     hid_t  src_file,
     hid_t  dst_file
     )
    {
      herr_t status;  
      
      hid_t ocpypl_id; 
      ocpypl_id = H5Pcreate(H5P_OBJECT_COPY); 
      status = H5Pset_copy_object(ocpypl_id, H5O_COPY_MERGE_COMMITTED_DTYPE_FLAG);
      throw_assert_nomsg(status == 0);
      status = H5Padd_merge_committed_dtype_path(ocpypl_id, h5types_path_join(POPULATIONS).c_str());
      throw_assert_nomsg(status == 0);
      status = H5Ocopy(src_file, H5_TYPES.c_str(), dst_file, H5_TYPES.c_str(), ocpypl_id, 
                       H5P_DEFAULT);
      throw_assert_nomsg(status == 0);
      status = H5Pclose(ocpypl_id);
      throw_assert_nomsg(status == 0);
      return status;
    }
  }
}
