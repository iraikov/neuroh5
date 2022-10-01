#ifndef HDF5_EDGE_ATTRIBUTES
#define HDF5_EDGE_ATTRIBUTES

#include <hdf5.h>

#include <vector>

#include "neuroh5_types.hh"
#include "path_names.hh"
#include "rank_range.hh"
#include "dataset_num_elements.hh"
#include "create_group.hh"
#include "read_template.hh"
#include "write_template.hh"
#include "file_access.hh"
#include "throw_assert.hh"
#include "mpe_seq.hh"
#include "debug.hh"

namespace neuroh5
{
  namespace hdf5
  {

    
    void size_edge_attributes
    (
     hid_t          loc,
     const string&  src_pop_name,
     const string&  dst_pop_name,
     const string&  attr_namespace,
     const string&  attr_name,
     hsize_t&       value_size
     );

    void create_projection_groups
    (
     const hid_t&   file,
     const string&  src_pop_name,
     const string&  dst_pop_name
     );
    
    void create_edge_attribute_datasets
    (
     const hid_t&   file,
     const string&  src_pop_name,
     const string&  dst_pop_name,
     const string&  attr_namespace,
     const string&  attr_name,
     const hid_t&   ftype,
     const size_t   chunk_size
     );

    
    template <typename T>
    void append_edge_attribute
    (
     MPI_Comm       comm,
     const hid_t&   loc,
     const string&  src_pop_name,
     const string&  dst_pop_name,
     const string&  attr_namespace,
     const string&  attr_name,
     const std::vector<T>& value
     )
    {
      int status;

      hid_t file = H5Iget_file_id(loc);
      throw_assert(file >= 0, "error in H5Iget_file_id");
    
      int ssize, srank; size_t size, rank;
      throw_assert(MPI_Comm_size(comm, &ssize) == MPI_SUCCESS, "error in MPI_Comm_size");
      throw_assert(MPI_Comm_rank(comm, &srank) == MPI_SUCCESS, "error in MPI_Comm_rank");
      throw_assert(ssize > 0, "invalid MPI comm size");
      throw_assert(srank >= 0, "invalid MPI comm rank");
      size = ssize;
      rank = srank;

      T dummy;
      hid_t ftype = infer_datatype(dummy);
      throw_assert(ftype >= 0, "error in infer_datatype");
      hid_t mtype = H5Tget_native_type(ftype, H5T_DIR_ASCEND);
      throw_assert(mtype >= 0, "error in H5Tget_native_type");

      hsize_t current_value_size;
      
      size_edge_attributes (loc,
                            src_pop_name,
                            dst_pop_name,
                            attr_namespace,
                            attr_name,
                            current_value_size);

      size_t my_count = value.size();
      std::vector<size_t> all_counts(size, 0);
      throw_assert(MPI_Allgather(&my_count, 1, MPI_SIZE_T, &all_counts[0], 1,
                                 MPI_SIZE_T, comm) == MPI_SUCCESS,
                   "append_edge_attribute: error in MPI_Allgather");
      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "append_edge_attribute: error in MPI_Barrier");

      // calculate the total dataset size and the offset of my piece
      hsize_t local_value_start = current_value_size,
        global_value_size = current_value_size,
        local_value_size = my_count;
      
      for (size_t p = 0; p < size; ++p)
        {
          if (p < rank)
            {
              local_value_start += (hsize_t) all_counts[p];
            }
          global_value_size += (hsize_t) all_counts[p];
        }

      if (global_value_size - current_value_size > 0)
        {
          /* Create property list for collective dataset write. */
          hid_t wapl = H5Pcreate (H5P_DATASET_XFER);
#ifdef HDF5_IS_PARALLEL
          status = H5Pset_dxpl_mpio (wapl, H5FD_MPIO_COLLECTIVE);
          throw_assert(status >= 0,
                       "append_edge_attribute: error in H5Pset_dxpl_mpio");
#endif
          
          string path = edge_attribute_path(src_pop_name, dst_pop_name,
                                            attr_namespace, attr_name);
          
          status = write<T> (file, path,
                             global_value_size, local_value_start, local_value_size,
                             mtype, value, wapl);
          
          throw_assert(H5Pclose(wapl) >= 0, "error in H5Pclose");
        }

      throw_assert(H5Tclose(mtype) >= 0, "error in H5Tclose");

      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "append_edge_attribute: error in MPI_Barrier");

      status = H5Fclose (file);
      throw_assert(status >= 0, "error in H5Fclose");

    }

  }
}

#endif
