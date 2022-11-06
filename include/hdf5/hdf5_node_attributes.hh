#ifndef HDF5_NODE_ATTRIBUTES
#define HDF5_NODE_ATTRIBUTES

#include <hdf5.h>

#include <vector>

#include "neuroh5_types.hh"
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

    void size_node_attributes
    (
     MPI_Comm         comm,
     hid_t            loc,
     const string&    path,
     hsize_t&         ptr_size,
     hsize_t&         index_size,
     hsize_t&         value_size
     );

    void create_node_attribute_datasets
    (
     const hid_t&   file,
     const string&  attr_namespace,
     const string&  attr_name,
     const hid_t&   ftype,
     const size_t   chunk_size,
     const size_t   value_chunk_size
     );


    template <typename T>
    herr_t read_node_attribute
    (
     MPI_Comm                  comm,
     const hid_t&              loc,
     const std::string&        path,
     std::vector<NODE_IDX_T>&  index,
     std::vector<ATTR_PTR_T>&  ptr,
     std::vector<T> &          values,
     size_t offset = 0,
     size_t numitems = 0
     )
    {
      herr_t status = 0;

      int size, rank;
      throw_assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS, "error in MPI_Comm_size");
      throw_assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS, "error in MPI_Comm_rank");

      hsize_t dset_size = dataset_num_elements (loc, path + "/" + NODE_INDEX);
      size_t read_size = 0;
      if (numitems > 0) 
        {
          if (offset < dset_size)
            {
              read_size = min((hsize_t)numitems*size, dset_size-offset);
            }
        }
      else
        {
          read_size = dset_size;
        }

    
      if (read_size > 0)
        {
          // determine which blocks of block_ptr are read by which rank
          vector< pair<hsize_t,hsize_t> > ranges;
          mpi::rank_ranges(read_size, size, ranges);
        
          hsize_t start = ranges[rank].first + offset;
          hsize_t end   = start + ranges[rank].second;
          hsize_t block = end - start;
    
          if (block > 0)
            {
              /* Create property list for collective dataset operations. */
              hid_t rapl = H5Pcreate (H5P_DATASET_XFER);

              status = H5Pset_dxpl_mpio (rapl, H5FD_MPIO_COLLECTIVE);
              throw_assert(status >= 0,
                           "read_node_attribute: error in H5Pset_dxpl_mpio");
            
              string index_path = path + "/" + NODE_INDEX;
              // read index
              index.resize(block);

              status = read<NODE_IDX_T> (loc, index_path,
                                         start, block,
                                         NODE_IDX_H5_NATIVE_T,
                                         index, rapl);
            
              // read pointer and determine ranges
              string ptr_path = path + "/" + ATTR_PTR;
              ptr.resize(block+1);
              status = read<ATTR_PTR_T> (loc, ptr_path,
                                         start, block+1,
                                         ATTR_PTR_H5_NATIVE_T,
                                         ptr, rapl);
            
              hsize_t value_start = ptr[0];
              hsize_t value_block = ptr.back()-value_start;
            
              // read values
              string value_path = path + "/" + ATTR_VAL;
              hid_t dset = H5Dopen(loc, value_path.c_str(), H5P_DEFAULT);
              throw_assert(dset >= 0, "error in H5Dopen");
              hid_t ftype = H5Dget_type(dset);
              throw_assert(ftype >= 0, "error in H5Dget_type");
              hid_t ntype = H5Tget_native_type(ftype, H5T_DIR_ASCEND);
              throw_assert(H5Dclose(dset)   >= 0, "error in H5Dclose");
              throw_assert(H5Tclose(ftype)  >= 0, "error in H5Tclose");
            
              values.resize(value_block);
              status = read<T> (loc, value_path,
                                value_start, value_block,
                                ntype,
                                values, rapl);
            
              throw_assert(H5Tclose(ntype)  >= 0, "error in H5Tclose");
              status = H5Pclose(rapl);
              throw_assert(status == 0, "error in H5Pclose");
            
              for (size_t i=0; i<block+1; i++)
                {
                  ptr[i] -= value_start;
                }
            }
        }

      return status;
    }

    
    template <typename T>
    void append_node_attribute
    (
     MPI_Comm                        comm,
     const hid_t&                    loc,
     const std::string&              path,
     const std::vector<NODE_IDX_T>&  index,
     const std::vector<ATTR_PTR_T>&  attr_ptr,
     const std::vector<T>&           value
     )
    {
      int status;
      throw_assert(index.size() == attr_ptr.size()-1, "invalid index size");
      std::vector<ATTR_PTR_T>  local_attr_ptr;

      hid_t file = H5Iget_file_id(loc);
      throw_assert(file >= 0, "H5Iget_file_id");
    
      int ssize, srank; size_t size, rank;
      throw_assert(MPI_Comm_size(comm, &ssize) == MPI_SUCCESS, "error in MPI_Comm_size");
      throw_assert(MPI_Comm_rank(comm, &srank) == MPI_SUCCESS, "error in MPI_Comm_rank");
      throw_assert(ssize > 0, "invalid MPI comm size");
      throw_assert(srank >= 0, "invalid MPI comm rank");
      size = ssize;
      rank = srank;

      // Determine the total size of index
      uint64_t local_index_size=index.size();
      std::vector<uint64_t> index_size_vector;
      index_size_vector.resize(size);
      status = MPI_Allgather(&local_index_size, 1, MPI_UINT64_T, &index_size_vector[0], 1, MPI_UINT64_T, comm);
      throw_assert(status == MPI_SUCCESS, "append_node_attribute: error in MPI_Allgather");
#ifdef NEUROH5_DEBUG
      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "append_node_attribute: error in MPI_Barrier");
#endif

      // Determine the total number of ptrs, add 1 to ptr of last rank
      uint64_t local_ptr_size=attr_ptr.size()-1;
      if (rank == size-1)
        {
          local_ptr_size=local_ptr_size+1;
        }
    
      std::vector<uint64_t> ptr_size_vector;
      ptr_size_vector.resize(size);
      status = MPI_Allgather(&local_ptr_size, 1, MPI_UINT64_T, &ptr_size_vector[0], 1, MPI_UINT64_T, comm);
      throw_assert(status == MPI_SUCCESS, "append_node_attribute: error in MPI_Allgather");
#ifdef NEUROH5_DEBUG
      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "append_node_attribute: error in MPI_Barrier");
#endif
    
      hsize_t local_value_size = value.size();
      std::vector<uint64_t> value_size_vector;
      value_size_vector.resize(size);
      status = MPI_Allgather(&local_value_size, 1, MPI_UINT64_T, &value_size_vector[0], 1, MPI_UINT64_T, comm);
      throw_assert(status == MPI_SUCCESS, "append_node_attribute: error in MPI_Allgather");
#ifdef NEUROH5_DEBUG
      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "append_node_attribute: error in MPI_Barrier");
#endif

      T dummy;
      hid_t ftype = infer_datatype(dummy);
      throw_assert(ftype >= 0, "error in infer_datatype");
      hid_t mtype = H5Tget_native_type(ftype, H5T_DIR_ASCEND);
      throw_assert(mtype >= 0, "error in H5Tget_native_type");

      // create datasets
      hsize_t ptr_size=0, index_size=0, value_size=0;
      size_node_attributes(comm, loc, path, ptr_size, index_size, value_size);

      // Determine starting positions
      hsize_t ptr_start=0, index_start=0, value_start=0;
      if (ptr_size>0)
        {
          ptr_start=ptr_size-1;
        }
      index_start=index_size; value_start=value_size;

      hsize_t local_value_start=value_start, local_index_start=index_start, local_ptr_start=ptr_start;
      // calculate the starting positions of this rank
      for (size_t i=0; i<rank; i++)
        {
          local_value_start = local_value_start + value_size_vector[i];
          local_index_start = local_index_start + index_size_vector[i];
          local_ptr_start = local_ptr_start + ptr_size_vector[i];
        }

      // calculate the new sizes of the datasets
      hsize_t global_value_size=0, global_index_size=0, global_ptr_size=0;
      for (size_t i=0; i<size; i++)
        {
          global_value_size  = global_value_size + value_size_vector[i];
          global_index_size  = global_index_size + index_size_vector[i];
          global_ptr_size  = global_ptr_size + ptr_size_vector[i];
        }

      // add local value offset to attr_ptr
      local_attr_ptr.resize(attr_ptr.size());
      for (size_t i=0; i<local_attr_ptr.size(); i++)
        {
          local_attr_ptr[i] = attr_ptr[i] + local_value_start;
        }

      // write to datasets
      /* Create property list for collective dataset write. */
      hid_t wapl = H5Pcreate (H5P_DATASET_XFER);
      if (size > 1)
        {
          status = H5Pset_dxpl_mpio (wapl, H5FD_MPIO_COLLECTIVE);
          throw_assert(status >= 0,
                       "append_node_attribute: error in H5Pset_dxpl_mpio");
        }
      
      // TODO:
      // if option index_mode is:
      // 1) create: create the node index in /Populations, otherwise validate with index in /Populations
      // 2) link: link to node index already in this attribute namespace
      
      status = write<NODE_IDX_T> (file, path + "/" + NODE_INDEX,
                                  global_index_size+index_start, local_index_start, local_index_size,
                                  NODE_IDX_H5_NATIVE_T,
                                  index, wapl);
    
      status = write<ATTR_PTR_T> (file, path + "/" + ATTR_PTR,
                                  global_ptr_size+ptr_start, local_ptr_start, local_ptr_size,
                                  ATTR_PTR_H5_NATIVE_T,
                                  local_attr_ptr, wapl);
    
      status = write<T> (file, path + "/" + ATTR_VAL,
                         global_value_size+value_start, local_value_start, local_value_size,
                         mtype, value, wapl);

      // clean house
      status = H5Fclose (file);
      throw_assert(status >= 0, "error in H5Fclose");
    
      throw_assert(H5Tclose(mtype) >= 0, "error in H5Tclose");
      status = H5Pclose(wapl);
      throw_assert(status == 0, "error in H5Pclose");
#ifdef NEUROH5_DEBUG
      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "append_node_attribute: error in MPI_Barrier");
#endif
    }


  
    template <typename T>
    void write_node_attribute
    (
     MPI_Comm                        comm,
     const hid_t&                    loc,
     const std::string&              path,
     const std::vector<NODE_IDX_T>&  index,
     const std::vector<ATTR_PTR_T>&  attr_ptr,
     const std::vector<T>&           value
     )
    {
      int status;
      throw_assert(index.size() == attr_ptr.size()-1, "invalid index");
      std::vector<ATTR_PTR_T>  local_attr_ptr;

      hid_t file = H5Iget_file_id(loc);
      throw_assert(file >= 0, "H5Iget_file_id");

      int ssize, srank; size_t size, rank;
      throw_assert(MPI_Comm_size(comm, &ssize) == MPI_SUCCESS, "error in MPI_Comm_size");
      throw_assert(MPI_Comm_rank(comm, &srank) == MPI_SUCCESS, "error in MPI_Comm_rank");
      throw_assert(ssize > 0, "invalid MPI comm size");
      throw_assert(srank >= 0, "invalid MPI comm rank");
      size = ssize;
      rank = srank;

      // Determine the total number of index
      hsize_t local_index_size=index.size();
      std::vector<uint64_t> index_size_vector;
      index_size_vector.resize(size);
      status = MPI_Allgather(&local_index_size, 1, MPI_UINT64_T, &index_size_vector[0], 1, MPI_UINT64_T, comm);
      throw_assert(status == MPI_SUCCESS, "write_node_attribute: error in MPI_Allgather");
#ifdef NEUROH5_DEBUG
      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "write_node_attribute: error in MPI_Barrier");
#endif

      // Determine the total number of ptrs, add 1 to ptr of last rank
      hsize_t local_ptr_size=attr_ptr.size()-1;
      if (rank == size-1)
        {
          local_ptr_size=local_ptr_size+1;
        }
    
      std::vector<uint64_t> ptr_size_vector;
      ptr_size_vector.resize(size);
      status = MPI_Allgather(&local_ptr_size, 1, MPI_UINT64_T, &ptr_size_vector[0], 1, MPI_UINT64_T, comm);
      throw_assert(status == MPI_SUCCESS, "write_node_attribute: error in MPI_Allgather");
#ifdef NEUROH5_DEBUG
      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "write_node_attribute: error in MPI_Barrier");
#endif
    
      hsize_t local_value_size = value.size();
      std::vector<uint64_t> value_size_vector;
      value_size_vector.resize(size);
      status = MPI_Allgather(&local_value_size, 1, MPI_UINT64_T, &value_size_vector[0], 1, MPI_UINT64_T, comm);
      throw_assert(status == MPI_SUCCESS, "write_node_attribute: error in MPI_Allgather");
#ifdef NEUROH5_DEBUG
      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "write_node_attribute: error in MPI_Barrier");
#endif

      hsize_t local_value_start=0, local_index_start=0, local_ptr_start=0;
      // calculate the starting positions of this rank
      for (size_t i=0; i<rank; i++)
        {
          local_value_start = local_value_start + value_size_vector[i];
          local_index_start = local_index_start + index_size_vector[i];
          local_ptr_start = local_ptr_start + ptr_size_vector[i];
        }
      // calculate the new sizes of the datasets
      hsize_t global_value_size=0, global_index_size=0, global_ptr_size=0;
      for (size_t i=0; i<size; i++)
        {
          global_value_size  = global_value_size + value_size_vector[i];
          global_index_size  = global_index_size + index_size_vector[i];
          global_ptr_size  = global_ptr_size + ptr_size_vector[i];
        }

      // add local value offset to attr_ptr
      local_attr_ptr.resize(attr_ptr.size());
      for (size_t i=0; i<local_attr_ptr.size(); i++)
        {
          local_attr_ptr[i] = attr_ptr[i] + local_value_start;
        }

      /* Create property list for collective dataset write. */
      hid_t wapl = H5Pcreate (H5P_DATASET_XFER);
      if (size > 1)
        {
          status = H5Pset_dxpl_mpio (wapl, H5FD_MPIO_COLLECTIVE);
          throw_assert(status >= 0,
                       "write_node_attribute: error in H5Pset_dxpl_mpio");
        }

      T dummy;
      hid_t ftype = infer_datatype(dummy);
      throw_assert(ftype >= 0, "error in infer_datatype");
      hid_t mtype = H5Tget_native_type(ftype, H5T_DIR_ASCEND);
      throw_assert(mtype >= 0, "error in H5Tget_native_type");
    
      if (global_value_size > 0)
        {
          // write to datasets
        
          status = write<NODE_IDX_T> (loc, path + "/" + NODE_INDEX,
                                      global_index_size, local_index_start, local_index_size,
                                      NODE_IDX_H5_NATIVE_T,
                                      index, wapl);
        
          status = write<ATTR_PTR_T> (loc, path + "/" + ATTR_PTR,
                                      global_ptr_size, local_ptr_start, local_ptr_size,
                                      ATTR_PTR_H5_NATIVE_T,
                                      local_attr_ptr, wapl);
        
          status = write<T> (loc, path + "/" + ATTR_VAL,
                             global_value_size, local_value_start, local_value_size,
                             mtype, value, wapl);
        }

      throw_assert(H5Tclose(mtype)  >= 0, "error in H5Tclose");
      status = H5Pclose(wapl);
      throw_assert(status == 0, "error in H5Pclose");

      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "write_node_attribute: error in MPI_Barrier");
    }

  }
  
}


#endif
