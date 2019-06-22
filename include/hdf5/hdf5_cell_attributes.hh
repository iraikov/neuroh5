#ifndef HDF5_CELL_ATTRIBUTES
#define HDF5_CELL_ATTRIBUTES

#include <hdf5.h>

#include <vector>

#include "neuroh5_types.hh"
#include "rank_range.hh"
#include "dataset_num_elements.hh"
#include "exists_dataset.hh"
#include "exists_group.hh"
#include "read_template.hh"
#include "write_template.hh"
#include "mpe_seq.hh"
#include "throw_assert.hh"


namespace neuroh5
{
  namespace hdf5
  {

    void size_cell_attributes
    (
     MPI_Comm                   comm,
     hid_t                      loc,
     const std::string&         path,
     const CellPtr&             ptr_type,
     hsize_t&                   ptr_size,
     hsize_t&                   index_size,
     hsize_t&                   value_size
     );
    
    template <typename T>
    herr_t read_cell_attribute
    (
     MPI_Comm                  comm,
     const hid_t&              loc,
     const std::string&        path,
     const CELL_IDX_T          pop_start,
     std::vector<CELL_IDX_T>&  index,
     std::vector<ATTR_PTR_T>&  ptr,
     std::vector<T> &          values,
     size_t offset = 0,
     size_t numitems = 0
     )
    {
      herr_t status = 0;

      int size, rank;
      assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS);
      assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS);

      status = exists_group (loc, path.c_str());
      assert(status > 0);
      
      hsize_t dset_size = dataset_num_elements (loc, path + "/" + CELL_INDEX);
      size_t read_size = 0;
      if (numitems > 0) 
        {
          if (offset < dset_size)
            {
              read_size = min((hsize_t)numitems, dset_size-offset);
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
    
          /* Create property list for collective dataset operations. */
          hid_t rapl = H5Pcreate (H5P_DATASET_XFER);
          status = H5Pset_dxpl_mpio (rapl, H5FD_MPIO_COLLECTIVE);
          
          string index_path = path + "/" + CELL_INDEX;
          string ptr_path = path + "/" + ATTR_PTR;
          string value_path = path + "/" + ATTR_VAL;
          
          // read index
          index.resize(block);

          status = read<NODE_IDX_T> (loc, index_path, start, block,
                                     NODE_IDX_H5_NATIVE_T, index, rapl);
          for (size_t i=0; i<index.size(); i++)
            {
              index[i] += pop_start;
            }
              
          // read pointer and determine ranges
          status = exists_dataset (loc, ptr_path.c_str());
          if (status > 0)
            {
              if (block > 0)
                {
                  ptr.resize(block+1);
                }
              status = read<ATTR_PTR_T> (loc, ptr_path, start, block > 0 ? block+1 : 0,
                                         ATTR_PTR_H5_NATIVE_T, ptr, rapl);
              assert (status >= 0);
            }
              
          hsize_t value_start, value_block;
          if (ptr.size() > 0)
            {
              value_start = ptr[0];
              value_block = ptr.back()-value_start;
            }
          else
            {
              value_start = 0;
              value_block = block > 0 ? dataset_num_elements (loc, value_path) : 0;
            }
          
          // read values
          hid_t dset = H5Dopen(loc, value_path.c_str(), H5P_DEFAULT);
          assert(dset >= 0);
          hid_t ftype = H5Dget_type(dset);
          assert(ftype >= 0);
          hid_t ntype = H5Tget_native_type(ftype, H5T_DIR_ASCEND);
          assert(H5Dclose(dset)   >= 0);
          assert(H5Tclose(ftype)  >= 0);
          
          values.resize(value_block);
          status = read<T> (loc, value_path, value_start, value_block,
                            ntype, values, rapl);
          
          assert(H5Tclose(ntype)  >= 0);
          status = H5Pclose(rapl);
          assert(status == 0);
          
          if (ptr.size() > 0)
            {
              for (size_t i=0; i<block+1; i++)
                {
                  ptr[i] -= value_start;
                }
            }
        }
      return status;
    }

    
    template <typename T>
    herr_t read_cell_attribute_selection
    (
     MPI_Comm                  comm,
     const hid_t&              loc,
     const std::string&        path,
     const CELL_IDX_T          pop_start,
     const std::vector<CELL_IDX_T>&  selection,
     std::vector<CELL_IDX_T> & selection_index,
     std::vector<ATTR_PTR_T> & selection_ptr,
     std::vector<T> &          values
     )
    {
      herr_t status = 0;
      std::vector<ATTR_PTR_T> ptr;
      std::vector<CELL_IDX_T> index;

      int size, rank;
      assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS);
      assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS);

      status = exists_group (loc, path.c_str());
      assert(status > 0);
      
      hsize_t dset_size = dataset_num_elements (loc, path + "/" + CELL_INDEX);
      vector< pair<hsize_t,hsize_t> > ranges;
      
      if (dset_size > 0)
        {
          /* Create property list for collective dataset operations. */
          hid_t rapl = H5Pcreate (H5P_DATASET_XFER);
          status = H5Pset_dxpl_mpio (rapl, H5FD_MPIO_COLLECTIVE);

          
          string index_path = path + "/" + CELL_INDEX;
          string ptr_path = path + "/" + ATTR_PTR;
          string value_path = path + "/" + ATTR_VAL;

          // read index
          index.resize(dset_size);

          status = read<NODE_IDX_T> (loc, index_path, 0, dset_size,
                                     NODE_IDX_H5_NATIVE_T, index, rapl);

          // read pointer and determine ranges
          status = exists_dataset (loc, ptr_path);
          if (status > 0)
            {
              ptr.resize(dset_size+1);
              status = read<ATTR_PTR_T> (loc, ptr_path, 0, dset_size+1,
                                         ATTR_PTR_H5_NATIVE_T, ptr, rapl);
              assert (status >= 0);
            }

          ATTR_PTR_T selection_ptr_pos = 0;
          if (ptr.size() > 0)
            {
              for (const CELL_IDX_T& s : selection) 
                {
                  if (s < pop_start) continue;
                  auto it = std::find(index.begin(), index.end(), s-pop_start);
                  if (it == index.end()) continue;
                  
                  throw_assert(it != index.end(),
                               "read_cell_attribute_selection: unable to find attribute "
                               << path << " for gid " << s);

                  ptrdiff_t pos = it - index.begin();

                  hsize_t value_start=ptr[pos];
                  hsize_t value_block=ptr[pos+1]-value_start;

                  ranges.push_back(make_pair(value_start, value_block));
                  selection_ptr.push_back(selection_ptr_pos);
                  selection_ptr_pos += value_block;
                  selection_index.push_back(s);
                }
              selection_ptr.push_back(selection_ptr_pos);
            }
          
          // read values
          hid_t dset = H5Dopen(loc, value_path.c_str(), H5P_DEFAULT);
          assert(dset >= 0);
          hid_t ftype = H5Dget_type(dset);
          assert(ftype >= 0);
          hid_t ntype = H5Tget_native_type(ftype, H5T_DIR_ASCEND);
          assert(H5Dclose(dset)   >= 0);
          assert(H5Tclose(ftype)  >= 0);
              
          values.resize(selection_ptr_pos);
          status = read_selection<T> (loc, value_path, ntype, ranges, values, rapl);
          
          status = H5Pclose(rapl);
          assert(status == 0);
          
          assert(H5Tclose(ntype)  >= 0);
        }
      
      return status;
    }

    
    template <typename T>
    void append_cell_attribute
    (
     const hid_t&                    loc,
     const std::string&              path,
     const std::vector<CELL_IDX_T>&  index,
     const std::vector<ATTR_PTR_T>&  attr_ptr,
     const std::vector<T>&           value,
     const data::optional_hid        data_type,
     const CellIndex                 index_type,
     const CellPtr                   ptr_type
     )
    {
      int status;
      throw_assert(index.size() == attr_ptr.size()-1,
                   "append_cell_attribute: mismatch of sizes of cell index and attribute pointer");
      std::vector<ATTR_PTR_T>  local_attr_ptr;

      hid_t file = H5Iget_file_id(loc);
      throw_assert(file >= 0,
                   "append_cell_attribute: invalid file handle");

      // get the I/O communicator
      MPI_Comm comm;
      MPI_Info info;
      hid_t fapl = H5Fget_access_plist(file);
      throw_assert(H5Pget_fapl_mpio(fapl, &comm, &info) >= 0,
                   "append_cell_attribute: unable to obtain MPI I/O communicator");
    
      int ssize, srank; size_t size, rank;
      assert(MPI_Comm_size(comm, &ssize) == MPI_SUCCESS);
      assert(MPI_Comm_rank(comm, &srank) == MPI_SUCCESS);
      assert(ssize > 0);
      assert(srank >= 0);
      size = ssize;
      rank = srank;

      // Determine the total size of index
      hsize_t local_index_size=index.size();
      std::vector<uint64_t> index_size_vector;
      index_size_vector.resize(size);
      status = MPI_Allgather(&local_index_size, 1, MPI_UINT64_T, &index_size_vector[0], 1, MPI_UINT64_T, comm);
      throw_assert(status == MPI_SUCCESS,
                   "append_cell_attribute: error in MPI_Allgather");

      // Determine the total number of ptrs, add 1 to ptr of last rank
      hsize_t local_ptr_size;

      if (attr_ptr.size() > 0)
        {
          local_ptr_size = attr_ptr.size()-1;
        }
      else
        {
          local_ptr_size = 0;
        }
        
      
      if (rank == size-1)
        {
          local_ptr_size=local_ptr_size+1;
        }
    
      std::vector<uint64_t> ptr_size_vector;
      ptr_size_vector.resize(size);
      status = MPI_Allgather(&local_ptr_size, 1, MPI_UINT64_T, &ptr_size_vector[0], 1, MPI_UINT64_T, comm);
      throw_assert(status == MPI_SUCCESS, "append_cell_attribute: error in MPI_Allgather");
    
      hsize_t local_value_size = value.size();
      std::vector<uint64_t> value_size_vector;
      value_size_vector.resize(size);
      status = MPI_Allgather(&local_value_size, 1, MPI_UINT64_T, &value_size_vector[0], 1, MPI_UINT64_T, comm);
      throw_assert(status == MPI_SUCCESS, "append_cell_attribute: error in MPI_Allgather");
      
      T dummy;
      hid_t ftype;
      if (data_type.has_value())
        ftype = data_type.value();
      else
        ftype = infer_datatype(dummy);
      throw_assert(ftype >= 0, "append_cell_attribute: unable to infer HDF5 datatype");

      hid_t mtype = H5Tget_native_type(ftype, H5T_DIR_ASCEND);
      throw_assert(mtype >= 0, "append_cell_attribute: unable to obtain native HDF5 datatype");

      // create datasets
      hsize_t ptr_size=0, index_size=0, value_size=0;
      size_cell_attributes(comm, loc, path, ptr_type, ptr_size, index_size, value_size);

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
      hsize_t global_value_size=value_start, global_index_size=index_start, global_ptr_size=ptr_start;
      for (size_t i=0; i<size; i++)
        {
          global_value_size  = global_value_size + value_size_vector[i];
          global_index_size  = global_index_size + index_size_vector[i];
          global_ptr_size    = global_ptr_size + ptr_size_vector[i];
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
        }


      switch (index_type)
        {
        case IndexOwner:
          // TODO: save to prefix and link to index in path
          status = write<CELL_IDX_T> (file, path + "/" + CELL_INDEX,
                                      global_index_size, local_index_start, local_index_size,
                                      CELL_IDX_H5_NATIVE_T,
                                      index, wapl);
          break;
        case IndexShared:
          break;
        case IndexNone:
          break;
        }
    
      switch (ptr_type.type)
        {
        case PtrOwner:
          {
            // TODO: save to prefix and link to index in path
            std::string ptr_name;
            if (ptr_type.shared_ptr_name.has_value())
              {
                ptr_name = ptr_type.shared_ptr_name.value();
              }
            else
              {
                ptr_name = ATTR_PTR;
              }
            status = write<ATTR_PTR_T> (file, path + "/" + ptr_name,
                                        global_ptr_size, local_ptr_start, local_ptr_size,
                                        ATTR_PTR_H5_NATIVE_T,
                                        local_attr_ptr, wapl);
          }
          break;
        case PtrShared:
          break;
        case PtrNone:
          break;
        }

      if (global_value_size > 0)
        {
          status = write<T> (file, path + "/" + ATTR_VAL,
                             global_value_size, local_value_start, local_value_size,
                             mtype, value, wapl);
        }

      status = H5Fclose (file);
      throw_assert(status >= 0, "append_cell_attribute: unable to close HDF5 file");
    
      assert(H5Tclose(mtype) >= 0);
      status = H5Pclose(wapl);
      assert(status == 0);
      status = H5Pclose(fapl);
      assert(status == 0);

      status = MPI_Comm_free(&comm);
      throw_assert(status == MPI_SUCCESS,
                   "append_cell_attribute: error in MPI_Comm_free");
      if (info != MPI_INFO_NULL)
        {
          status = MPI_Info_free(&info);
          throw_assert(status == MPI_SUCCESS,
                       "append_cell_attribute: error in MPI_Info_free");
        
        }
    }


  
    template <typename T>
    void write_cell_attribute
    (
     const hid_t&                    loc,
     const std::string&              path,
     const std::vector<CELL_IDX_T>&  index,
     const std::vector<ATTR_PTR_T>&  attr_ptr,
     const std::vector<T>&           value,
     const CellIndex                 index_type,
     const CellPtr                   ptr_type
     )
    {
      int status;
      assert(index.size() == attr_ptr.size()-1);
      std::vector<ATTR_PTR_T>  local_attr_ptr;
    
      hid_t file = H5Iget_file_id(loc);
      assert(file >= 0);

      // get the I/O communicator
      MPI_Comm comm;
      MPI_Info info;
      hid_t fapl = H5Fget_access_plist(file);
      assert(H5Pget_fapl_mpio(fapl, &comm, &info) >= 0);
      
      int ssize, srank; size_t size, rank;
      assert(MPI_Comm_size(comm, &ssize) == MPI_SUCCESS);
      assert(MPI_Comm_rank(comm, &srank) == MPI_SUCCESS);
      assert(ssize > 0);
      assert(srank >= 0);
      size = ssize;
      rank = srank;

      // Determine the total number of index
      hsize_t local_index_size=index.size();
      std::vector<uint64_t> index_size_vector;
      index_size_vector.resize(size);
      status = MPI_Allgather(&local_index_size, 1, MPI_UINT64_T, &index_size_vector[0], 1, MPI_UINT64_T, comm);
      assert(status == MPI_SUCCESS);

      // Determine the total number of ptrs, add 1 to ptr of last rank
      hsize_t local_ptr_size=attr_ptr.size()-1;
      if (rank == size-1)
        {
          local_ptr_size=local_ptr_size+1;
        }
    
      std::vector<uint64_t> ptr_size_vector;
      ptr_size_vector.resize(size);
      status = MPI_Allgather(&local_ptr_size, 1, MPI_UINT64_T, &ptr_size_vector[0], 1, MPI_UINT64_T, comm);
      assert(status == MPI_SUCCESS);
    
      hsize_t local_value_size = value.size();
      std::vector<uint64_t> value_size_vector;
      value_size_vector.resize(size);
      status = MPI_Allgather(&local_value_size, 1, MPI_UINT64_T, &value_size_vector[0], 1, MPI_UINT64_T, comm);
      assert(status == MPI_SUCCESS);

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

      T dummy;
      hid_t ftype = infer_datatype(dummy);
      assert(ftype >= 0);
      hid_t mtype = H5Tget_native_type(ftype, H5T_DIR_ASCEND);
      assert(mtype >= 0);

      /* Create property list for collective dataset write. */
      hid_t wapl = H5Pcreate (H5P_DATASET_XFER);
      if (size > 1)
        {
          status = H5Pset_dxpl_mpio (wapl, H5FD_MPIO_COLLECTIVE);
        }
    
      if (global_value_size > 0)
        {
          // write to datasets

          switch (index_type)
            {
            case IndexOwner:
              status = write<CELL_IDX_T> (file, path + "/" + CELL_INDEX,
                                          global_index_size, local_index_start, local_index_size,
                                          CELL_IDX_H5_NATIVE_T,
                                          index, wapl);
              break;
            case IndexShared:
              // TODO: validate index
              break;
            case IndexNone:
              break;
            }

          switch (ptr_type.type)
            {
            case PtrOwner:
              status = write<ATTR_PTR_T> (file, path + "/" + ATTR_PTR,
                                          global_ptr_size, local_ptr_start, local_ptr_size,
                                          ATTR_PTR_H5_NATIVE_T,
                                          local_attr_ptr, wapl);
            case PtrShared:
              // TODO: validate ptr
              break;

            case PtrNone:
              break;
            }
        
          status = write<T> (file, path + "/" + ATTR_VAL,
                             global_value_size, local_value_start, local_value_size,
                             mtype, value, wapl);
        }

      status = H5Fclose (file);
      assert(status >= 0);

      assert(H5Tclose(mtype)  >= 0);
      status = H5Pclose(wapl);
      assert(status == 0);

      status = H5Pclose(fapl);
      assert(status == 0);

      status = MPI_Comm_free(&comm);
      assert(status == MPI_SUCCESS);
      if (info != MPI_INFO_NULL)
        {
          status = MPI_Info_free(&info);
          assert(status == MPI_SUCCESS);
        
        }
    }

  }
  
}

#endif
