#ifndef HDF5_CELL_ATTRIBUTES
#define HDF5_CELL_ATTRIBUTES

#include <hdf5.h>

#include <vector>
#include <algorithm>

#include "neuroh5_types.hh"
#include "rank_range.hh"
#include "dataset_num_elements.hh"
#include "exists_dataset.hh"
#include "exists_group.hh"
#include "file_access.hh"
#include "read_template.hh"
#include "write_template.hh"
#include "sort_permutation.hh"
#include "throw_assert.hh"
#include "mpe_seq.hh"
#include "debug.hh"


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

    
    herr_t read_cell_index
    (
     const hid_t&              loc,
     const std::string&        path,
     std::vector<CELL_IDX_T> & index,
     bool collective = false
     );


    herr_t read_cell_index_ptr
    (
     const hid_t&              loc,
     const std::string&        path,
     std::vector<CELL_IDX_T> & index,
     std::vector<ATTR_PTR_T> & ptr,
     bool collective = false
     );


    
    template <typename T>
    herr_t read_cell_attribute
    (
     MPI_Comm                  comm,
     const hid_t&              loc,
     const std::string&        path,
     const CELL_IDX_T          pop_start,
     const std::vector<CELL_IDX_T>&  index,
     const std::vector<ATTR_PTR_T>&  ptr,
     std::vector<CELL_IDX_T>&  value_index,
     std::vector<ATTR_PTR_T> & value_ptr,
     std::vector<T> &          value,
     size_t offset = 0,
     size_t numitems = 0
     )
    {
      herr_t status = 0;

      int size, rank;
      throw_assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS, "error in MPI_Comm_size");
      throw_assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS, "error in MPI_Comm_rank");

      if (rank == 0) 
	{
	  status = exists_group (loc, path.c_str());
	  throw_assert(status > 0, "group " << path << " does not exist");
	}
      
      hsize_t dset_size = index.size();
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
          throw_assert(status >= 0,
                       "read_cell_attribute: error in H5Pset_dxpl_mpio");
          
          string value_path = path + "/" + ATTR_VAL;

          hsize_t value_start, value_block;
          if (ptr.size() > 0)
            {
              value_start = ptr[start];
              value_block = ptr[end]-value_start;
              value_ptr.resize(block+1, 0);
              for (size_t i=start, j=0; i<end+1; i++, j++)
                {
                  value_ptr[j] = ptr[i] - value_start;
                }
            }
          else
            {
              value_start = 0;
              value_block = block > 0 ? dataset_num_elements (loc, value_path) : 0;
            }
          value_index.resize(block, 0);
          for (size_t i=start, j=0; i<end; i++, j++)
            {
              value_index[j] = index[i];
            }

          // read values
          hid_t dset = H5Dopen(loc, value_path.c_str(), H5P_DEFAULT);
          throw_assert(dset >= 0, "error in H5Dopen");
          hid_t ftype = H5Dget_type(dset);
          throw_assert(ftype >= 0, "error in H5Dget_type");
          hid_t ntype = H5Tget_native_type(ftype, H5T_DIR_ASCEND);
          throw_assert(H5Dclose(dset)   >= 0, "error  in H5Dclose");
          throw_assert(H5Tclose(ftype)  >= 0, "error  in H5Tclose");
          
          value.resize(value_block, 0);
          status = read<T> (loc, value_path, value_start, value_block,
                            ntype, value, rapl);
          
          throw_assert(H5Tclose(ntype)  >= 0, "error in H5Tclose");
          status = H5Pclose(rapl);
          throw_assert(status == 0, "error in H5Pclose");
          
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
     const std::vector<CELL_IDX_T>&  index,
     const std::vector<ATTR_PTR_T>&  ptr,
     std::vector<CELL_IDX_T> & selection_index,
     std::vector<ATTR_PTR_T> & selection_ptr,
     std::vector<T> &          values
     )
    {
      herr_t status = 0;

      int size, rank;
      throw_assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS, "error in MPI_Comm_size");
      throw_assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS, "error in MPI_Comm_size");

      status = exists_group (loc, path.c_str());
      throw_assert(status > 0, "group " << path << " does not exist");
      
      hsize_t dset_size = index.size();
      vector< pair<hsize_t,hsize_t> > ranges;

      auto compare_idx = [](const CELL_IDX_T& a, const CELL_IDX_T& b) { return (a < b); };
      
      if (dset_size > 0)
        {

          string value_path = path + "/" + ATTR_VAL;

          ATTR_PTR_T selection_ptr_pos = 0;
          if (ptr.size() > 0)
            {
	      vector<size_t> p = data::sort_permutation(index, compare_idx);
	      std::vector<CELL_IDX_T> sorted_index = data::apply_permutation(index, p);

              for (const CELL_IDX_T& s : selection) 
                {
                  if (s < pop_start) continue;
		  auto rp = std::equal_range(sorted_index.begin(), sorted_index.end(), s);
		  for ( auto it = rp.first; it != rp.second; ++it )
                    {
                      ptrdiff_t spos = it - sorted_index.begin();
		      ptrdiff_t pos = p[spos];
                      hsize_t value_start=ptr[pos];
                      hsize_t value_block=ptr[pos+1]-value_start;

                      ranges.push_back(make_pair(value_start, value_block));
                      selection_index.push_back(s);
                    }
                }
          

	      auto compare_range_idx = [](const std::pair<hsize_t, hsize_t>& a, const std::pair<hsize_t, hsize_t>& b) 
		{ return (a.first < b.first); };
	  
	      vector<size_t> range_sort_p = data::sort_permutation(ranges, compare_range_idx);
	      
	      data::apply_permutation_in_place(selection_index, range_sort_p);
	      data::apply_permutation_in_place(ranges, range_sort_p);

	      for (const auto& range: ranges)
		{
		  hsize_t value_start=range.first;
		  hsize_t value_block=range.second;

		  selection_ptr.push_back(selection_ptr_pos);
		  selection_ptr_pos += value_block;
		}
	      selection_ptr.push_back(selection_ptr_pos);
            }

          hid_t dset = H5Dopen(loc, value_path.c_str(), H5P_DEFAULT);
          throw_assert(dset >= 0, "error in H5Dopen");
          hid_t ftype = H5Dget_type(dset);
          throw_assert(ftype >= 0, "error in H5Dget_type");
          hid_t ntype = H5Tget_native_type(ftype, H5T_DIR_ASCEND);
          throw_assert(H5Tclose(ftype)  >= 0, "error in H5Tclose");
          throw_assert(H5Dclose(dset)   >= 0, "error in H5Dclose");

          /* Create property list for collective dataset operations. */
          hid_t rapl = H5Pcreate (H5P_DATASET_XFER);

          status = H5Pset_dxpl_mpio (rapl, H5FD_MPIO_COLLECTIVE);
          throw_assert(status >= 0,
                       "read_cell_attribute_selection: error in H5Pset_dxpl_mpio");
              
          values.resize(selection_ptr_pos, 0);

          status = read_selection<T> (loc, value_path, ntype, ranges, values, rapl);
          throw_assert(H5Pclose(rapl)   >= 0, "error in H5Pclose");
          throw_assert(H5Tclose(ntype)  >= 0, "error in H5Tclose");
        }

      return status;
    }

   
    template <typename T>
    void append_cell_attribute
    (
     MPI_Comm                        comm,
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

      int ssize, srank; size_t size, rank;
      throw_assert(MPI_Comm_size(comm, &ssize) == MPI_SUCCESS, "error in MPI_Comm_size");
      throw_assert(MPI_Comm_rank(comm, &srank) == MPI_SUCCESS, "error in MPI_Comm_rank");
      throw_assert(ssize > 0, "invalid MPI comm size");
      throw_assert(srank >= 0, "invalid MPI comm rank");
      size = ssize;
      rank = srank;

      // Determine the total size of index
      hsize_t local_index_size=index.size();
      std::vector<uint64_t> index_size_vector(size, 0);
      status = MPI_Allgather(&local_index_size, 1, MPI_UINT64_T, &index_size_vector[0], 1, MPI_UINT64_T, comm);
      throw_assert(status == MPI_SUCCESS,
                   "append_cell_attribute: error in MPI_Allgather");
#ifdef NEUROH5_DEBUG
      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "append_cell_attribute: error in MPI_Barrier");
#endif

      // Determine last rank that has data
      size_t last_rank = size-1;

      for (size_t r=last_rank; r >= 0; r--)
	{
	  if (index_size_vector[r] > 0)
	    {
	      last_rank = r;
	      break;
	    }
	}
      
      // Determine the total number of ptrs, add 1 to ptr of last rank
      hsize_t local_ptr_size=0;

      if (attr_ptr.size() > 0)
        {
          if (rank == last_rank)
            {
              local_ptr_size = attr_ptr.size();
            }
          else
            {
              local_ptr_size = attr_ptr.size()-1;
            }
        }
    
      std::vector<uint64_t> ptr_size_vector(size, 0);
      status = MPI_Allgather(&local_ptr_size, 1, MPI_UINT64_T, &ptr_size_vector[0], 1, MPI_UINT64_T, comm);
      throw_assert(status == MPI_SUCCESS, "append_cell_attribute: error in MPI_Allgather");
#ifdef NEUROH5_DEBUG
      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "append_cell_attribute: error in MPI_Barrier");
#endif
    
      hsize_t local_value_size = value.size();
      std::vector<uint64_t> value_size_vector(size, 0);
      status = MPI_Allgather(&local_value_size, 1, MPI_UINT64_T, &value_size_vector[0], 1, MPI_UINT64_T, comm);
      throw_assert(status == MPI_SUCCESS, "append_cell_attribute: error in MPI_Allgather");
#ifdef NEUROH5_DEBUG
      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "append_cell_attribute: error in MPI_Barrier");
#endif
      
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
      local_attr_ptr.resize(local_ptr_size, 0);
      for (size_t i=0; i<local_ptr_size; i++)
        {
          local_attr_ptr[i] = attr_ptr[i] + local_value_start;
	  throw_assert(local_attr_ptr[i] <= global_value_size,
		       "append_cell_attribute: path " << path << 
		       ": attribute pointer value " << local_attr_ptr[i] <<
		       " exceeds global value size " << global_value_size);
        }

      // write to datasets
      /* Create property list for collective dataset write. */
      hid_t wapl = H5Pcreate (H5P_DATASET_XFER);

      status = H5Pset_dxpl_mpio (wapl, H5FD_MPIO_COLLECTIVE);
      throw_assert(status >= 0,
		   "append_cell_attribute: error in H5Pset_dxpl_mpio");

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
    
      throw_assert(H5Tclose(mtype) >= 0, "error in H5Tclose");
      status = H5Pclose(wapl);
      throw_assert(status == 0, "error in H5Pclose");

      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "append_cell_attribute: error in MPI_Barrier");
    }


  
    template <typename T>
    void write_cell_attribute
    (
     MPI_Comm                        comm,
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
      throw_assert(index.size() == attr_ptr.size()-1, "invalid index size");
      std::vector<ATTR_PTR_T>  local_attr_ptr;
    
      hid_t file = H5Iget_file_id(loc);
      throw_assert(file >= 0, "error in H5Iget_file_id");

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
      index_size_vector.resize(size, 0);
      status = MPI_Allgather(&local_index_size, 1, MPI_UINT64_T, &index_size_vector[0], 1, MPI_UINT64_T, comm);
      throw_assert(status == MPI_SUCCESS, "write_cell_attribute: error in MPI_Allgather");
#ifdef NEUROH5_DEBUG
      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "write_cell_attribute: error in MPI_Barrier");
#endif

      // Determine last rank that has data
      size_t last_rank = size-1;

      for (size_t r=last_rank; r >= 0; r--)
	{
	  if (index_size_vector[r] > 0)
	    {
	      last_rank = r;
	      break;
	    }
	}
      
      // Determine the total number of ptrs, add 1 to ptr of last rank
      hsize_t local_ptr_size=0;

      if (attr_ptr.size() > 0)
        {
          if (rank == last_rank)
            {
              local_ptr_size = attr_ptr.size();
            }
          else
            {
              local_ptr_size = attr_ptr.size()-1;
            }
        }
    
      std::vector<uint64_t> ptr_size_vector;
      ptr_size_vector.resize(size, 0);
      status = MPI_Allgather(&local_ptr_size, 1, MPI_UINT64_T, &ptr_size_vector[0], 1, MPI_UINT64_T, comm);
      throw_assert(status == MPI_SUCCESS, "write_cell_attribute; error in MPI_Allgather");
#ifdef NEUROH5_DEBUG
      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "write_cell_attribute: error in MPI_Barrier");
#endif
    
      hsize_t local_value_size = value.size();
      std::vector<uint64_t> value_size_vector;
      value_size_vector.resize(size, 0);
      status = MPI_Allgather(&local_value_size, 1, MPI_UINT64_T, &value_size_vector[0], 1, MPI_UINT64_T, comm);
      throw_assert(status == MPI_SUCCESS, "write_cell_attribute: error in MPI_Allgather");
#ifdef NEUROH5_DEBUG
      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "write_cell_attribute: error in MPI_Barrier");
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
      local_attr_ptr.resize(attr_ptr.size(), 0);
      for (size_t i=0; i<local_attr_ptr.size(); i++)
        {
          local_attr_ptr[i] = attr_ptr[i] + local_value_start;
	  throw_assert(local_attr_ptr[i] <= global_value_size,
		       "write_cell_attribute: path " << path << 
		       ": attribute pointer value " << local_attr_ptr[i] <<
		       " exceeds global value size " << global_value_size);

        }

      T dummy;
      hid_t ftype = infer_datatype(dummy);
      throw_assert(ftype >= 0, "error in infer_datatype");
      hid_t mtype = H5Tget_native_type(ftype, H5T_DIR_ASCEND);
      throw_assert(mtype >= 0, "error in H5Tget_native_type");

      /* Create property list for collective dataset write. */
      hid_t wapl = H5Pcreate (H5P_DATASET_XFER);

      status = H5Pset_dxpl_mpio (wapl, H5FD_MPIO_COLLECTIVE);
      throw_assert(status >= 0,
		   "write_cell_attribute: error in H5Pset_dxpl_mpio");
    
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
      throw_assert(status >= 0, "error in H5Fclose");

      throw_assert(H5Tclose(mtype)  >= 0, "error in H5Tclose");
      status = H5Pclose(wapl);
      throw_assert(status == 0, "error in H5Pclose");

      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "write_cell_attribute: error in MPI_Barrier");
    }

  }
  
}

#endif
