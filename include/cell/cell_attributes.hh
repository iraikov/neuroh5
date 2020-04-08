#ifndef CELL_ATTRIBUTES_HH
#define CELL_ATTRIBUTES_HH

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>
#include <set>

#include <hdf5.h>
#include <mpi.h>

#include "mpe_seq.hh"
#include "neuroh5_types.hh"
#include "infer_datatype.hh"
#include "infer_mpi_datatype.hh"
#include "path_names.hh"
#include "hdf5_cell_attributes.hh"
#include "exists_dataset.hh"
#include "file_access.hh"
#include "attr_map.hh"
#include "compact_optional.hh"
#include "optional_value.hh"
#include "range_sample.hh"
#include "throw_assert.hh"

namespace neuroh5
{
  namespace cell
  {
        
  
    void create_cell_attribute_datasets
    (
     const hid_t&     file,
     const string&    attr_namespace,
     const string&    pop_name,
     const string&    attr_name,
     const hid_t&     ftype,
     const CellIndex& index_type,
     const CellPtr&   ptr_type,
     const size_t     chunk_size,
     const size_t     value_chunk_size
     );

    
    herr_t get_cell_attribute_name_spaces
    (
     const string&       file_name,
     const string&       pop_name,
     vector<string>&     out_name_spaces
     );

    
    herr_t get_cell_attributes
    (
     const string&                 file_name,
     const string&                 name_space,
     const string&                 pop_name,
     vector< pair<string,AttrKind> >& out_attributes
     );

    
    void read_cell_attributes
    (
     MPI_Comm         comm,
     const string&    file_name,
     const string&    name_space,
     const set<string>& attr_mask,
     const string&    pop_name,
     const CELL_IDX_T& pop_start,
     data::NamedAttrMap&    attr_values,
     size_t offset = 0,
     size_t numitems = 0
     );

    void read_cell_attribute_selection
    (
     MPI_Comm         comm,
     const string& file_name,
     const string& name_space,
     const set<string>& attr_mask,
     const string& pop_name,
     const CELL_IDX_T& pop_start,
     const std::vector<CELL_IDX_T>&  selection,
     data::NamedAttrMap& attr_values
     );
    
    int scatter_read_cell_attributes
    (
     MPI_Comm                      all_comm,
     const string                 &file_name,
     const int                     io_size,
     const string                 &attr_name_space,
     const set<string>            &attr_mask,
     // A vector that maps nodes to compute ranks
     const map<CELL_IDX_T, rank_t> &node_rank_map,
     const string                 &pop_name,
     const CELL_IDX_T             &pop_start,
     data::NamedAttrMap           &attr_map,
     // if positive, these arguments specify offset and number of entries to read
     // from the entries available to the current rank
     size_t offset   = 0,
     size_t numitems = 0
     );

    
    void bcast_cell_attributes
    (
     MPI_Comm               comm,
     const int              root,
     const std::string&           file_name,
     const std::string&           name_space,
     const std::set<std::string>& attr_mask,
     const std::string&           pop_name,
     const CELL_IDX_T&       pop_start,
     data::NamedAttrMap&    attr_values,
     size_t offset = 0,
     size_t numitems = 0
     );

  
    template <typename T>
    void append_cell_attribute
    (
     MPI_Comm                              comm,
     const std::string&                    file_name,
     const std::string&                    attr_namespace,
     const std::string&                    pop_name,
     const CELL_IDX_T&                     pop_start,
     const std::string&                    attr_name,
     const std::vector<CELL_IDX_T>&        index,
     const std::vector<ATTR_PTR_T>         attr_ptr,
     const std::vector<T>&                 values,
     const data::optional_hid              data_type,
     const CellIndex                       index_type,
     const CellPtr                         ptr_type,
     const size_t chunk_size = 4000,
     const size_t value_chunk_size = 4000,
     const size_t cache_size = 1*1024*1024
     )
    {
      int status;
      throw_assert(index.size() == attr_ptr.size()-1,
                   "append_cell_attribute: mismatch between sizes of cell index and attribute pointer");
      std::vector<ATTR_PTR_T>  local_attr_ptr;
    
      int size, rank;
      throw_assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS,
                   "append_cell_attribute: unable to obtain MPI communicator size");
      throw_assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS,
                   "append_cell_attribute: unable to obtain MPI communicator rank");

      hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
      throw_assert(H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL) >= 0,
                   "append_cell_attribute: HDF5 mpio error");
      /* Cache parameters: */
      int nelemts;    /* Dummy parameter in API, no longer used */ 
      size_t nslots;  /* Number of slots in the 
                         hash table */ 
      size_t nbytes; /* Size of chunk cache in bytes */ 
      double w0;      /* Chunk preemption policy */ 
      /* Retrieve default cache parameters */ 
      throw_assert(H5Pget_cache(fapl, &nelemts, &nslots, &nbytes, &w0) >=0,
                   "error in H5Pget_cache");
      /* Set cache size and instruct the cache to discard the fully read chunk */ 
      nbytes = cache_size; w0 = 1.;
      throw_assert(H5Pset_cache(fapl, nelemts, nslots, nbytes, w0)>= 0,
                   "error in H5Pset_cache");
      
      hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDWR, fapl);
      throw_assert(file >= 0,
                   "append_cell_attribute: HDF5 file open error");

      T dummy;
      hid_t ftype;
      if (data_type.has_value())
        {
          ftype = data_type.value();
        }
      else
        ftype = infer_datatype(dummy);
      throw_assert(ftype >= 0,
                   "append_cell_attribute: unable to infer HDF5 data type");

      string attr_prefix = hdf5::cell_attribute_prefix(attr_namespace, pop_name);
      string attr_path = hdf5::cell_attribute_path(attr_namespace, pop_name, attr_name);

      if (!(hdf5::exists_dataset (file, attr_path) > 0))
        {
          create_cell_attribute_datasets(file, attr_namespace, pop_name, attr_name,
                                         ftype, index_type, ptr_type,
                                         chunk_size, value_chunk_size
                                         );
        }

      vector<CELL_IDX_T> rindex;

      for (const CELL_IDX_T& gid: index)
        {
          rindex.push_back(gid - pop_start);
        }
      
      hdf5::append_cell_attribute<T>(file, attr_path, rindex, attr_ptr, values,
                                     data_type, index_type, ptr_type);
    
      status = H5Fclose(file);
      throw_assert(status == 0, "append_cell_attribute: unable to close HDF5 file");
      status = H5Pclose(fapl);
      throw_assert(status == 0, "append_cell_attribute: unable to close HDF5 file properties list");
    }


    template <typename T>
    void append_cell_attribute_map
    (
     MPI_Comm                        comm,
     const std::string&              file_name,
     const std::string&              attr_namespace,
     const std::string&              pop_name,
     const CELL_IDX_T&               pop_start,
     const std::string&              attr_name,
     const std::map<CELL_IDX_T, vector<T>>& value_map,
     const size_t io_size,
     const data::optional_hid        data_type,
     const CellIndex                 index_type = IndexOwner,
     const CellPtr                   ptr_type = CellPtr(PtrOwner),
     const size_t chunk_size = 4000,
     const size_t value_chunk_size = 4000,
     const size_t cache_size = 1*1024*1024
     )
    {
      vector<CELL_IDX_T>  index_vector;
      vector<T>  value_vector;
    
      int ssize, srank; size_t size, rank; size_t io_size_value=0;
      throw_assert(MPI_Comm_size(comm, &ssize) == MPI_SUCCESS, "error in MPI_Comm_size");
      throw_assert(MPI_Comm_rank(comm, &srank) == MPI_SUCCESS, "error in MPI_Comm_rank");
      throw_assert(ssize > 0, "invalid MPI comm size");
      throw_assert(srank >= 0, "invalid MPI rank");
      rank = srank;
      size = ssize;
      
      if (size < io_size)
	{
	  io_size_value = size;
	}
      else
	{
	  io_size_value = io_size;
	}

      set<size_t> io_rank_set;
      data::range_sample(size, io_size_value, io_rank_set);
      bool is_io_rank = false;
      if (io_rank_set.find(rank) != io_rank_set.end())
        is_io_rank = true;

      
      vector< pair<hsize_t,hsize_t> > ranges;
      mpi::rank_ranges(size, io_size_value, ranges);

      // Determine I/O ranks to which to send the values
      vector <size_t> io_dests(size); 
      for (size_t r=0; r<size; r++)
        {
          for (size_t i=ranges.size()-1; i>=0; i--)
            {
              if (ranges[i].first <= r)
                {
                  io_dests[r] = *std::next(io_rank_set.begin(), i);
                  break;
                }
            }
        }

      // Determine number of values for each rank
      vector<uint32_t> sendbuf_num_values(size, value_map.size());
      vector<uint32_t> recvbuf_num_values(size);
      throw_assert(MPI_Allgather(&sendbuf_num_values[0], 1, MPI_UINT32_T,
                                 &recvbuf_num_values[0], 1, MPI_UINT32_T, comm)
                   == MPI_SUCCESS, "append_cell_attribute_map: error in MPI_Allgather");
      sendbuf_num_values.clear();

      // Determine local value size and offset
      uint32_t local_value_size=0;
      for (auto const& element : value_map)
        {
          const vector<T> &v = element.second;
          local_value_size += v.size();
        }
      vector<uint32_t> sendbuf_size_values(size, local_value_size);
      vector<uint32_t> recvbuf_size_values(size);
      throw_assert(MPI_Allgather(&sendbuf_size_values[0], 1, MPI_UINT32_T,
                                 &recvbuf_size_values[0], 1, MPI_UINT32_T, comm)
                   == MPI_SUCCESS, "append_cell_attribute_map: error in MPI_Allgather");
      sendbuf_size_values.clear();
    
      // Create gid, attr_ptr, value arrays
      vector<ATTR_PTR_T>  attr_ptr;
      ATTR_PTR_T value_offset = 0;
      hsize_t global_value_size = 0;
      for (size_t i=0; i<size; i++)
        {
          if (i<rank)
            {
              value_offset += recvbuf_size_values[i];
            }
          global_value_size += recvbuf_size_values[i];
        }

      for (auto const& element : value_map)
        {
          const CELL_IDX_T gid = element.first;
          const vector<T> &v = element.second;
          index_vector.push_back(gid);
          attr_ptr.push_back(value_offset);
          value_vector.insert(value_vector.end(),v.begin(),v.end());
          value_offset = value_offset + v.size();
        }
      //attr_ptr.push_back(value_offset);

      vector<int> value_sendcounts(size, 0), value_sdispls(size, 0), value_recvcounts(size, 0), value_rdispls(size, 0);
      value_sendcounts[io_dests[rank]] = local_value_size;

    
      // Compute size of values to receive for all I/O rank
      if (is_io_rank)
        {
          for (size_t s=0; s<size; s++)
            {
              if (io_dests[s] == rank)
                {
                  value_recvcounts[s] += recvbuf_size_values[s];
                }
            }
        }

      // Each ALL_COMM rank accumulates the vector sizes and allocates
      // a receive buffer, recvcounts, and rdispls
      size_t value_recvbuf_size = value_recvcounts[0];
      for (int p = 1; p < ssize; ++p)
        {
          value_rdispls[p] = value_rdispls[p-1] + value_recvcounts[p-1];
          value_recvbuf_size += value_recvcounts[p];
        }
      //assert(recvbuf_size > 0);
      vector<T> value_recvbuf(value_recvbuf_size);

      T dummy;
      MPI_Datatype mpi_type = infer_mpi_datatype(dummy);
      throw_assert(MPI_Alltoallv(&value_vector[0], &value_sendcounts[0], &value_sdispls[0], mpi_type,
                                 &value_recvbuf[0], &value_recvcounts[0], &value_rdispls[0], mpi_type,
                                 comm) == MPI_SUCCESS,
             "append_cell_attribute_map: error in MPI_Alltoallv");
      value_vector.clear();
    
      vector<int> ptr_sendcounts(size, 0), ptr_sdispls(size, 0), ptr_recvcounts(size, 0), ptr_rdispls(size, 0);
      ptr_sendcounts[io_dests[rank]] = attr_ptr.size();

      // Compute number of values to receive for all I/O rank
      if (is_io_rank)
        {
          for (size_t s=0; s<size; s++)
            {
              if (io_dests[s] == rank)
                {
                  ptr_recvcounts[s] += recvbuf_num_values[s];
                }
            }
        }
    
      // Each ALL_COMM rank accumulates the vector sizes and allocates
      // a receive buffer, recvcounts, and rdispls
      size_t ptr_recvbuf_size = ptr_recvcounts[0];
      for (int p = 1; p < ssize; ++p)
        {
          ptr_rdispls[p] = ptr_rdispls[p-1] + ptr_recvcounts[p-1];
          ptr_recvbuf_size += ptr_recvcounts[p];
        }
      //assert(recvbuf_size > 0);
      vector<ATTR_PTR_T> attr_ptr_recvbuf(ptr_recvbuf_size);
    
      // Each ALL_COMM rank participates in the MPI_Alltoallv
      throw_assert(MPI_Alltoallv(&attr_ptr[0], &ptr_sendcounts[0], &ptr_sdispls[0], MPI_ATTR_PTR_T,
                                 &attr_ptr_recvbuf[0], &ptr_recvcounts[0], &ptr_rdispls[0], MPI_ATTR_PTR_T,
                                 comm) == MPI_SUCCESS,
                   "append_cell_attribute_map: error in MPI_Alltoallv");
      if ((is_io_rank) && (attr_ptr_recvbuf.size() > 0))
        {
          attr_ptr_recvbuf.push_back(attr_ptr_recvbuf[0]+value_recvbuf.size());
        }
      else
        {
          attr_ptr_recvbuf.push_back(value_recvbuf.size());
        }
      if (attr_ptr_recvbuf.size() > 0)
        {
          ATTR_PTR_T attr_ptr_recvbuf_base = attr_ptr_recvbuf[0];
          for (size_t i=0; i<attr_ptr_recvbuf.size(); i++)
            {
              attr_ptr_recvbuf[i] -= attr_ptr_recvbuf_base;
            }
        }
    
      attr_ptr.clear();
      ptr_sendcounts.clear();
      ptr_sdispls.clear();
      ptr_recvcounts.clear();
      ptr_rdispls.clear();

      vector<int> idx_sendcounts(size, 0), idx_sdispls(size, 0), idx_recvcounts(size, 0), idx_rdispls(size, 0);
      idx_sendcounts[io_dests[rank]] = index_vector.size();

      // Compute number of values to receive for all I/O rank
      if (is_io_rank)
        {
          for (size_t s=0; s<size; s++)
            {
              if (io_dests[s] == rank)
                {
                  idx_recvcounts[s] += recvbuf_num_values[s];
                }
            }
        }
    
      // Each ALL_COMM rank accumulates the vector sizes and allocates
      // a receive buffer, recvcounts, and rdispls
      size_t idx_recvbuf_size = idx_recvcounts[0];
      for (int p = 1; p < ssize; ++p)
        {
          idx_rdispls[p] = idx_rdispls[p-1] + idx_recvcounts[p-1];
          idx_recvbuf_size += idx_recvcounts[p];
        }

      vector<CELL_IDX_T> gid_recvbuf(idx_recvbuf_size);
    
      throw_assert(MPI_Alltoallv(&index_vector[0], &idx_sendcounts[0], &idx_sdispls[0], MPI_CELL_IDX_T,
                                 &gid_recvbuf[0], &idx_recvcounts[0], &idx_rdispls[0], MPI_CELL_IDX_T,
                                 comm) == MPI_SUCCESS,
                   "append_cell_attribute_map: error in MPI_Alltoallv");
      index_vector.clear();
      idx_sendcounts.clear();
      idx_sdispls.clear();
      idx_recvcounts.clear();
      idx_rdispls.clear();

    
      // MPI Communicator for I/O ranks
      MPI_Comm io_comm;
      // MPI group color value used for I/O ranks
      int io_color = 1, color;
    
      // Am I an I/O rank?
      if (is_io_rank)
        {
          color = io_color;
        }
      else
        {
          color = 0;
        }
      MPI_Comm_split(comm,color,rank,&io_comm);
      MPI_Comm_set_errhandler(io_comm, MPI_ERRORS_RETURN);

      if (is_io_rank)
        {
          append_cell_attribute<T>(io_comm, file_name,
                                   attr_namespace, pop_name, pop_start, attr_name,
                                   gid_recvbuf, attr_ptr_recvbuf, value_recvbuf,
                                   data_type, index_type, ptr_type, 
                                   chunk_size, value_chunk_size, cache_size);
        }
      
      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "append_cell_attribute_map: error in MPI_Barrier");
      throw_assert(MPI_Barrier(io_comm) == MPI_SUCCESS,
                   "append_cell_attribute_map: error in MPI_Barrier");
      throw_assert(MPI_Comm_free(&io_comm) == MPI_SUCCESS,
                   "append_cell_attribute_map: error in MPI_Comm_free");
    }


  

    template <typename T>
    void write_cell_attribute
    (
     MPI_Comm                        comm,
     const std::string&              file_name,
     const std::string&              attr_namespace,
     const std::string&              pop_name,
     const CELL_IDX_T&               pop_start,
     const std::string&              attr_name,
     const std::vector<CELL_IDX_T>&  index,
     const std::vector<ATTR_PTR_T>&  attr_ptr,
     const std::vector<T>&           value,
     const data::optional_hid        data_type,
     const CellIndex                 index_type,
     const CellPtr                   ptr_type,
     const size_t chunk_size = 4000,
     const size_t value_chunk_size = 4000,
     const size_t cache_size = 1*1024*1024
     )
    {
      int status;
      throw_assert(index.size() == attr_ptr.size()-1, "invalid index");
      std::vector<ATTR_PTR_T>  local_attr_ptr;
    
    
      // get a file handle and retrieve the MPI info
      hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
      throw_assert(H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL) >= 0,
		   "write_cell_attribute: unable to set fapl mpio property");

      /* Cache parameters: */
      int nelemts;    /* Dummy parameter in API, no longer used */ 
      size_t nslots;  /* Number of slots in the 
                         hash table */ 
      size_t nbytes; /* Size of chunk cache in bytes */ 
      double w0;      /* Chunk preemption policy */ 
      /* Retrieve default cache parameters */ 
      throw_assert(H5Pget_cache(fapl, &nelemts, &nslots, &nbytes, &w0) >=0,
                   "error in H5Pget_cache");
      /* Set cache size and instruct the cache to discard the fully read chunk */ 
      nbytes = cache_size; w0 = 1.;
      throw_assert(H5Pset_cache(fapl, nelemts, nslots, nbytes, w0)>= 0,
                   "error in H5Pset_cache");

      hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDWR, fapl);
      throw_assert(file >= 0, "write_cell_attribute: unable to open file");

      T dummy;
      hid_t ftype;
      if (data_type.has_value())
        ftype = data_type.value();
      else
        ftype = infer_datatype(dummy);
      throw_assert(ftype >= 0, "error in infer_datatype");

      string attr_path = hdf5::cell_attribute_path(attr_namespace, pop_name, attr_name);

      create_cell_attribute_datasets(file, attr_namespace, pop_name, attr_name,
                                     ftype, index_type, ptr_type,
                                     chunk_size, value_chunk_size);

      vector<CELL_IDX_T> rindex;

      for (const CELL_IDX_T& gid: index)
        {
          rindex.push_back(gid - pop_start);
        }
    
      hdf5::write_cell_attribute<T> (file, attr_path,
                                     rindex, attr_ptr, value,
                                     index_type, ptr_type);


      
      status = H5Fclose(file);
      throw_assert(status == 0, "error in H5Fclose");
      status = H5Pclose(fapl);
      throw_assert(status == 0, "error in H5Pclose");
    }


    template <typename T>
    void write_cell_attribute_map
    (
     MPI_Comm                        comm,
     const std::string&              file_name,
     const std::string&              attr_namespace,
     const std::string&              pop_name,
     const CELL_IDX_T&               pop_start,
     const std::string&              attr_name,
     const std::map<CELL_IDX_T, vector<T>>& value_map,
     const size_t io_size,
     const data::optional_hid        data_type,
     const CellIndex                 index_type = IndexOwner,
     const CellPtr                   ptr_type = CellPtr(PtrOwner),
     const size_t chunk_size = 4000,
     const size_t value_chunk_size = 4000,
     const size_t cache_size = 1*1024*1024
     )
    {
      vector<CELL_IDX_T>  index_vector;
      vector<T>  value_vector;
    
      int ssize, srank; size_t size, rank; size_t io_size_value=0;
      throw_assert(MPI_Comm_size(comm, &ssize) == MPI_SUCCESS, "error in MPI_Comm_size");
      throw_assert(MPI_Comm_rank(comm, &srank) == MPI_SUCCESS, "error in MPI_Comm_rank");
      throw_assert(ssize > 0, "invalid MPI comm size");
      throw_assert(srank >= 0, "invalid MPI comm rank");
      rank = srank;
      size = ssize;
      
      if (size < io_size)
	{
	  io_size_value = size;
	}
      else
	{
	  io_size_value = io_size;
	}

      set<size_t> io_rank_set;
      data::range_sample(size, io_size_value, io_rank_set);
      bool is_io_rank = false;
      if (io_rank_set.find(rank) != io_rank_set.end())
        is_io_rank = true;

      
      vector< pair<hsize_t,hsize_t> > ranges;
      mpi::rank_ranges(size, io_size_value, ranges);

      // Determine I/O ranks to which to send the values
      vector <size_t> io_dests(size); 
      for (size_t r=0; r<size; r++)
        {
          for (size_t i=ranges.size()-1; i>=0; i--)
            {
              if (ranges[i].first <= r)
                {
                  io_dests[r] = *std::next(io_rank_set.begin(), i);
                  break;
                }
            }
        }

      // Determine number of values for each rank
      vector<uint32_t> sendbuf_num_values(size, value_map.size());
      vector<uint32_t> recvbuf_num_values(size);
      throw_assert(MPI_Allgather(&sendbuf_num_values[0], 1, MPI_UINT32_T,
                                 &recvbuf_num_values[0], 1, MPI_UINT32_T, comm)
                   == MPI_SUCCESS, "write_cell_attribute_map: error in MPI_Allgather");
      sendbuf_num_values.clear();

      // Determine local value size and offset
      uint32_t local_value_size=0;
      for (auto const& element : value_map)
        {
          const vector<T> &v = element.second;
          local_value_size += v.size();
        }
      vector<uint32_t> sendbuf_size_values(size, local_value_size);
      vector<uint32_t> recvbuf_size_values(size);
      throw_assert(MPI_Allgather(&sendbuf_size_values[0], 1, MPI_UINT32_T,
                                 &recvbuf_size_values[0], 1, MPI_UINT32_T, comm)
                   == MPI_SUCCESS, "write_cell_attribute_map: error in MPI_Allgather");
      sendbuf_size_values.clear();
    
      // Create gid, attr_ptr, value arrays
      vector<ATTR_PTR_T>  attr_ptr;
      ATTR_PTR_T value_offset = 0;
      hsize_t global_value_size = 0;
      for (size_t i=0; i<size; i++)
        {
          if (i<rank)
            {
              value_offset += recvbuf_size_values[i];
            }
          global_value_size += recvbuf_size_values[i];
        }

      for (auto const& element : value_map)
        {
          const CELL_IDX_T gid = element.first;
          const vector<T> &v = element.second;
          index_vector.push_back(gid);
          attr_ptr.push_back(value_offset);
          value_vector.insert(value_vector.end(),v.begin(),v.end());
          value_offset = value_offset + v.size();
        }
      //attr_ptr.push_back(value_offset);

      vector<int> value_sendcounts(size, 0), value_sdispls(size, 0), value_recvcounts(size, 0), value_rdispls(size, 0);
      value_sendcounts[io_dests[rank]] = local_value_size;

    
      // Compute size of values to receive for all I/O rank
      if (is_io_rank)
        {
          for (size_t s=0; s<size; s++)
            {
              if (io_dests[s] == rank)
                {
                  value_recvcounts[s] += recvbuf_size_values[s];
                }
            }
        }

      // Each ALL_COMM rank accumulates the vector sizes and allocates
      // a receive buffer, recvcounts, and rdispls
      size_t value_recvbuf_size = value_recvcounts[0];
      for (int p = 1; p < ssize; ++p)
        {
          value_rdispls[p] = value_rdispls[p-1] + value_recvcounts[p-1];
          value_recvbuf_size += value_recvcounts[p];
        }
      //assert(recvbuf_size > 0);
      vector<T> value_recvbuf(value_recvbuf_size);

      T dummy;
      MPI_Datatype mpi_type = infer_mpi_datatype(dummy);
      throw_assert(MPI_Alltoallv(&value_vector[0], &value_sendcounts[0], &value_sdispls[0], mpi_type,
                                 &value_recvbuf[0], &value_recvcounts[0], &value_rdispls[0], mpi_type,
                                 comm) == MPI_SUCCESS,
             "write_cell_attribute_map: error in MPI_Alltoallv");
      value_vector.clear();
    
      vector<int> ptr_sendcounts(size, 0), ptr_sdispls(size, 0), ptr_recvcounts(size, 0), ptr_rdispls(size, 0);
      ptr_sendcounts[io_dests[rank]] = attr_ptr.size();

      // Compute number of values to receive for all I/O rank
      if (is_io_rank)
        {
          for (size_t s=0; s<size; s++)
            {
              if (io_dests[s] == rank)
                {
                  ptr_recvcounts[s] += recvbuf_num_values[s];
                }
            }
        }
    
      // Each ALL_COMM rank accumulates the vector sizes and allocates
      // a receive buffer, recvcounts, and rdispls
      size_t ptr_recvbuf_size = ptr_recvcounts[0];
      for (int p = 1; p < ssize; ++p)
        {
          ptr_rdispls[p] = ptr_rdispls[p-1] + ptr_recvcounts[p-1];
          ptr_recvbuf_size += ptr_recvcounts[p];
        }
      //assert(recvbuf_size > 0);
      vector<ATTR_PTR_T> attr_ptr_recvbuf(ptr_recvbuf_size);
    
      // Each ALL_COMM rank participates in the MPI_Alltoallv
      throw_assert(MPI_Alltoallv(&attr_ptr[0], &ptr_sendcounts[0], &ptr_sdispls[0], MPI_ATTR_PTR_T,
                                 &attr_ptr_recvbuf[0], &ptr_recvcounts[0], &ptr_rdispls[0], MPI_ATTR_PTR_T,
                                 comm) == MPI_SUCCESS,
                   "write_cell_attribute_map: error in MPI_Alltoallv");
      if ((is_io_rank) && (attr_ptr_recvbuf.size() > 0))
        {
          attr_ptr_recvbuf.push_back(attr_ptr_recvbuf[0]+value_recvbuf.size());
        }
      else
        {
          attr_ptr_recvbuf.push_back(value_recvbuf.size());
        }
      if (attr_ptr_recvbuf.size() > 0)
        {
          ATTR_PTR_T attr_ptr_recvbuf_base = attr_ptr_recvbuf[0];
          for (size_t i=0; i<attr_ptr_recvbuf.size(); i++)
            {
              attr_ptr_recvbuf[i] -= attr_ptr_recvbuf_base;
            }
        }
    
      attr_ptr.clear();
      ptr_sendcounts.clear();
      ptr_sdispls.clear();
      ptr_recvcounts.clear();
      ptr_rdispls.clear();

      vector<int> idx_sendcounts(size, 0), idx_sdispls(size, 0), idx_recvcounts(size, 0), idx_rdispls(size, 0);
      idx_sendcounts[io_dests[rank]] = index_vector.size();

      // Compute number of values to receive for all I/O rank
      if (is_io_rank)
        {
          for (size_t s=0; s<size; s++)
            {
              if (io_dests[s] == rank)
                {
                  idx_recvcounts[s] += recvbuf_num_values[s];
                }
            }
        }
    
      // Each ALL_COMM rank accumulates the vector sizes and allocates
      // a receive buffer, recvcounts, and rdispls
      size_t idx_recvbuf_size = idx_recvcounts[0];
      for (int p = 1; p < ssize; ++p)
        {
          idx_rdispls[p] = idx_rdispls[p-1] + idx_recvcounts[p-1];
          idx_recvbuf_size += idx_recvcounts[p];
        }

      vector<CELL_IDX_T> gid_recvbuf(idx_recvbuf_size);
    
      throw_assert(MPI_Alltoallv(&index_vector[0], &idx_sendcounts[0], &idx_sdispls[0], MPI_CELL_IDX_T,
                                 &gid_recvbuf[0], &idx_recvcounts[0], &idx_rdispls[0], MPI_CELL_IDX_T,
                                 comm) == MPI_SUCCESS,
                   "write_cell_attribute_map: error in MPI_Alltoallv");
      index_vector.clear();
      idx_sendcounts.clear();
      idx_sdispls.clear();
      idx_recvcounts.clear();
      idx_rdispls.clear();

    
      // MPI Communicator for I/O ranks
      MPI_Comm io_comm;
      // MPI group color value used for I/O ranks
      int io_color = 1, color;
    
      // Am I an I/O rank?
      if (is_io_rank)
        {
          color = io_color;
        }
      else
        {
          color = 0;
        }
      MPI_Comm_split(comm,color,rank,&io_comm);
      MPI_Comm_set_errhandler(io_comm, MPI_ERRORS_RETURN);

      if (is_io_rank)
        {
          write_cell_attribute<T>(io_comm, file_name,
                                  attr_namespace, pop_name, pop_start, attr_name,
                                  gid_recvbuf, attr_ptr_recvbuf, value_recvbuf,
                                  data_type, index_type, ptr_type, 
                                  chunk_size, value_chunk_size, cache_size);
        }
      
      throw_assert(MPI_Barrier(io_comm) == MPI_SUCCESS,
                   "write_cell_attribute_map: error in MPI_Barrier");
      throw_assert(MPI_Comm_free(&io_comm) == MPI_SUCCESS,
                   "write_cell_attribute_map: error in MPI_Comm_free");
      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "write_cell_attribute_map: error in MPI_Barrier");
    }


  }
  
}



#endif
