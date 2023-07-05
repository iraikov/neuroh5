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
#include "alltoallv_template.hh"
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


    herr_t get_cell_attribute_index
    (
     const string&                 file_name,
     const string&                 name_space,
     const string&                 pop_name,
     const CELL_IDX_T&             pop_start,
     vector < tuple<string,AttrKind,vector <CELL_IDX_T> > >& out_attributes
     );


    herr_t get_cell_attribute_index_ptr
    (
     const string&                 file_name,
     const string&                 name_space,
     const string&                 pop_name,
     const CELL_IDX_T&             pop_start,
     vector< tuple<string,AttrKind,vector<CELL_IDX_T>,vector<ATTR_PTR_T> > >& out_attributes
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

    void scatter_read_cell_attribute_selection
    (
     MPI_Comm         comm,
     const string& file_name,
     const int io_size,
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
     const node_rank_map_t        &node_rank_map,
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

    
    void append_cell_attribute_maps (
                                     MPI_Comm                        comm,
                                     const std::string&              file_name,
                                     const std::string&              attr_namespace,
                                     const std::string&              pop_name,
                                     const CELL_IDX_T&               pop_start,
                                     const map<string, map<CELL_IDX_T, deque<uint32_t> >>& attr_values_uint32,
                                     const map<string, map<CELL_IDX_T, deque<int32_t> >> attr_values_int32,
                                     const map<string, map<CELL_IDX_T, deque<uint16_t> >>& attr_values_uint16,
                                     const map<string, map<CELL_IDX_T, deque<int16_t> >>& attr_values_int16,
                                     const map<string, map<CELL_IDX_T, deque<uint8_t> >>&  attr_values_uint8,
                                     const map<string, map<CELL_IDX_T, deque<int8_t> >>&  attr_values_int8,
                                     const map<string, map<CELL_IDX_T, deque<float> >>&  attr_values_float,
                                     const size_t io_size,
                                     const data::optional_hid        data_type,
                                     const CellIndex                 index_type = IndexOwner,
                                     const CellPtr                   ptr_type = CellPtr(PtrOwner),
                                     const size_t chunk_size = 4000,
                                     const size_t value_chunk_size = 4000,
                                     const size_t cache_size = 1*1024*1024
                                     );

  
    template <typename T>
    void append_cell_attribute
    (
     const hid_t&                          loc,
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
    

      string attr_prefix = hdf5::cell_attribute_prefix(attr_namespace, pop_name);
      string attr_path = hdf5::cell_attribute_path(attr_namespace, pop_name, attr_name);

      T dummy;
      hid_t ftype;
      if (data_type.has_value())
        ftype = data_type.value();
      else
        ftype = infer_datatype(dummy);

      hid_t file = H5Iget_file_id(loc);
      throw_assert(file >= 0,
                   "append_cell_attribute: invalid file handle");

      hid_t fapl;
      throw_assert((fapl = H5Fget_access_plist(file)) >= 0,
                   "append_cell_attribute_map: error in H5Fget_access_plist");

      MPI_Comm comm;
      MPI_Info comm_info;
      
      throw_assert(H5Pget_fapl_mpio(fapl, &comm, &comm_info) >= 0,
                   "append_cell_attribute: error in H5Pget_fapl_mpio");
                   
      int size, rank;
      throw_assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS,
                   "append_cell_attribute: unable to obtain MPI communicator size");
      throw_assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS,
                   "append_cell_attribute: unable to obtain MPI communicator rank");
      
      throw_assert(H5Pclose(fapl) == 0,
                   "append_cell_attribute: error in H5Pclose");

      
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
	  throw_assert(gid >= pop_start,
		       "append_cell_attribute: invalid gid");
          rindex.push_back(gid - pop_start);
        }
      
      hdf5::append_cell_attribute<T>(comm, file, attr_path, rindex, attr_ptr, values,
                                     data_type, index_type, ptr_type);
    
      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "append_cell_attribute: error in MPI_Barrier");

      status = H5Fclose(file);
      throw_assert(status == 0, "append_cell_attribute: unable to close HDF5 file");

    }

    
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

      string attr_prefix = hdf5::cell_attribute_prefix(attr_namespace, pop_name);
      string attr_path = hdf5::cell_attribute_path(attr_namespace, pop_name, attr_name);

      if (rank == 0)
        {
          hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
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
          
          
          if (!(hdf5::exists_dataset (file, attr_path) > 0))
            {
              create_cell_attribute_datasets(file, attr_namespace, pop_name, attr_name,
                                             ftype, index_type, ptr_type,
                                             chunk_size, value_chunk_size
                                             );
            }
          status = H5Fclose(file);
          throw_assert(status == 0, "append_cell_attribute: unable to close HDF5 file");
        }
      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "append_cell_attribute: error in MPI_Barrier");

      hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDWR, fapl);
      throw_assert(file >= 0,
                   "append_cell_attribute: HDF5 file open error");

      append_cell_attribute<T> (file, attr_namespace, pop_name, pop_start,
                                attr_name, index, attr_ptr, values,
                                data_type, index_type, ptr_type,
                                chunk_size, value_chunk_size, cache_size);
         
      status = H5Fclose(file);
      throw_assert(status == 0, "append_cell_attribute: unable to close HDF5 file");
      status = H5Pclose(fapl);
      throw_assert(status == 0, "append_cell_attribute: unable to close HDF5 file properties list");

      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "append_cell_attribute: error in MPI_Barrier");

    }


    template <typename T>
    void append_cell_attribute_map
    (
     MPI_Comm                        comm,
     const hid_t&                    loc,
     const std::string&              attr_namespace,
     const std::string&              pop_name,
     const CELL_IDX_T&               pop_start,
     const std::string&              attr_name,
     const std::map<CELL_IDX_T, deque<T>>& value_map,
     const data::optional_hid        data_type,
     const set<size_t>&              io_rank_set,
     const CellIndex                 index_type = IndexOwner,
     const CellPtr                   ptr_type = CellPtr(PtrOwner),
     const size_t chunk_size = 4000,
     const size_t value_chunk_size = 4000,
     const size_t cache_size = 1*1024*1024
     )
    {

      herr_t status;
      int ssize=0, srank=0; size_t size=0, rank=0; size_t io_size=io_rank_set.size();
      throw_assert(MPI_Comm_size(comm, &ssize) == MPI_SUCCESS, "error in MPI_Comm_size");
      throw_assert(MPI_Comm_rank(comm, &srank) == MPI_SUCCESS, "error in MPI_Comm_rank");
      throw_assert(ssize > 0, "invalid MPI comm size");
      throw_assert(srank >= 0, "invalid MPI rank");
      rank = srank;
      size = ssize;
      throw_assert(io_size <= size, "invalid io_size");

      bool is_io_rank = false;
      if (io_rank_set.find(rank) != io_rank_set.end())
        is_io_rank = true;

      hid_t fapl;
      hid_t file;
      MPI_Comm io_comm;
      MPI_Info io_comm_info;
      
      if (is_io_rank)
        {
          int io_size_value=0;
          
          file = H5Iget_file_id(loc);
          throw_assert(file >= 0,
                       "append_cell_attribute_map: invalid file handle");

          throw_assert((fapl = H5Fget_access_plist(file)) >= 0,
                       "append_cell_attribute_map: error in H5Fget_access_plist");
      
          throw_assert(H5Pget_fapl_mpio(fapl, &io_comm, &io_comm_info) >= 0,
                       "append_cell_attribute_map: error in H5Pget_fapl_mpio");
          
          throw_assert(MPI_Comm_size(io_comm, &io_size_value) == MPI_SUCCESS, "error in MPI_Comm_size");
          throw_assert(io_size_value == io_size, "io_size mismatch");
          
          throw_assert(H5Pclose(fapl) == 0,
                       "append_cell_attribute_map: error in H5Pclose");
        }
      
      vector< pair<hsize_t,hsize_t> > ranges;
      mpi::rank_ranges(size, io_size, ranges);

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
      vector<uint32_t> recvbuf_num_values(size, 0);
      {
        vector<uint32_t> sendbuf_num_values(size, value_map.size());
        throw_assert(MPI_Allgather(&sendbuf_num_values[0], 1, MPI_UINT32_T,
                                   &recvbuf_num_values[0], 1, MPI_UINT32_T, comm)
                     == MPI_SUCCESS, "append_cell_attribute_map: error in MPI_Allgather");
      }

      // Determine local value size and offset
      uint32_t local_value_size=0;
      vector<ATTR_PTR_T> local_attr_size_vector;
      vector<CELL_IDX_T> local_index_vector;
      for (auto const& element : value_map)
        {
          const CELL_IDX_T gid = element.first;
          const deque<T> &v = element.second;
          local_value_size += v.size();
	  local_index_vector.push_back(gid);
	  local_attr_size_vector.push_back(v.size());
        }
      
      vector<CELL_IDX_T> gid_recvbuf;
      {
        vector<int> idx_sendcounts(size, 0), idx_sdispls(size, 0), idx_recvcounts(size, 0), idx_rdispls(size, 0);
        idx_sendcounts[io_dests[rank]] = local_index_vector.size();
        
        throw_assert(mpi::alltoallv_vector<CELL_IDX_T>(comm, MPI_CELL_IDX_T,
                                                       idx_sendcounts, idx_sdispls, local_index_vector,
                                                       idx_recvcounts, idx_rdispls, gid_recvbuf) >= 0,
                     "append_cell_attribute_map: error in MPI_Alltoallv");
      }

      vector<ATTR_PTR_T> attr_ptr;
      {
        vector<ATTR_PTR_T> attr_size_recvbuf;
        vector<int> attr_size_sendcounts(size, 0), attr_size_sdispls(size, 0), attr_size_recvcounts(size, 0), attr_size_rdispls(size, 0);
        attr_size_sendcounts[io_dests[rank]] = local_attr_size_vector.size();

        throw_assert(mpi::alltoallv_vector<ATTR_PTR_T>(comm, MPI_ATTR_PTR_T,
                                                       attr_size_sendcounts, attr_size_sdispls, local_attr_size_vector,
                                                       attr_size_recvcounts, attr_size_rdispls, attr_size_recvbuf) >= 0,
                     "append_cell_attribute_map: error in MPI_Alltoallv");
        
        if ((is_io_rank) && (attr_size_recvbuf.size() > 0))
          {
            ATTR_PTR_T attr_ptr_offset = 0;
            for (size_t s=0; s<ssize; s++)
              {
                int count = attr_size_recvcounts[s];
                for (size_t i=attr_size_rdispls[s]; i<attr_size_rdispls[s]+count; i++)
                  {
                    ATTR_PTR_T this_attr_size = attr_size_recvbuf[i];
                    attr_ptr.push_back(attr_ptr_offset);
                    attr_ptr_offset += this_attr_size;
                  }
              }
            attr_ptr.push_back(attr_ptr_offset);
          }
      }

      
      vector<T> value_recvbuf;
      {
        vector<uint32_t> recvbuf_size_values(size, 0);
        vector<uint32_t> sendbuf_size_values(size, local_value_size);
        throw_assert(MPI_Allgather(&sendbuf_size_values[0], 1, MPI_UINT32_T,
                                   &recvbuf_size_values[0], 1, MPI_UINT32_T, comm)
                     == MPI_SUCCESS, "append_cell_attribute_map: error in MPI_Allgather");
        throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                       "append_cell_attribute_map: error in MPI_Barrier");
        
        vector<T>  local_value_vector;
        for (auto const& element : value_map)
          {
            const CELL_IDX_T gid = element.first;
            const deque<T> &v = element.second;
            local_value_vector.insert(local_value_vector.end(),v.begin(),v.end());
          }
        
        vector<int> value_sendcounts(size, 0), value_sdispls(size, 0), value_recvcounts(size, 0), value_rdispls(size, 0);
        value_sendcounts[io_dests[rank]] = local_value_size;
        

        T dummy;
        MPI_Datatype mpi_type = infer_mpi_datatype(dummy);
        throw_assert(mpi::alltoallv_vector<T>(comm, mpi_type,
                                              value_sendcounts, value_sdispls, local_value_vector,
                                              value_recvcounts, value_rdispls, value_recvbuf) >= 0,
                     "append_cell_attribute_map: error in MPI_Alltoallv");
      }
      
      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "append_cell_attribute_map: error in MPI_Barrier");
    
      if (is_io_rank)
        {
          append_cell_attribute<T>(file,
                                   attr_namespace, pop_name, pop_start, attr_name,
                                   gid_recvbuf, attr_ptr, value_recvbuf,
                                   data_type, index_type, ptr_type, 
                                   chunk_size, value_chunk_size, cache_size);
        }

      if (is_io_rank)
        {
          throw_assert(MPI_Barrier(io_comm) == MPI_SUCCESS,
                       "append_cell_attribute_map: error in MPI_Barrier");
          throw_assert(MPI_Comm_free(&io_comm) == MPI_SUCCESS,
                       "append_cell_attribute_map: error in MPI_Comm_free");
          
          if (io_comm_info != MPI_INFO_NULL)
            {
              throw_assert(MPI_Info_free(&io_comm_info) == MPI_SUCCESS,
                           "append_cell_attribute_map: error in MPI_Info_free");
            }
          status = H5Fclose(file);
          throw_assert(status == 0, "append_cell_attribute_map: unable to close HDF5 file");
        }
      
      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "append_cell_attribute_map: error in MPI_Barrier");
                   
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
     const std::map<CELL_IDX_T, deque<T>>& value_map,
     const size_t io_size,
     const data::optional_hid        data_type,
     const CellIndex                 index_type = IndexOwner,
     const CellPtr                   ptr_type = CellPtr(PtrOwner),
     const size_t chunk_size = 4000,
     const size_t value_chunk_size = 4000,
     const size_t cache_size = 1*1024*1024
     )
    {
      herr_t status;
      int size, rank;
      throw_assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS,
                   "append_cell_attribute: unable to obtain MPI communicator size");
      throw_assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS,
                   "append_cell_attribute: unable to obtain MPI communicator rank");
      
      hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);

      set<size_t> io_rank_set;
      data::range_sample(size, io_size, io_rank_set);
      bool is_io_rank = false;
      if (io_rank_set.find(rank) != io_rank_set.end())
        is_io_rank = true;
      throw_assert(io_rank_set.size() > 0, "invalid I/O rank set");

      int io_color = 1, color;
      MPI_Comm io_comm;
      
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

      throw_assert(H5Pset_fapl_mpio(fapl, io_comm, MPI_INFO_NULL) >= 0,
                   "append_cell_attribute_map: HDF5 mpio error");

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

      string attr_path = hdf5::cell_attribute_path(attr_namespace, pop_name, attr_name);

      if (rank == 0)
        {
          hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
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
          
          
          if (!(hdf5::exists_dataset (file, attr_path) > 0))
            {
              create_cell_attribute_datasets(file, attr_namespace, pop_name, attr_name,
                                             ftype, index_type, ptr_type,
                                             chunk_size, value_chunk_size
                                             );
            }
          status = H5Fclose(file);
          throw_assert(status == 0, "append_cell_attribute: unable to close HDF5 file");
        }
      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "append_cell_attribute: error in MPI_Barrier");

      hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDWR, fapl);
      throw_assert(file >= 0,
                   "append_cell_attribute: HDF5 file open error");

      append_cell_attribute_map<T>(comm, file, attr_namespace, pop_name, pop_start, attr_name, value_map,
                                   io_size, data_type, IndexOwner, CellPtr(PtrOwner),
                                   chunk_size, value_chunk_size, cache_size);

      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS, "error in MPI_Barrier");
      throw_assert(MPI_Comm_free(&io_comm) == MPI_SUCCESS,
                   "append_cell_attribute_map: error in MPI_Comm_free");

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
     const std::map<CELL_IDX_T, deque<T>>& value_map,
     const size_t io_size,
     const data::optional_hid        data_type,
     const size_t chunk_size = 4000,
     const size_t value_chunk_size = 4000,
     const size_t cache_size = 1*1024*1024
     )
    {
      append_cell_attribute_map<T>(comm, file_name, attr_namespace, pop_name, pop_start, attr_name, value_map,
                                   io_size, data_type, IndexOwner, CellPtr(PtrOwner),
                                   chunk_size, value_chunk_size, cache_size);
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

      int size, rank;
      throw_assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS,
                   "append_cell_attribute: unable to obtain MPI communicator size");
      throw_assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS,
                   "append_cell_attribute: unable to obtain MPI communicator rank");
      
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


      T dummy;
      hid_t ftype;
      if (data_type.has_value())
        ftype = data_type.value();
      else
        ftype = infer_datatype(dummy);

      throw_assert(ftype >= 0, "error in infer_datatype");

      string attr_path = hdf5::cell_attribute_path(attr_namespace, pop_name, attr_name);

      if (rank == 0)
        {
          hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
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
          
          
          if (!(hdf5::exists_dataset (file, attr_path) > 0))
            {
              create_cell_attribute_datasets(file, attr_namespace, pop_name, attr_name,
                                             ftype, index_type, ptr_type,
                                             chunk_size, value_chunk_size
                                             );
            }
          status = H5Fclose(file);
          throw_assert(status == 0, "append_cell_attribute: unable to close HDF5 file");
        }
      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "append_cell_attribute: error in MPI_Barrier");
    
      hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDWR, fapl);
      throw_assert(file >= 0, "write_cell_attribute: unable to open file");

      vector<CELL_IDX_T> rindex;

      for (const CELL_IDX_T& gid: index)
        {
          rindex.push_back(gid - pop_start);
        }
    
      hdf5::write_cell_attribute<T> (comm, file, attr_path,
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
     const std::map<CELL_IDX_T, deque<T>>& value_map,
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
      throw_assert(io_rank_set.size() > 0, "invalid I/O rank set");
      
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
      vector<uint32_t> recvbuf_num_values(size, 0);
      {
        vector<uint32_t> sendbuf_num_values(size, value_map.size());
        throw_assert(MPI_Allgather(&sendbuf_num_values[0], 1, MPI_UINT32_T,
                                   &recvbuf_num_values[0], 1, MPI_UINT32_T, comm)
                     == MPI_SUCCESS, "write_cell_attribute_map: error in MPI_Allgather");
      }

      // Determine local value size and offset
      uint32_t local_value_size=0;
      vector<ATTR_PTR_T> local_attr_size_vector;
      vector<CELL_IDX_T> local_index_vector;
      for (auto const& element : value_map)
        {
          const CELL_IDX_T gid = element.first;
          const deque<T> &v = element.second;
          local_value_size += v.size();
	  local_index_vector.push_back(gid);
	  local_attr_size_vector.push_back(v.size());
        }
      
      vector<CELL_IDX_T> gid_recvbuf;
      {
        vector<int> idx_sendcounts(size, 0), idx_sdispls(size, 0), idx_recvcounts(size, 0), idx_rdispls(size, 0);
        idx_sendcounts[io_dests[rank]] = local_index_vector.size();
        
        throw_assert(mpi::alltoallv_vector<CELL_IDX_T>(comm, MPI_CELL_IDX_T,
                                                       idx_sendcounts, idx_sdispls, local_index_vector,
                                                       idx_recvcounts, idx_rdispls, gid_recvbuf) >= 0,
                     "write_cell_attribute_map: error in MPI_Alltoallv");
      }

      vector<ATTR_PTR_T> attr_ptr;
      {
        vector<ATTR_PTR_T> attr_size_recvbuf;
        vector<int> attr_size_sendcounts(size, 0), attr_size_sdispls(size, 0), attr_size_recvcounts(size, 0), attr_size_rdispls(size, 0);
        attr_size_sendcounts[io_dests[rank]] = local_attr_size_vector.size();

        throw_assert(mpi::alltoallv_vector<ATTR_PTR_T>(comm, MPI_ATTR_PTR_T,
                                                       attr_size_sendcounts, attr_size_sdispls, local_attr_size_vector,
                                                       attr_size_recvcounts, attr_size_rdispls, attr_size_recvbuf) >= 0,
                     "write_cell_attribute_map: error in MPI_Alltoallv");
        
        if ((is_io_rank) && (attr_size_recvbuf.size() > 0))
          {
            ATTR_PTR_T attr_ptr_offset = 0;
            for (size_t s=0; s<ssize; s++)
              {
                int count = attr_size_recvcounts[s];
                for (size_t i=attr_size_rdispls[s]; i<attr_size_rdispls[s]+count; i++)
                  {
                    ATTR_PTR_T this_attr_size = attr_size_recvbuf[i];
                    attr_ptr.push_back(attr_ptr_offset);
                    attr_ptr_offset += this_attr_size;
                  }
              }
            attr_ptr.push_back(attr_ptr_offset);
          }
      }

      vector<T> value_recvbuf;
      {
        vector<uint32_t> recvbuf_size_values(size, 0);
        vector<uint32_t> sendbuf_size_values(size, local_value_size);
        throw_assert(MPI_Allgather(&sendbuf_size_values[0], 1, MPI_UINT32_T,
                                   &recvbuf_size_values[0], 1, MPI_UINT32_T, comm)
                     == MPI_SUCCESS, "write_cell_attribute_map: error in MPI_Allgather");
        throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                       "write_cell_attribute_map: error in MPI_Barrier");
        
        vector<T> local_value_vector;
        for (auto const& element : value_map)
          {
            const CELL_IDX_T gid = element.first;
            const deque<T> &v = element.second;
            local_value_vector.insert(local_value_vector.end(),v.begin(),v.end());
          }
        
        vector<int> value_sendcounts(size, 0), value_sdispls(size, 0), value_recvcounts(size, 0), value_rdispls(size, 0);
        value_sendcounts[io_dests[rank]] = local_value_size;
        

        T dummy;
        MPI_Datatype mpi_type = infer_mpi_datatype(dummy);
        throw_assert(mpi::alltoallv_vector<T>(comm, mpi_type,
                                              value_sendcounts, value_sdispls, local_value_vector,
                                              value_recvcounts, value_rdispls, value_recvbuf) >= 0,
                     "write_cell_attribute_map: error in MPI_Alltoallv");
      }
      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "write_cell_attribute_map: error in MPI_Barrier");
      
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
                                  gid_recvbuf, attr_ptr, value_recvbuf,
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


    template <typename T>
    void write_cell_attribute_map
    (
     MPI_Comm                        comm,
     const std::string&              file_name,
     const std::string&              attr_namespace,
     const std::string&              pop_name,
     const CELL_IDX_T&               pop_start,
     const std::string&              attr_name,
     const std::map<CELL_IDX_T, deque<T>>& value_map,
     const size_t io_size,
     const data::optional_hid        data_type,
     const size_t chunk_size = 4000,
     const size_t value_chunk_size = 4000,
     const size_t cache_size = 1*1024*1024
     )
    {
      write_cell_attribute_map<T>(comm, file_name, attr_namespace, pop_name, pop_start, attr_name,
                                  value_map, io_size, data_type, IndexOwner, CellPtr(PtrOwner),
                                  chunk_size, value_chunk_size, cache_size);
    }
  }
  
}



#endif
