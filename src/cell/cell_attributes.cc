// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file cell_attributes.cc
///
///  Routines for manipulation of scalar and vector attributes associated with a cell id.
///
///  Copyright (C) 2016-2020 Project NeuroH5.
//==============================================================================

#include "neuroh5_types.hh"
#include "path_names.hh"
#include "read_template.hh"
#include "write_template.hh"
#include "hdf5_cell_attributes.hh"
#include "exists_dataset.hh"
#include "dataset_num_elements.hh"
#include "create_group.hh"
#include "append_rank_attr_map.hh"
#include "attr_map.hh"
#include "infer_datatype.hh"
#include "attr_kind_datatype.hh"
#include "alltoallv_template.hh"
#include "serialize_data.hh"
#include "serialize_cell_attributes.hh"
#include "range_sample.hh"
#include "mpe_seq.hh"
#include "throw_assert.hh"

#include <hdf5.h>
#include <mpi.h>

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>
#include <set>

using namespace std;
using namespace neuroh5;

namespace neuroh5
{
  
  namespace hdf5
  {
    void size_cell_attributes
    (
     MPI_Comm         comm,
     hid_t            loc,
     const string&    path,
     const CellPtr&   ptr_type,
     hsize_t&         ptr_size,
     hsize_t&         index_size,
     hsize_t&         value_size
     )
    {
      string ptr_name;
      switch (ptr_type.type)
        {
        case PtrOwner:
          {
            string ptr_name;
            if (ptr_type.shared_ptr_name.has_value())
              {
                ptr_name = ptr_type.shared_ptr_name.value();
              }
            else
              {
                ptr_name = hdf5::ATTR_PTR;
              }
            ptr_size = hdf5::dataset_num_elements(loc, path + "/" + ptr_name);
          }
          break;
        case PtrShared:
          {
            string ptr_name;
            throw_assert (ptr_type.shared_ptr_name.has_value(),
                          "size_cell_attributes: shared attribute pointer has no value");
            ptr_name = ptr_type.shared_ptr_name.value();
            ptr_size = hdf5::dataset_num_elements(loc, ptr_name);
          }
          break;
        case PtrNone:
          break;
        }
      index_size = hdf5::dataset_num_elements(loc, path + "/" + hdf5::CELL_INDEX);
      value_size = hdf5::dataset_num_elements(loc, path + "/" + hdf5::ATTR_VAL);
    }
  }
  
  namespace cell
  {
    herr_t name_space_iterate_cb
    (
     hid_t             grp,
     const char*       name,
     const H5L_info_t* info,
     void*             op_data
     )
    {
      vector<string>* ptr = (vector<string>*)op_data;
      ptr->push_back(string(name));
      return 0;
    }


    
    herr_t get_cell_attribute_name_spaces
    (
     const string&       file_name,
     const string&       pop_name,
     vector< string>&    out_name_spaces
     )
    {
      hid_t in_file;
      herr_t ierr;
    
      in_file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      throw_assert(in_file >= 0,
                   "get_cell_attribute_name_spaces: unable to open file " << file_name);
      out_name_spaces.clear();
    
      string path = "/" + hdf5::POPULATIONS + "/" + pop_name;
      if (hdf5::exists_group (in_file, path) > 0)
        {

          hid_t grp = H5Gopen2(in_file, path.c_str(), H5P_DEFAULT);
          throw_assert(grp >= 0,
                       "get_cell_attribute_name_spaces: unable to open group " << path);
          
          hsize_t idx = 0;
          ierr = H5Literate(grp, H5_INDEX_NAME, H5_ITER_NATIVE, &idx,
                            &name_space_iterate_cb, (void*) &out_name_spaces);
          
          throw_assert(H5Gclose(grp) >= 0,
                       "get_cell_attribute_name_spaces: unable to close group " << path);
        }

      ierr = H5Fclose(in_file);
    
      return ierr;
    }
    
    // Callback for H5Literate
    static herr_t cell_attribute_cb
    (
     hid_t             grp,
     const char*       name,
     const H5L_info_t* info,
     void*             op_data
     )
    {
      herr_t ierr;
      
      /* Save old error handler */
      H5E_auto2_t error_handler;
      void *client_data;
      H5Eget_auto(H5E_DEFAULT, &error_handler, &client_data);
      
      /* Turn off error handling */
      H5Eset_auto(H5E_DEFAULT, NULL, NULL);
 
      ierr = H5Gget_objinfo (grp, name, 0, NULL);
      if (ierr == 0)
        {
          string value_path = string(name) + "/" + hdf5::ATTR_VAL;

          /* Restore previous error handler */
          hid_t dset = H5Dopen2(grp, value_path.c_str(), H5P_DEFAULT);
          if (dset < 0) // skip the link, if this is not a dataset
            {
              H5Eset_auto(H5E_DEFAULT, error_handler, client_data);
              return 0;
            }
    
          hid_t ftype = H5Dget_type(dset);
          throw_assert(ftype >= 0,
                       "cell_attributes_cb: unable to get data set type");
          
          vector< pair<string,AttrKind> >* ptr =
            (vector< pair<string,AttrKind> >*) op_data;
          ptr->push_back(make_pair(name, hdf5::h5type_attr_kind(ftype)));
          
          throw_assert(H5Dclose(dset) >= 0,
                       "cell_attributes_cb: unable to close data set");
        }
      
      H5Eset_auto(H5E_DEFAULT, error_handler, client_data);
      return 0;
    }


    // Callback for H5Literate
    static herr_t cell_attribute_index_ptr_cb
    (
     hid_t             grp,
     const char*       name,
     const H5L_info_t* info,
     void*             op_data
     )
    {
      herr_t ierr;
      
      /* Save old error handler */
      H5E_auto2_t error_handler;
      void *client_data;
      H5Eget_auto(H5E_DEFAULT, &error_handler, &client_data);
      
      /* Turn off error handling */
      H5Eset_auto(H5E_DEFAULT, NULL, NULL);
 
      ierr = H5Gget_objinfo (grp, name, 0, NULL);
      if (ierr == 0)
        {
          string value_path = string(name) + "/" + hdf5::ATTR_VAL;

          /* Restore previous error handler */
          hid_t dset = H5Dopen2(grp, value_path.c_str(), H5P_DEFAULT);
          if (dset < 0) // skip the link, if this is not a dataset
            {
              H5Eset_auto(H5E_DEFAULT, error_handler, client_data);
              return 0;
            }
    
          H5Eset_auto(H5E_DEFAULT, error_handler, client_data);
          
          hid_t ftype = H5Dget_type(dset);
          throw_assert(ftype >= 0,
                       "cell_attributes_cb: unable to get data set type");
          
          vector< tuple<string,AttrKind,vector<CELL_IDX_T>,vector<ATTR_PTR_T> > >* ptr =
            (vector< tuple<string,AttrKind,vector<CELL_IDX_T>,vector<ATTR_PTR_T> > >*) op_data;


          vector<CELL_IDX_T> attr_index; vector<ATTR_PTR_T> attr_ptr;
          string attr_path = string(name);
          ierr = hdf5::read_cell_index_ptr(grp, attr_path, attr_index, attr_ptr);
          throw_assert(ierr >= 0,
                       "cell_attributes_index_ptr_cb: error in hdf5::read_cell_index_ptr");

          ptr->push_back(make_tuple(name, hdf5::h5type_attr_kind(ftype), attr_index, attr_ptr));
          
          throw_assert(H5Dclose(dset) >= 0,
                       "cell_attributes_cb: unable to close data set");
        }
      
      H5Eset_auto(H5E_DEFAULT, error_handler, client_data);
      return 0;
    }

  
    herr_t get_cell_attributes
    (
     const string&                 file_name,
     const string&                 name_space,
     const string&                 pop_name,
     vector< pair<string,AttrKind> >& out_attributes
     )
    {
      hid_t in_file;
      herr_t ierr;
    
      in_file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      throw_assert(in_file >= 0, "get_cell_attributes unable to open file " << file_name);
      out_attributes.clear();
    
      string path = hdf5::cell_attribute_prefix(name_space, pop_name);

      if (hdf5::exists_dataset (in_file, path) > 0)
        {
          hid_t grp = H5Gopen2(in_file, path.c_str(), H5P_DEFAULT);
          throw_assert(grp >= 0,
                       "get_cell_attributes: unable to open group " << path);

          
          hsize_t idx = 0;
          ierr = H5Literate(grp, H5_INDEX_NAME, H5_ITER_NATIVE, &idx,
                            &cell_attribute_cb, (void*) &out_attributes);
          
          ierr = H5Gclose(grp);
          throw_assert(ierr >= 0,
                       "get_cell_attributes: unable to close group " << path);
        }
      ierr = H5Fclose(in_file);

      return ierr;
    }


    herr_t get_cell_attributes_index_ptr
    (
     const string&                 file_name,
     const string&                 name_space,
     const string&                 pop_name,
     const CELL_IDX_T&             pop_start,
     vector< tuple<string,AttrKind,vector<CELL_IDX_T>,vector<ATTR_PTR_T> > >& out_attributes
     )
    {
      hid_t in_file;
      herr_t ierr;
    
      in_file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      throw_assert(in_file >= 0, "get_call_attributes_index_ptr: unable to open file " << file_name);
      out_attributes.clear();
    
      string path = hdf5::cell_attribute_prefix(name_space, pop_name);

      if (hdf5::exists_dataset (in_file, path) > 0)
        {
          hid_t grp = H5Gopen2(in_file, path.c_str(), H5P_DEFAULT);
          throw_assert(grp >= 0,
                       "get_cell_attributes_index_ptr: unable to open group " << path);

          
          hsize_t idx = 0;
          ierr = H5Literate(grp, H5_INDEX_NAME, H5_ITER_NATIVE, &idx,
                            &cell_attribute_index_ptr_cb, (void*) &out_attributes);

          for (size_t i=0; i<out_attributes.size(); i++)
            {
              vector<CELL_IDX_T>& index  = get<2>(out_attributes[i]);
                
              for (size_t j=0; j<index.size(); j++)
                {
                  index[j] += pop_start;
                }
            }
          
          ierr = H5Gclose(grp);
          throw_assert(ierr >= 0,
                       "get_cell_attributes_index_ptr: unable to close group " << path);
        }
      
      ierr = H5Fclose(in_file);
      return ierr;
    }

    

    herr_t num_cell_attributes
    (
     const vector< pair<string,AttrKind> >& attributes,
     vector <size_t> &num_attrs
     )
    {
      herr_t ierr = 0;
      num_attrs.resize(data::AttrMap::num_attr_types);
      for (size_t i = 0; i < attributes.size(); i++)
        {
          AttrKind attr_kind = attributes[i].second;
          size_t attr_size = attr_kind.size;
          switch (attr_kind.type)
            {
            case UIntVal:
              if (attr_size == 4)
                {
                  num_attrs[data::AttrMap::attr_index_uint32]++;
                }
              else if (attr_size == 2)
                {
                  num_attrs[data::AttrMap::attr_index_uint16]++;
                }
              else if (attr_size == 1)
                {
                  num_attrs[data::AttrMap::attr_index_uint8]++;
                }
              else
                {
                  throw runtime_error("Unsupported integer attribute size");
                };
              break;
            case SIntVal:
              if (attr_size == 4)
                {
                  num_attrs[data::AttrMap::attr_index_int32]++;
                }
              else if (attr_size == 2)
                {
                  num_attrs[data::AttrMap::attr_index_int16]++;
                }
              else if (attr_size == 1)
                {
                  num_attrs[data::AttrMap::attr_index_int8]++;
                }
              else
                {
                  throw runtime_error("Unsupported integer attribute size");
                };
              break;
              
            case FloatVal:
              num_attrs[data::AttrMap::attr_index_float]++;
              break;
            case EnumVal:
              if (attr_size == 1)
                {
                  num_attrs[data::AttrMap::attr_index_uint8]++;
                }
              else
                {
                  throw runtime_error("Unsupported enumerated attribute size");
                };
              break;
            default:
              throw runtime_error("Unsupported attribute type");
              break;
            }

        }

      return ierr;
    }


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
     )
    {
      herr_t status;
      hsize_t maxdims[1] = {H5S_UNLIMITED};
      hsize_t cdims[1]   = {chunk_size}; /* chunking dimensions */		
      hsize_t initial_size = 0;
    
      hid_t plist  = H5Pcreate (H5P_DATASET_CREATE);
      status = H5Pset_layout(plist, H5D_CHUNKED);
      throw_assert(status == 0, "create_cell_attribute_datasets: unable to set chunked layout");
      status = H5Pset_chunk(plist, 1, cdims);
      throw_assert(status == 0,
                   "create_cell_attribute_datasets: unable to set chunk size");
      status = H5Pset_alloc_time(plist, H5D_ALLOC_TIME_EARLY);
      throw_assert(status == 0,
                   "create_cell_attribute_datasets: unable to set allocation time");
#ifdef H5_HAS_PARALLEL_DEFLATE
      status = H5Pset_deflate(plist, 6);
      throw_assert(status == 0,
                   "create_cell_attribute_datasets: unable to add deflate filter");
#endif

      
      hsize_t value_cdims[1]   = {value_chunk_size}; /* chunking dimensions for value dataset */		
      hid_t value_plist = H5Pcreate (H5P_DATASET_CREATE);
      status = H5Pset_layout(value_plist, H5D_CHUNKED);
      throw_assert(status == 0,
                   "create_cell_attribute_datasets: unable to set chunked layout");
      status = H5Pset_chunk(value_plist, 1, value_cdims);
      throw_assert(status == 0,
                   "create_cell_attribute_datasets: unable to set chunk size");
      status = H5Pset_alloc_time(value_plist, H5D_ALLOC_TIME_EARLY);
      throw_assert(status == 0,
                   "create_cell_attribute_datasets: unable to set allocation time");
#ifdef H5_HAS_PARALLEL_DEFLATE
      status = H5Pset_deflate(value_plist, 6);
      throw_assert(status == 0,
                   "create_cell_attribute_datasets: unable to add deflate filter");
#endif
      
      hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
      throw_assert(lcpl >= 0,
                   "create_cell_attribute_datasets: unable to create link property list");
      throw_assert(H5Pset_create_intermediate_group(lcpl, 1) >= 0,
                   "create_cell_attribute_datasets: unable to set create intermediate group property");
    
      if (!(hdf5::exists_dataset (file, ("/" + hdf5::POPULATIONS)) > 0))
        {
          hdf5::create_group(file, ("/" + hdf5::POPULATIONS).c_str());
        }

      if (!(hdf5::exists_dataset (file, hdf5::population_path(pop_name)) > 0))
        {
          hdf5::create_group(file, hdf5::population_path(pop_name));
        }

      string attr_prefix = hdf5::cell_attribute_prefix(attr_namespace, pop_name);
      if (!(hdf5::exists_dataset (file, attr_prefix) > 0))
        {
          hdf5::create_group(file, attr_prefix);
        }

      string attr_path = hdf5::cell_attribute_path(attr_namespace, pop_name, attr_name);
      if (!(hdf5::exists_dataset (file, attr_path) > 0))
        {
          hdf5::create_group(file, attr_path);
        }

      hid_t mspace, dset;
      
      switch (index_type)
        {
        case IndexOwner:
          {
            mspace = H5Screate_simple(1, &initial_size, maxdims);
            throw_assert(mspace >= 0,
                         "create_cell_attribute_datasets: unable to create memory space");

            dset = H5Dcreate2(file, (attr_path + "/" + hdf5::CELL_INDEX).c_str(),
                              CELL_IDX_H5_FILE_T,
                              mspace, lcpl, plist, H5P_DEFAULT);
            throw_assert(H5Dclose(dset) >= 0,
                         "create_cell_attribute_datasets: unable to create data set");
            throw_assert(H5Sclose(mspace) >= 0,
                         "create_cell_attribute_datasets: unable to close memory space");
            
          }
          break;
        case IndexShared:
          {
            dset = H5Dopen2(file, (attr_prefix + "/" + hdf5::CELL_INDEX).c_str(), H5P_DEFAULT);
            throw_assert(dset >= 0,
                         "create_cell_attribute_datasets: unable to open cell index data set");
            status = H5Olink(dset, file, (attr_path + "/" + hdf5::CELL_INDEX).c_str(), H5P_DEFAULT, H5P_DEFAULT);
            throw_assert(status >= 0,
                         "create_cell_attribute_datasets: unable to link to shared index data set");
            throw_assert(H5Dclose(dset) >= 0,
                         "create_cell_attribute_datasets: unable to close index data set");
          }
          break;
        case IndexNone:
          break;
        }
      
      switch (ptr_type.type)
        {
        case PtrOwner:
          {
            string ptr_name;
            if (ptr_type.shared_ptr_name.has_value())
              {
                ptr_name = ptr_type.shared_ptr_name.value();
              }
            else
              {
                ptr_name = hdf5::ATTR_PTR;
              }

            mspace = H5Screate_simple(1, &initial_size, maxdims);
            throw_assert(mspace >= 0,
                         "create_cell_attribute_datasets: unable to create memory space");

            dset = H5Dcreate2(file, (attr_path + "/" + ptr_name).c_str(), ATTR_PTR_H5_FILE_T,
                              mspace, lcpl, plist, H5P_DEFAULT);
            throw_assert(status >= 0,
                         "create_cell_attribute_datasets: unable to create attribute pointer data set");
            if (ptr_name.compare(hdf5::ATTR_PTR) != 0)
              {
                status = H5Olink(dset, file, (attr_path + "/" + hdf5::ATTR_PTR).c_str(), H5P_DEFAULT, H5P_DEFAULT);
                throw_assert(status >= 0,
                         "create_cell_attribute_datasets: unable to link attribute pointer data set");
              }
            throw_assert(H5Dclose(dset) >= 0,
                         "create_cell_attribute_datasets: unable to close pointer data set");
            throw_assert(H5Sclose(mspace) >= 0,
                         "create_cell_attribute_datasets: unable to close memory space");
          }
          break;
        case PtrShared:
          {
            throw_assert(ptr_type.shared_ptr_name.has_value(),
                         "create_cell_attribute_datasets: shared pointer data set name has not been set");

            dset = H5Dopen2(file, (ptr_type.shared_ptr_name.value()).c_str(), H5P_DEFAULT);
            throw_assert(dset >= 0,
                         "create_cell_attribute_datasets: unable to open shared attribute pointer data set");
            status = H5Olink(dset, file, (attr_path + "/" + hdf5::ATTR_PTR).c_str(), H5P_DEFAULT, H5P_DEFAULT);
            throw_assert(status >= 0,
                         "create_cell_attribute_datasets: unable to link shared attribute pointer data set");
            throw_assert(H5Dclose(dset) >= 0,
                         "create_cell_attribute_datasets: unable to close shared attribute pointer data set");
          }
          break;
        case PtrNone:
          break;
        }
      
    mspace = H5Screate_simple(1, &initial_size, maxdims);
    dset = H5Dcreate2(file, (attr_path + "/" + hdf5::ATTR_VAL).c_str(), ftype, mspace,
                      lcpl, value_plist, H5P_DEFAULT);
    throw_assert(H5Dclose(dset) >= 0,
                 "create_cell_attribute_datasets: unable to close attribute value data set");
                 
    throw_assert(H5Sclose(mspace) >= 0,
                 "create_cell_attribute_datasets: unable to close memory space");
    
    throw_assert(H5Pclose(lcpl) >= 0,
                 "create_cell_attribute_datasets: unable to close link creation property list");
    
    status = H5Pclose(plist);
    throw_assert(status == 0,
                 "create_cell_attribute_datasets: unable to close property list");
    status = H5Pclose(value_plist);
    
    throw_assert(status == 0,
                 "create_cell_attribute_datasets: unable to close value property list");
    
  }

  
    void read_cell_attributes
    (
     MPI_Comm      comm,
     const string& file_name,
     const string& name_space,
     const set<string>& attr_mask,
     const string& pop_name,
     const CELL_IDX_T& pop_start,
     data::NamedAttrMap& attr_values,
     size_t offset,
     size_t numitems
     )
    {
      herr_t status; 

      unsigned int rank, size;
      throw_assert(MPI_Comm_size(comm, (int*)&size) >= 0, "read_cell_attributes: error in MPI_Comm_size");
      throw_assert(MPI_Comm_rank(comm, (int*)&rank) >= 0, "read_cell_attributes: error in MPI_Comm_rank");

      vector< tuple<string,AttrKind,vector<CELL_IDX_T>,vector<ATTR_PTR_T> > > attr_info;
      map<CELL_IDX_T, rank_t> node_rank_map;

      if (rank == 0)
        {
          status = get_cell_attributes_index_ptr (file_name, name_space, pop_name, pop_start, attr_info);
          throw_assert(status == 0,
                       "read_cell_attributes: error in get_cell_attributes_index_ptr");
          // round-robin node to rank assignment from file
          if (attr_info.size() > 0)
            {
              const vector<CELL_IDX_T>& index  = get<2>(attr_info[0]);
              for (size_t i = 0; i < index.size(); i++)
                {
                  auto it = node_rank_map.find(index[i]);
                  if (it == node_rank_map.end())
                    {
                      node_rank_map.insert(make_pair(index[i], i%size));
                    }
                }
            }
        }
      // Broadcast the attribute names, types, indices, and pointers
      {
        vector<char> sendbuf; size_t sendbuf_size=0;
        if (rank == 0)
          {
            data::serialize_data(attr_info, sendbuf);
            sendbuf_size = sendbuf.size();
          }

        throw_assert(MPI_Bcast(&sendbuf_size, 1, MPI_SIZE_T, 0, comm) == MPI_SUCCESS,
                     "read_cell_attributes: error in MPI_Bcast");
        sendbuf.resize(sendbuf_size);
        throw_assert(MPI_Bcast(&sendbuf[0], sendbuf_size, MPI_CHAR, 0, comm) == MPI_SUCCESS,
                     "read_cell_attributes: error in MPI_Bcast");
        
        if (rank != 0)
          {
            data::deserialize_data(sendbuf, attr_info);
          }
      }
      // Broadcast the node rank map
      {
        vector<char> sendbuf; size_t sendbuf_size=0;
        if (rank == 0)
          {
            data::serialize_data(node_rank_map, sendbuf);
            sendbuf_size = sendbuf.size();
          }

        throw_assert(MPI_Bcast(&sendbuf_size, 1, MPI_SIZE_T, 0, comm) == MPI_SUCCESS,
                     "read_cell_attributes: error in MPI_Bcast");
        sendbuf.resize(sendbuf_size);
        throw_assert(MPI_Bcast(&sendbuf[0], sendbuf_size, MPI_CHAR, 0, comm) == MPI_SUCCESS,
                     "read_cell_attributes: error in MPI_Bcast");
        
        if (rank != 0)
          {
            data::deserialize_data(sendbuf, node_rank_map);
          }
      }

      // get a file handle and retrieve the MPI info
      hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
      throw_assert_nomsg(H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL) >= 0);
      hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, fapl);
      throw_assert_nomsg(file >= 0);

      for (size_t i=0; i<attr_info.size(); i++)
        {
          vector<CELL_IDX_T>  value_index;
          vector<ATTR_PTR_T>  value_ptr;

          string attr_name  = get<0>(attr_info[i]);
          AttrKind attr_kind = get<1>(attr_info[i]);
          size_t attr_size  = attr_kind.size;
          const vector<CELL_IDX_T>& index  = get<2>(attr_info[i]);
          const vector<ATTR_PTR_T>& ptr  = get<3>(attr_info[i]);
          
          string attr_path  = hdf5::cell_attribute_path (name_space, pop_name, attr_name);

          if ((attr_mask.size() > 0) && (attr_mask.count(attr_name) == 0))
            continue;
          
          switch (attr_kind.type)
            {
            case UIntVal:
              if (attr_size == 4)
                {
                  vector<uint32_t> attr_values_uint32;
                  status = hdf5::read_cell_attribute(comm, file, attr_path, pop_start,
                                                     index, ptr, value_index, value_ptr,
                                                     attr_values_uint32,
                                                     offset, numitems);
                  attr_values.insert(attr_name, value_index, value_ptr, attr_values_uint32);
                }
              else if (attr_size == 2)
                {
                  vector<uint16_t> attr_values_uint16;
                  status = hdf5::read_cell_attribute(comm, file, attr_path, pop_start,
                                                     index, ptr, value_index, value_ptr, attr_values_uint16,
                                                     offset, numitems);
                  attr_values.insert(attr_name, value_index, value_ptr, attr_values_uint16);
                }
              else if (attr_size == 1)
                {
                  vector<uint8_t> attr_values_uint8;
                  status = hdf5::read_cell_attribute(comm, file, attr_path, pop_start,
                                                     index, ptr, value_index, value_ptr, attr_values_uint8,
                                                     offset, numitems);
                  attr_values.insert(attr_name, value_index, value_ptr, attr_values_uint8);
                }
              else
                {
                  throw runtime_error("Unsupported integer attribute size");
                };
              break;
            case SIntVal:
              {
                if (attr_size == 4)
                  {
                    vector<int32_t> attr_values_int32;
                    status = hdf5::read_cell_attribute(comm, file, attr_path, pop_start,
                                                       index, ptr, value_index, value_ptr, attr_values_int32,
                                                       offset, numitems);
                    attr_values.insert(attr_name, value_index, value_ptr, attr_values_int32);
                  }
                else if (attr_size == 2)
                  {
                    vector<int16_t> attr_values_int16;
                    status = hdf5::read_cell_attribute(comm, file, attr_path, pop_start,
                                                       index, ptr, value_index, value_ptr, attr_values_int16,
                                                       offset, numitems);
                    attr_values.insert(attr_name, value_index, value_ptr, attr_values_int16);
                  }
                else if (attr_size == 1)
                  {
                    vector<int8_t> attr_values_int8;
                    status = hdf5::read_cell_attribute(comm, file, attr_path, pop_start,
                                                       index, ptr, value_index, value_ptr, attr_values_int8,
                                                       offset, numitems);
                    attr_values.insert(attr_name, value_index, value_ptr, attr_values_int8);
                  }
                else
                  {
                    throw runtime_error("Unsupported integer attribute size");
                  };
              }
              break;
            case FloatVal:
              {
                vector<float> attr_values_float;
                status = hdf5::read_cell_attribute(comm, file, attr_path, pop_start,
                                                   index, ptr, value_index, value_ptr, attr_values_float,
                                                   offset, numitems);
                attr_values.insert(attr_name, value_index, value_ptr, attr_values_float);
              }
              break;
            case EnumVal:
              {
                if (attr_size == 1)
                  {
                    vector<uint8_t> attr_values_uint8;
                    status = hdf5::read_cell_attribute(comm, file, attr_path, pop_start,
                                                       index, ptr, value_index, value_ptr, attr_values_uint8,
                                                       offset, numitems);
                    attr_values.insert(attr_name, value_index, value_ptr, attr_values_uint8);
                  }
                else
                  {
                    throw runtime_error("Unsupported enumerated attribute size");
                  };
              }
              break;
            default:
              throw runtime_error("Unsupported attribute type");
              break;
            }
        }

      status = H5Fclose(file);
      throw_assert_nomsg(status == 0);
      status = H5Pclose(fapl);
      throw_assert_nomsg(status == 0);


    }


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
     )
    {
      int srank, ssize; size_t rank, size;
      throw_assert_nomsg(MPI_Comm_size(all_comm, &ssize) >= 0);
      throw_assert_nomsg(MPI_Comm_rank(all_comm, &srank) >= 0);
      throw_assert_nomsg(ssize > 0);
      throw_assert_nomsg(srank >= 0);
      size = ssize;
      rank = srank;

      vector< size_t > num_attrs(data::AttrMap::num_attr_types, 0);
      vector< vector<string> > attr_names;

      // MPI Communicator for I/O ranks
      MPI_Comm io_comm;
      // MPI group color value used for I/O ranks
      int io_color = 1;

      throw_assert_nomsg(io_size > 0);
    
      vector<char> sendbuf; 
      vector<int> sendcounts(size,0), sdispls(size,0), recvcounts(size,0), rdispls(size,0);

      set<size_t> io_rank_set;
      data::range_sample(size, io_size, io_rank_set);
      bool is_io_rank = false;
      if (io_rank_set.find(rank) != io_rank_set.end())
        is_io_rank = true;

      if (is_io_rank)
        {
          // Am I an I/O rank?
          MPI_Comm_split(all_comm,io_color,rank,&io_comm);
          MPI_Comm_set_errhandler(io_comm, MPI_ERRORS_RETURN);

          map <rank_t, data::AttrMap > rank_attr_map;
          {
            data::NamedAttrMap  attr_values;
            read_cell_attributes(io_comm, file_name, attr_name_space, attr_mask, pop_name, pop_start,
                                 attr_values, offset, numitems * size);
            data::append_rank_attr_map(attr_values, node_rank_map, rank_attr_map);
            attr_values.num_attrs(num_attrs);
            attr_values.attr_names(attr_names);
          }

          data::serialize_rank_attr_map (size, rank, rank_attr_map, sendcounts, sendbuf, sdispls);
        }
      else
        {

          MPI_Comm_split(all_comm,0,rank,&io_comm);
        }
      MPI_Barrier(io_comm);
      MPI_Barrier(all_comm);
      throw_assert_nomsg(MPI_Comm_free(&io_comm) == MPI_SUCCESS);
    
      vector<size_t> num_attrs_bcast(num_attrs.size());
      for (size_t i=0; i<num_attrs.size(); i++)
        {
          num_attrs_bcast[i] = num_attrs[i];
        }
      // 4. Broadcast the number of attributes of each type to all ranks
      throw_assert_nomsg(MPI_Bcast(&num_attrs_bcast[0], num_attrs_bcast.size(), MPI_SIZE_T, 0, all_comm) >= 0);
      for (size_t i=0; i<num_attrs.size(); i++)
        {
          num_attrs[i] = num_attrs_bcast[i];
        }
    
      // 5. Broadcast the names of each attributes of each type to all ranks
      {
        vector<char> sendbuf; size_t sendbuf_size=0;
        if (rank == 0)
          {
            data::serialize_data(attr_names, sendbuf);
            sendbuf_size = sendbuf.size();
          }

        throw_assert_nomsg(MPI_Bcast(&sendbuf_size, 1, MPI_SIZE_T, 0, all_comm) >= 0);
        sendbuf.resize(sendbuf_size);
        throw_assert_nomsg(MPI_Bcast(&sendbuf[0], sendbuf_size, MPI_CHAR, 0, all_comm) >= 0);
        
        if (rank != 0)
          {
            data::deserialize_data(sendbuf, attr_names);
          }
      }
      
      for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_float]; i++)
        {
          attr_map.insert_name<float>(attr_names[data::AttrMap::attr_index_float][i]);
        }
      for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_uint8]; i++)
        {
          attr_map.insert_name<uint8_t>(attr_names[data::AttrMap::attr_index_uint8][i]);
        }
      for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_int8]; i++)
        {
          attr_map.insert_name<int8_t>(attr_names[data::AttrMap::attr_index_int8][i]);
        }
      for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_uint16]; i++)
        {
          attr_map.insert_name<uint16_t>(attr_names[data::AttrMap::attr_index_uint16][i]);
        }
      for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_int16]; i++)
        {
          attr_map.insert_name<int16_t>(attr_names[data::AttrMap::attr_index_int16][i]);
        }
      for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_uint32]; i++)
        {
          attr_map.insert_name<uint32_t>(attr_names[data::AttrMap::attr_index_uint32][i]);
        }
      for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_int32]; i++)
        {
          attr_map.insert_name<int32_t>(attr_names[data::AttrMap::attr_index_int32][i]);
        }
    
      // 6. Each ALL_COMM rank sends an attribute set size to
      //    every other ALL_COMM rank (non IO_COMM ranks pass zero)
    
      throw_assert_nomsg(MPI_Alltoall(&sendcounts[0], 1, MPI_INT,
                          &recvcounts[0], 1, MPI_INT, all_comm) >= 0);
    
      // 7. Each ALL_COMM rank accumulates the vector sizes and allocates
      //    a receive buffer, recvcounts, and rdispls
      size_t recvbuf_size;
      vector<char> recvbuf;

      recvbuf_size = recvcounts[0];
      for (int p = 1; p < ssize; ++p)
        {
          rdispls[p] = rdispls[p-1] + recvcounts[p-1];
          recvbuf_size += recvcounts[p];
        }
      if (recvbuf_size > 0)
        recvbuf.resize(recvbuf_size);

      // 8. Each ALL_COMM rank participates in the MPI_Alltoallv
      throw_assert_nomsg(mpi::alltoallv_vector<char>(all_comm, MPI_CHAR, sendcounts, sdispls, sendbuf,
                                                     recvcounts, rdispls, recvbuf) >= 0);

    
      sendbuf.clear();
    
      if (recvbuf.size() > 0)
        {
          data::deserialize_rank_attr_map (size, recvbuf, recvcounts, rdispls, attr_map);
        }
      recvbuf.clear();
      

      return 0;
    }

  
    void bcast_cell_attributes
    (
     MPI_Comm      comm,
     const int     root,
     const string& file_name,
     const string& name_space,
     const set<string>& attr_mask,
     const string& pop_name,
     const CELL_IDX_T& pop_start,
     data::NamedAttrMap& attr_map,
     size_t offset = 0,
     size_t numitems = 0
     )
    {
      herr_t status; 
      
      unsigned int rank, size;
      throw_assert_nomsg(MPI_Comm_size(comm, (int*)&size) >= 0);
      throw_assert_nomsg(MPI_Comm_rank(comm, (int*)&rank) >= 0);

      vector<char> sendrecvbuf; 
      vector< tuple<string,AttrKind,vector<CELL_IDX_T>,vector<ATTR_PTR_T> > > attr_info;

      if (rank == (unsigned int)root)
        {
          status = get_cell_attributes_index_ptr (file_name, name_space, pop_name, pop_start, attr_info);
          throw_assert(status == 0,
                       "read_cell_attributes: error in get_cell_attributes_index_ptr");
        }


      vector< size_t > num_attrs;
      num_attrs.resize(data::AttrMap::num_attr_types);
      vector< vector<string> > attr_names;
      attr_names.resize(data::AttrMap::num_attr_types);

      MPI_Comm io_comm;
      int io_color = 1;
      if (rank == (unsigned int)root)
        {
          MPI_Comm_split(comm,io_color,rank,&io_comm);
          MPI_Comm_set_errhandler(io_comm, MPI_ERRORS_RETURN);
        }
      else
        {
          MPI_Comm_split(comm,0,rank,&io_comm);
        }
      MPI_Barrier(comm);

      // get a file handle and retrieve the MPI info
      hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
      throw_assert_nomsg(H5Pset_fapl_mpio(fapl, io_comm, MPI_INFO_NULL) >= 0);
    
      hid_t file;
      if (rank == (unsigned int)root)
        {

          file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, fapl);
          throw_assert_nomsg(file >= 0);
        
          for (size_t i=0; i<attr_info.size(); i++)
            {
              vector<CELL_IDX_T>  value_index;
              vector<ATTR_PTR_T>  value_ptr;
            
              string attr_name  = get<0>(attr_info[i]);
              AttrKind attr_kind = get<1>(attr_info[i]);
              size_t attr_size  = attr_kind.size;
              const vector<CELL_IDX_T>& index  = get<2>(attr_info[i]);
              const vector<ATTR_PTR_T>& ptr  = get<3>(attr_info[i]);

              string attr_path  = hdf5::cell_attribute_path (name_space, pop_name, attr_name);

              if ((attr_mask.size() > 0) && (attr_mask.count(attr_name) == 0))
                continue;
              
              switch (attr_kind.type)
                {
                case UIntVal:
                  {
                    if (attr_size == 4)
                      {
                        vector<uint32_t> attr_map_uint32;
                        status = hdf5::read_cell_attribute(io_comm, file, attr_path, pop_start,
                                                           index, ptr, value_index, value_ptr,
                                                           attr_map_uint32);
                        attr_map.insert(attr_name, value_index, value_ptr, attr_map_uint32);
                      }
                    else if (attr_size == 2)
                      {
                        vector<uint16_t> attr_map_uint16;
                        status = hdf5::read_cell_attribute(io_comm, file, attr_path, pop_start,
                                                           index, ptr, value_index, value_ptr,
                                                           attr_map_uint16);
                        attr_map.insert(attr_name, value_index, value_ptr, attr_map_uint16);
                      }
                    else if (attr_size == 1)
                      {
                        vector<uint8_t> attr_map_uint8;
                        status = hdf5::read_cell_attribute(io_comm, file, attr_path, pop_start,
                                                           index, ptr, value_index, value_ptr,
                                                           attr_map_uint8);
                        attr_map.insert(attr_name, value_index, value_ptr, attr_map_uint8);
                      }
                    else
                      {
                        throw runtime_error("Unsupported integer attribute size");
                      };
                  }
                  break;
                case SIntVal:
                  {
                    if (attr_size == 4)
                      {
                        vector<int32_t> attr_map_int32;
                        status = hdf5::read_cell_attribute(io_comm, file, attr_path, pop_start,
                                                           index, ptr, value_index, value_ptr,
                                                           attr_map_int32);
                        attr_map.insert(attr_name, value_index, value_ptr, attr_map_int32);
                      }
                    else if (attr_size == 2)
                      {
                        vector<uint16_t> attr_map_int16;
                        status = hdf5::read_cell_attribute(io_comm, file, attr_path, pop_start,
                                                           index, ptr, value_index, value_ptr,
                                                           attr_map_int16);
                        attr_map.insert(attr_name, value_index, value_ptr, attr_map_int16);
                      }
                    else if (attr_size == 1)
                      {
                        vector<uint8_t> attr_map_int8;
                        status = hdf5::read_cell_attribute(io_comm, file, attr_path, pop_start,
                                                           index, ptr, value_index, value_ptr,
                                                           attr_map_int8);
                        attr_map.insert(attr_name, value_index, value_ptr, attr_map_int8);
                      }
                    else
                      {
                        throw runtime_error("Unsupported integer attribute size");
                      };
                  }
                  break;
                case FloatVal:
                  {
                    vector<float> attr_map_float;
                    status = hdf5::read_cell_attribute(io_comm, file, attr_path, pop_start,
                                                       index, ptr, value_index, value_ptr,
                                                       attr_map_float);
                    attr_map.insert(attr_name, value_index, value_ptr, attr_map_float);
                  }
                  break;
                case EnumVal:
                  {
                    if (attr_size == 1)
                      {
                        vector<uint8_t> attr_map_uint8;
                        status = hdf5::read_cell_attribute(io_comm, file, attr_path, pop_start,
                                                           index, ptr, value_index, value_ptr,
                                                           attr_map_uint8);
                        attr_map.insert(attr_name, value_index, value_ptr, attr_map_uint8);
                      }
                    else
                      {
                        throw runtime_error("Unsupported enumerated attribute size");
                      };
                  }
                  break;
                default:
                  throw runtime_error("Unsupported attribute type");
                  break;

                }

            
            }
          status = H5Fclose(file);
          throw_assert_nomsg(status == 0);

          attr_map.num_attrs(num_attrs);
          attr_map.attr_names(attr_names);

          data::serialize_data(attr_map, sendrecvbuf);

        }
      MPI_Barrier(io_comm);
      status = H5Pclose(fapl);
      throw_assert_nomsg(status == 0);
      throw_assert_nomsg(MPI_Comm_free(&io_comm) == MPI_SUCCESS);
      

      vector<size_t> num_attrs_bcast(num_attrs.size());
      for (size_t i=0; i<num_attrs.size(); i++)
        {
          num_attrs_bcast[i] = num_attrs[i];
        }
      // Broadcast the number of attributes of each type to all ranks
      throw_assert_nomsg(MPI_Bcast(&num_attrs_bcast[0], num_attrs_bcast.size(), MPI_SIZE_T, root, comm) >= 0);
      for (size_t i=0; i<num_attrs.size(); i++)
        {
          num_attrs[i] = num_attrs_bcast[i];
        }
    
      // Broadcast the names of each attributes of each type to all ranks
      {
        vector<char> sendbuf;
        size_t sendbuf_size=0;
        if (rank == (unsigned int)root)
          {
            data::serialize_data(attr_names, sendbuf);
            sendbuf_size = sendbuf.size();
          }
        
        throw_assert_nomsg(MPI_Bcast(&sendbuf_size, 1, MPI_SIZE_T, root, comm) >= 0);
        
        sendbuf.resize(sendbuf_size);
        throw_assert_nomsg(MPI_Bcast(&sendbuf[0], sendbuf.size(), MPI_CHAR, root, comm) >= 0);
        
        if (rank != (unsigned int)root)
          {
            data::deserialize_data(sendbuf, attr_names);
          }
      }
          
      
      for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_float]; i++)
        {
          attr_map.insert_name<float>(attr_names[data::AttrMap::attr_index_float][i]);
        }
      for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_uint8]; i++)
        {
          attr_map.insert_name<uint8_t>(attr_names[data::AttrMap::attr_index_uint8][i]);
        }
      for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_int8]; i++)
        {
          attr_map.insert_name<int8_t>(attr_names[data::AttrMap::attr_index_int8][i]);
        }
      for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_uint16]; i++)
        {
          attr_map.insert_name<uint16_t>(attr_names[data::AttrMap::attr_index_uint16][i]);
        }
      for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_int16]; i++)
        {
          attr_map.insert_name<int16_t>(attr_names[data::AttrMap::attr_index_int16][i]);
        }
      for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_uint32]; i++)
        {
          attr_map.insert_name<uint32_t>(attr_names[data::AttrMap::attr_index_uint32][i]);
        }
      for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_int32]; i++)
        {
          attr_map.insert_name<int32_t>(attr_names[data::AttrMap::attr_index_int32][i]);
        }

      size_t sendrecvbuf_size = sendrecvbuf.size();
      throw_assert_nomsg(MPI_Bcast(&sendrecvbuf_size, 1, MPI_SIZE_T, root, comm) == MPI_SUCCESS);
      sendrecvbuf.resize(sendrecvbuf_size);
      throw_assert_nomsg(MPI_Bcast(&sendrecvbuf[0], sendrecvbuf_size, MPI_CHAR, root, comm) == MPI_SUCCESS);
      if (rank != (unsigned int)root)
        {
          data::deserialize_data(sendrecvbuf, attr_map);
        }
    }

      
    void read_cell_attribute_selection
    (
     MPI_Comm      comm,
     const string& file_name,
     const string& name_space,
     const set<string>& attr_mask,
     const string& pop_name,
     const CELL_IDX_T& pop_start,
     const std::vector<CELL_IDX_T>&  selection,
     data::NamedAttrMap& attr_values
     )
    {
      herr_t status; 
      unsigned int rank, size;
      throw_assert_nomsg(MPI_Comm_size(comm, (int*)&size) >= 0);
      throw_assert_nomsg(MPI_Comm_rank(comm, (int*)&rank) >= 0);

      vector< tuple<string,AttrKind,vector<CELL_IDX_T>,vector<ATTR_PTR_T> > > attr_info;

      if (rank == 0)
        {
          status = get_cell_attributes_index_ptr (file_name, name_space, pop_name, pop_start, attr_info);
          throw_assert(status == 0,
                       "read_cell_attributes: error in get_cell_attributes_index_ptr");
        }
      // Broadcast the attribute names, types, indices, and pointers
      {
        vector<char> sendbuf; size_t sendbuf_size=0;
        if (rank == 0)
          {
            data::serialize_data(attr_info, sendbuf);
            sendbuf_size = sendbuf.size();
          }

        throw_assert(MPI_Bcast(&sendbuf_size, 1, MPI_SIZE_T, 0, comm) == MPI_SUCCESS,
                     "read_cell_attributes: error in MPI_Bcast");
        sendbuf.resize(sendbuf_size);
        throw_assert(MPI_Bcast(&sendbuf[0], sendbuf_size, MPI_CHAR, 0, comm) == MPI_SUCCESS,
                     "read_cell_attributes: error in MPI_Bcast");
        
        if (rank != 0)
          {
            data::deserialize_data(sendbuf, attr_info);
          }
      }

      size_t selection_size = selection.size();
      int data_color = 2;
      MPI_Comm data_comm;
      // In cases where some ranks do not have any data to read, split
      // the communicator, so that collective operations can be executed
      // only on the ranks that do have data.
      if (selection_size > 0)
        {
          MPI_Comm_split(comm,data_color,0,&data_comm);
        }
      else
        {
          MPI_Comm_split(comm,0,0,&data_comm);
        }
      MPI_Comm_set_errhandler(data_comm, MPI_ERRORS_RETURN);


      if (selection_size > 0)
        {
          // get a file handle and retrieve the MPI info
          hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
          throw_assert(H5Pset_fapl_mpio(fapl, data_comm, MPI_INFO_NULL) >= 0,
                       "read_cell_attribute_selection: error setting MPI driver for file access");
          
          hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, fapl);
          throw_assert(file >= 0,
                       "read_cell_attribute_selection: unable to open file " << file_name);
          
          status = H5Pclose(fapl);
          throw_assert(status == 0,
                       "read_cell_attribute_selection: unable to close file access property list");
          
          for (size_t i=0; i<attr_info.size(); i++)
            {
              vector<ATTR_PTR_T> value_ptr;
              vector<CELL_IDX_T> value_index;
              
              string attr_name  = get<0>(attr_info[i]);
              AttrKind attr_kind = get<1>(attr_info[i]);
              size_t attr_size  = attr_kind.size;
              const vector<CELL_IDX_T>& index  = get<2>(attr_info[i]);
              const vector<ATTR_PTR_T>& ptr  = get<3>(attr_info[i]);
              string attr_path  = hdf5::cell_attribute_path (name_space, pop_name, attr_name);
              
              if ((attr_mask.size() > 0) && (attr_mask.count(attr_name) == 0))
                continue;
              
              switch (attr_kind.type)
                {
                case UIntVal:
                  {
                    if (attr_size == 4)
                      {
                        vector<uint32_t> attr_values_uint32;
                        status = hdf5::read_cell_attribute_selection(data_comm, file, attr_path, pop_start,
                                                                     selection, index, ptr, value_index, value_ptr,
                                                                     attr_values_uint32);
                        attr_values.insert(attr_name, value_index, value_ptr, attr_values_uint32);
                      }
                    else if (attr_size == 2)
                      {
                        vector<uint16_t> attr_values_uint16;
                        status = hdf5::read_cell_attribute_selection(data_comm, file, attr_path, pop_start,
                                                                     selection, index, ptr, value_index, value_ptr,
                                                                     attr_values_uint16);
                        attr_values.insert(attr_name, value_index, value_ptr, attr_values_uint16);
                      }
                    else if (attr_size == 1)
                      {
                        vector<uint8_t> attr_values_uint8;
                        status = hdf5::read_cell_attribute_selection(data_comm, file, attr_path, pop_start,
                                                                     selection, index, ptr, value_index, value_ptr,
                                                                     attr_values_uint8);
                        attr_values.insert(attr_name, value_index, value_ptr, attr_values_uint8);
                      }
                    else
                      {
                        throw runtime_error("Unsupported integer attribute size");
                      };
                  }
                  break;
                case SIntVal:
                  {
                    if (attr_size == 4)
                      {
                        vector<int32_t> attr_values_int32;
                        status = hdf5::read_cell_attribute_selection(data_comm, file, attr_path, pop_start,
                                                                     selection, index, ptr, value_index, value_ptr,
                                                                     attr_values_int32);
                        attr_values.insert(attr_name, value_index, value_ptr, attr_values_int32);
                      }
                    else if (attr_size == 2)
                      {
                        vector<int16_t> attr_values_int16;
                        status = hdf5::read_cell_attribute_selection(data_comm, file, attr_path, pop_start,
                                                                     selection, index, ptr, value_index, value_ptr,
                                                                     attr_values_int16);
                        attr_values.insert(attr_name, value_index, value_ptr, attr_values_int16);
                      }
                    else if (attr_size == 1)
                      {
                        vector<int8_t> attr_values_int8;
                        status = hdf5::read_cell_attribute_selection(data_comm, file, attr_path, pop_start,
                                                                     selection, index, ptr, value_index, value_ptr,
                                                                     attr_values_int8);
                        attr_values.insert(attr_name, value_index, value_ptr, attr_values_int8);
                      }
                    else
                      {
                        throw runtime_error("Unsupported integer attribute size");
                      };
                  }
                  break;
                case FloatVal:
                  {
                    vector<float> attr_values_float;
                    status = hdf5::read_cell_attribute_selection(data_comm, file, attr_path, pop_start,
                                                                 selection, index, ptr, value_index, value_ptr,
                                                                 attr_values_float);
                    attr_values.insert(attr_name, value_index, value_ptr, attr_values_float);
                  }
                  break;
                case EnumVal:
                  {
                    if (attr_size == 1)
                      {
                        vector<uint8_t> attr_values_uint8;
                        status = hdf5::read_cell_attribute_selection(data_comm, file, attr_path, pop_start,
                                                                     selection, index, ptr, value_index, value_ptr,
                                                                     attr_values_uint8);
                        attr_values.insert(attr_name, value_index, value_ptr, attr_values_uint8);
                      }
                    else
                      {
                        throw runtime_error("Unsupported enumerated attribute size");
                      };
                  }
                  break;
                default:
                  throw runtime_error("Unsupported attribute type");
                  break;
                }

            }

          status = H5Fclose(file);
          throw_assert(status == 0,
                       "read_cell_attribute_selection: unable to close file " << file_name);
        }

      MPI_Barrier(data_comm);
      MPI_Barrier(comm);
      throw_assert(MPI_Comm_free(&data_comm) == MPI_SUCCESS,
                   "read_cell_attribute_selection: error in MPI_Comm_free ");
      
    }

    
    void scatter_read_cell_attribute_selection
    (
     MPI_Comm      comm,
     const string& file_name,
     const int     io_size,
     const string& attr_name_space,
     const set<string>& attr_mask,
     const string& pop_name,
     const CELL_IDX_T& pop_start,
     const std::vector<CELL_IDX_T>&  selection,
     data::NamedAttrMap& attr_values
     )
    {
      herr_t status; 
      throw_assert_nomsg(io_size > 0);

      size_t selection_size = selection.size();
      int data_color = 2;
      MPI_Comm data_comm;
      // In cases where some ranks do not have any data to read, split
      // the communicator, so that collective operations can be executed
      // only on the ranks that do have data.
      if (selection_size > 0)
        {
          MPI_Comm_split(comm,data_color,0,&data_comm);
        }
      else
        {
          MPI_Comm_split(comm,0,0,&data_comm);
        }
      MPI_Comm_set_errhandler(data_comm, MPI_ERRORS_RETURN);
      unsigned int data_rank, data_size;
      throw_assert_nomsg(MPI_Comm_size(data_comm, (int*)&data_size) >= 0);
      throw_assert_nomsg(MPI_Comm_rank(data_comm, (int*)&data_rank) >= 0);

      vector<CELL_IDX_T> all_selections;
      if (selection_size > 0)
        {
          map<CELL_IDX_T, rank_t> node_rank_map;
          {
            vector<size_t> sendbuf_selection_size(data_size, selection_size);
            vector<size_t> recvbuf_selection_size(data_size);
            vector<int> recvcounts(data_size, 0);
            vector<int> displs(data_size+1, 0);

            // Each DATA_COMM rank sends its selection to every other DATA_COMM rank
            throw_assert_nomsg(MPI_Allgather(&sendbuf_selection_size[0], 1, MPI_SIZE_T,
                                             &recvbuf_selection_size[0], 1, MPI_SIZE_T, data_comm)
                               == MPI_SUCCESS);
            throw_assert_nomsg(MPI_Barrier(data_comm) == MPI_SUCCESS);

            size_t total_selection_size = 0;
            for (size_t p=0; p<data_size; p++)
              {
                total_selection_size = total_selection_size + recvbuf_selection_size[p];
                displs[p+1] = displs[p] + recvbuf_selection_size[p];
                recvcounts[p] = recvbuf_selection_size[p];
              }

            all_selections.resize(total_selection_size);
            throw_assert_nomsg(MPI_Allgatherv(&selection[0], selection_size, MPI_CELL_IDX_T,
                                              &all_selections[0], &recvcounts[0], &displs[0], MPI_NODE_IDX_T,
                                              data_comm) == MPI_SUCCESS);
            throw_assert_nomsg(MPI_Barrier(data_comm) == MPI_SUCCESS);

            // Construct node rank map based on selection information.
            for (rank_t p=0; p<data_size; p++)
              {
                for (size_t i = displs[p]; i<displs[p+1]; i++)
                  {
                    node_rank_map.insert ( make_pair(all_selections[i], p) );
                  }

              }
            
          }
          
          vector<int> sendcounts(data_size,0), sdispls(data_size,0), recvcounts(data_size,0), rdispls(data_size,0);
          vector<char> sendbuf; 

          vector< size_t > num_attrs;
          num_attrs.resize(data::AttrMap::num_attr_types);
          vector< vector<string> > attr_names;
          attr_names.resize(data::AttrMap::num_attr_types);

          // MPI Communicator for I/O ranks
          MPI_Comm io_comm;
          // MPI group color value used for I/O ranks
          int io_color = 1;
          size_t io_data_size = io_size;
          
          if (io_data_size > data_size)
            io_data_size = data_size;
      
          set<size_t> io_rank_set;
          data::range_sample(data_size, io_data_size, io_rank_set);
          bool is_io_rank = false;
          if (io_rank_set.find(data_rank) != io_rank_set.end())
            is_io_rank = true;

          if (is_io_rank)
            {
              // Am I an I/O rank?
              MPI_Comm_split(data_comm,io_color,data_rank,&io_comm);
              MPI_Comm_set_errhandler(io_comm, MPI_ERRORS_RETURN);

              unsigned int io_rank, io_size;
              throw_assert_nomsg(MPI_Comm_size(io_comm, (int*)&io_size) >= 0);
              throw_assert_nomsg(MPI_Comm_rank(io_comm, (int*)&io_rank) >= 0);
      
              std::vector<CELL_IDX_T> io_selection;
              for (const CELL_IDX_T& s : all_selections)
                {
                  if (s % io_size == io_rank)
                    {
                      io_selection.push_back(s);
                    }
                }
              map <rank_t, data::AttrMap > rank_attr_map;
              {
                data::NamedAttrMap  attr_values;
                read_cell_attribute_selection(io_comm, file_name, attr_name_space, attr_mask, pop_name, pop_start,
                                              io_selection, attr_values);
                data::append_rank_attr_map(attr_values, node_rank_map, rank_attr_map);
                attr_values.num_attrs(num_attrs);
                attr_values.attr_names(attr_names);
              }
              
              data::serialize_rank_attr_map (data_size, data_rank, rank_attr_map, sendcounts, sendbuf, sdispls);
            }
          else
            {
              
              MPI_Comm_split(data_comm,0,data_rank,&io_comm);
            }
          MPI_Barrier(io_comm);
          MPI_Barrier(data_comm);
          throw_assert_nomsg(MPI_Comm_free(&io_comm) == MPI_SUCCESS);

          vector<size_t> num_attrs_bcast(num_attrs.size());
          for (size_t i=0; i<num_attrs.size(); i++)
            {
              num_attrs_bcast[i] = num_attrs[i];
            }
          // 4. Broadcast the number of attributes of each type to all ranks
          throw_assert_nomsg(MPI_Bcast(&num_attrs_bcast[0], num_attrs_bcast.size(), MPI_SIZE_T, 0, data_comm) >= 0);
          for (size_t i=0; i<num_attrs.size(); i++)
            {
              num_attrs[i] = num_attrs_bcast[i];
            }
          
          // 5. Broadcast the names of each attributes of each type to all data ranks
          {
            vector<char> sendbuf; size_t sendbuf_size=0;
            if (data_rank == 0)
              {
                data::serialize_data(attr_names, sendbuf);
                sendbuf_size = sendbuf.size();
              }

            throw_assert_nomsg(MPI_Bcast(&sendbuf_size, 1, MPI_SIZE_T, 0, data_comm) >= 0);
            sendbuf.resize(sendbuf_size);
            throw_assert_nomsg(MPI_Bcast(&sendbuf[0], sendbuf_size, MPI_CHAR, 0, data_comm) >= 0);
            
            if (data_rank != 0)
              {
                data::deserialize_data(sendbuf, attr_names);
              }
          }
      
          for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_float]; i++)
            {
              attr_values.insert_name<float>(attr_names[data::AttrMap::attr_index_float][i]);
            }
          for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_uint8]; i++)
            {
              attr_values.insert_name<uint8_t>(attr_names[data::AttrMap::attr_index_uint8][i]);
            }
          for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_int8]; i++)
            {
              attr_values.insert_name<int8_t>(attr_names[data::AttrMap::attr_index_int8][i]);
            }
          for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_uint16]; i++)
            {
              attr_values.insert_name<uint16_t>(attr_names[data::AttrMap::attr_index_uint16][i]);
            }
          for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_int16]; i++)
            {
              attr_values.insert_name<int16_t>(attr_names[data::AttrMap::attr_index_int16][i]);
            }
          for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_uint32]; i++)
            {
              attr_values.insert_name<uint32_t>(attr_names[data::AttrMap::attr_index_uint32][i]);
            }
          for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_int32]; i++)
            {
              attr_values.insert_name<int32_t>(attr_names[data::AttrMap::attr_index_int32][i]);
            }
          
          // 6. Each DATA_COMM rank sends an attribute set size to
          //    every other DATA_COMM rank (non IO_COMM ranks pass zero)
          throw_assert_nomsg(MPI_Alltoall(&sendcounts[0], 1, MPI_INT,
                                          &recvcounts[0], 1, MPI_INT, data_comm) >= 0);
          MPI_Barrier(data_comm);
          
          // 7. Each DATA_COMM rank accumulates the vector sizes and allocates
          //    a receive buffer, recvcounts, and rdispls
          size_t recvbuf_size;
          vector<char> recvbuf;
          
          recvbuf_size = recvcounts[0];
          for (rank_t p = 1; p < data_size; ++p)
            {
              rdispls[p] = rdispls[p-1] + recvcounts[p-1];
              recvbuf_size += recvcounts[p];
            }
          if (recvbuf_size > 0)
            recvbuf.resize(recvbuf_size);
          
          // 8. Each DATA_COMM rank participates in the MPI_Alltoallv
          throw_assert_nomsg(mpi::alltoallv_vector<char>(data_comm, MPI_CHAR, sendcounts, sdispls, sendbuf,
                                                         recvcounts, rdispls, recvbuf) >= 0);

          sendbuf.clear();
          MPI_Barrier(data_comm);
          
          if (recvbuf.size() > 0)
            {
              data::deserialize_rank_attr_map (data_size, recvbuf, recvcounts, rdispls, attr_values);
            }
          recvbuf.clear();
        }

      MPI_Barrier(data_comm);
      MPI_Barrier(comm);
      throw_assert(MPI_Comm_free(&data_comm) == MPI_SUCCESS,
                   "scatter_read_cell_attribute_selection: error in MPI_Comm_free ");
      

    }

    
  }
  
}
