// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file node_attributes.cc
///
///  Routines for manipulation of scalar and vector attributes associated with a graph node.
///
///  Copyright (C) 2016-2020 Project NeuroH5.
//==============================================================================

#include "neuroh5_types.hh"
#include "path_names.hh"
#include "read_template.hh"
#include "write_template.hh"
#include "hdf5_node_attributes.hh"
#include "dataset_num_elements.hh"
#include "exists_dataset.hh"
#include "create_group.hh"
#include "attr_map.hh"
#include "infer_datatype.hh"
#include "alltoallv_template.hh"
#include "serialize_data.hh"
#include "serialize_cell_attributes.hh"
#include "throw_assert.hh"

#include <hdf5.h>
#include <mpi.h>

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

using namespace std;
using namespace neuroh5;

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
     )
    {
      ptr_size = hdf5::dataset_num_elements(loc, path+"/"+hdf5::ATTR_PTR);
      index_size = hdf5::dataset_num_elements(loc, path+"/"+hdf5::NODE_INDEX);
      value_size = hdf5::dataset_num_elements(loc, path+"/"+hdf5::ATTR_VAL);
    }

    void create_node_attribute_datasets
    (
     const hid_t&   file,
     const string&  attr_namespace,
     const string&  attr_name,
     const hid_t&   ftype,
     const size_t   chunk_size,
     const size_t   value_chunk_size
     )
    {
      herr_t status;
      hsize_t maxdims[1] = {H5S_UNLIMITED};
      hsize_t cdims[1]   = {chunk_size}; /* chunking dimensions */		
      hsize_t initial_size = 0;
    
      hid_t plist  = H5Pcreate (H5P_DATASET_CREATE);
      status = H5Pset_chunk(plist, 1, cdims);
      throw_assert_nomsg(status == 0);
#ifdef H5_HAS_PARALLEL_DEFLATE
      status = H5Pset_deflate(plist, 6);
      throw_assert_nomsg(status == 0);
#endif

      hsize_t value_cdims[1]   = {value_chunk_size}; /* chunking dimensions for value dataset */		
      hid_t value_plist = H5Pcreate (H5P_DATASET_CREATE);
      status = H5Pset_chunk(value_plist, 1, value_cdims);
      throw_assert_nomsg(status == 0);
#ifdef H5_HAS_PARALLEL_DEFLATE
      status = H5Pset_deflate(value_plist, 6);
      throw_assert_nomsg(status == 0);
#endif
      
      hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
      throw_assert_nomsg(lcpl >= 0);
      throw_assert_nomsg(H5Pset_create_intermediate_group(lcpl, 1) >= 0);
    
      if (!(hdf5::exists_dataset (file, ("/" + hdf5::NODES)) > 0))
        {
          hdf5::create_group(file, ("/" + hdf5::NODES).c_str());
        }

      string attr_prefix = hdf5::node_attribute_prefix(attr_namespace);
      if (!(hdf5::exists_dataset (file, attr_prefix) > 0))
        {
          hdf5::create_group(file, attr_prefix);
        }

      string attr_path = hdf5::node_attribute_path(attr_namespace, attr_name);

      hid_t mspace = H5Screate_simple(1, &initial_size, maxdims);
      throw_assert_nomsg(mspace >= 0);
      hid_t dset = H5Dcreate2(file, (attr_path + "/" + hdf5::NODE_INDEX).c_str(), NODE_IDX_H5_FILE_T,
                              mspace, lcpl, plist, H5P_DEFAULT);
      throw_assert_nomsg(H5Dclose(dset) >= 0);
      throw_assert_nomsg(H5Sclose(mspace) >= 0);

      mspace = H5Screate_simple(1, &initial_size, maxdims);
      throw_assert_nomsg(mspace >= 0);
      dset = H5Dcreate2(file, (attr_path + "/" + hdf5::ATTR_PTR).c_str(), ATTR_PTR_H5_FILE_T,
                        mspace, lcpl, plist, H5P_DEFAULT);
      throw_assert_nomsg(H5Dclose(dset) >= 0);
      throw_assert_nomsg(H5Sclose(mspace) >= 0);
    
      mspace = H5Screate_simple(1, &initial_size, maxdims);
      dset = H5Dcreate2(file, (attr_path + "/" + hdf5::ATTR_VAL).c_str(), ftype, mspace,
                        lcpl, value_plist, H5P_DEFAULT);
      throw_assert_nomsg(H5Dclose(dset) >= 0);
      throw_assert_nomsg(H5Sclose(mspace) >= 0);
    
      throw_assert_nomsg(H5Pclose(lcpl) >= 0);
    
      status = H5Pclose(plist);
      throw_assert_nomsg(status == 0);
      status = H5Pclose(value_plist);
      throw_assert_nomsg(status == 0);
    
    }
  }
  
  namespace graph
  {
    // Callback for H5Literate
    static herr_t node_attribute_cb
    (
     hid_t             grp,
     const char*       name,
     const H5L_info_t* info,
     void*             op_data
     )
    {
      string value_path = string(name) + "/" + hdf5::ATTR_VAL;
      hid_t dset = H5Dopen2(grp, value_path.c_str(), H5P_DEFAULT);
      if (dset < 0) // skip the link, if this is not a dataset
        {
          return 0;
        }
    
      hid_t ftype = H5Dget_type(dset);
      throw_assert_nomsg(ftype >= 0);
    
      vector< pair<string,hid_t> >* ptr =
        (vector< pair<string,hid_t> >*) op_data;
      ptr->push_back(make_pair(name, ftype));

      throw_assert_nomsg(H5Dclose(dset) >= 0);
    
      return 0;
    }
  
    herr_t get_node_attributes
    (
     const string&                 file_name,
     const string&                 name_space,
     vector< pair<string,hid_t> >& out_attributes
     )
    {
      hid_t in_file;
      herr_t ierr;
    
      in_file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      throw_assert_nomsg(in_file >= 0);
      out_attributes.clear();
    
      string path = hdf5::node_attribute_prefix(name_space);

      hid_t grp = H5Gopen2(in_file, path.c_str(), H5P_DEFAULT);
      if (grp >= 0)
        {
    
          hsize_t idx = 0;
          ierr = H5Literate(grp, H5_INDEX_NAME, H5_ITER_NATIVE, &idx,
                            &node_attribute_cb, (void*) &out_attributes);
          
          throw_assert_nomsg(H5Gclose(grp) >= 0);
        }
      
      ierr = H5Fclose(in_file);
    
      return ierr;
    }
  

    herr_t num_node_attributes
    (
     const vector< pair<string,hid_t> >& attributes,
     vector <size_t> &num_attrs
     )
    {
      herr_t ierr = 0;
      num_attrs.resize(data::AttrVal::num_attr_types);
      for (size_t i = 0; i < attributes.size(); i++)
        {
          hid_t attr_h5type = attributes[i].second;
          size_t attr_size = H5Tget_size(attr_h5type);
          switch (H5Tget_class(attr_h5type))
            {
            case H5T_INTEGER:
              if (attr_size == 4)
                {
                  if (H5Tget_sign( attr_h5type ) == H5T_SGN_NONE)
                    {
                      num_attrs[data::AttrVal::attr_index_uint32]++;
                    }
                  else
                    {
                      num_attrs[data::AttrVal::attr_index_int32]++;
                    }
                }
              else if (attr_size == 2)
                {
                  if (H5Tget_sign( attr_h5type ) == H5T_SGN_NONE)
                    {
                      num_attrs[data::AttrVal::attr_index_uint16]++;
                    }
                  else
                    {
                      num_attrs[data::AttrVal::attr_index_int16]++;
                    }
                }
              else if (attr_size == 1)
                {
                  if (H5Tget_sign( attr_h5type ) == H5T_SGN_NONE)
                    {
                      num_attrs[data::AttrVal::attr_index_uint8]++;
                    }
                  else
                    {
                      num_attrs[data::AttrVal::attr_index_int8]++;
                    }
                }
              else
                {
                  throw runtime_error("Unsupported integer attribute size");
                };
              break;
            case H5T_FLOAT:
              num_attrs[data::AttrVal::attr_index_float]++;
              break;
            case H5T_ENUM:
              if (attr_size == 1)
                {
                  if (H5Tget_sign( attr_h5type ) == H5T_SGN_NONE)
                    {
                      num_attrs[data::AttrVal::attr_index_uint8]++;
                    }
                  else
                    {
                      num_attrs[data::AttrVal::attr_index_int8]++;
                    }
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

          throw_assert_nomsg(ierr >= 0);
        }

      return ierr;
    }
  
    void read_node_attributes
    (
     MPI_Comm      comm,
     const string& file_name,
     const string& name_space,
     data::NamedAttrMap& attr_values,
     size_t offset = 0,
     size_t numitems = 0
     )
    {
      herr_t status; 

      unsigned int rank, size;
      throw_assert_nomsg(MPI_Comm_size(comm, (int*)&size) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Comm_rank(comm, (int*)&rank) == MPI_SUCCESS);

      vector< pair<string,hid_t> > attr_info;
    
      status = get_node_attributes (file_name, name_space, attr_info);

      // get a file handle and retrieve the MPI info
      hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
      throw_assert_nomsg(H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL) >= 0);
      hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, fapl);
      throw_assert_nomsg(file >= 0);

      for (size_t i=0; i<attr_info.size(); i++)
        {
          vector<NODE_IDX_T>  index;
          vector<ATTR_PTR_T>  ptr;

          string attr_name  = attr_info[i].first;
          hid_t attr_h5type = attr_info[i].second;
          size_t attr_size  = H5Tget_size(attr_h5type);
          string attr_path  = hdf5::node_attribute_path (name_space, attr_name);

          switch (H5Tget_class(attr_h5type))
            {
            case H5T_INTEGER:
              if (attr_size == 4)
                {
                  if (H5Tget_sign( attr_h5type ) == H5T_SGN_NONE)
                    {
                      vector<uint32_t> attr_values_uint32;
                      status = hdf5::read_node_attribute(comm, file, attr_path,
                                                         index, ptr, attr_values_uint32,
                                                         offset, numitems);
                      attr_values.insert(attr_name, index, ptr, attr_values_uint32);
                    }
                  else
                    {
                      vector<int32_t> attr_values_int32;
                      status = hdf5::read_node_attribute(comm, file, attr_path,
                                                         index, ptr, attr_values_int32,
                                                         offset, numitems);
                      attr_values.insert(attr_name, index, ptr, attr_values_int32);
                    }
                }
              else if (attr_size == 2)
                {
                  if (H5Tget_sign( attr_h5type ) == H5T_SGN_NONE)
                    {
                      vector<uint16_t> attr_values_uint16;
                      status = hdf5::read_node_attribute(comm, file, attr_path,
                                                         index, ptr, attr_values_uint16,
                                                         offset, numitems);
                      attr_values.insert(attr_name, index, ptr, attr_values_uint16);
                    }
                  else
                    {
                      vector<int16_t> attr_values_int16;
                      status = hdf5::read_node_attribute(comm, file, attr_path,
                                                         index, ptr, attr_values_int16,
                                                         offset, numitems);
                      attr_values.insert(attr_name, index, ptr, attr_values_int16);
                    }
                }
              else if (attr_size == 1)
                {
                  if (H5Tget_sign( attr_h5type ) == H5T_SGN_NONE)
                    {
                      vector<uint8_t> attr_values_uint8;
                      status = hdf5::read_node_attribute(comm, file, attr_path,
                                                         index, ptr, attr_values_uint8,
                                                         offset, numitems);
                      attr_values.insert(attr_name, index, ptr, attr_values_uint8);
                    }
                  else
                    {
                      vector<int8_t> attr_values_int8;
                      status = hdf5::read_node_attribute(comm, file, attr_path,
                                                         index, ptr, attr_values_int8,
                                                         offset, numitems);
                      attr_values.insert(attr_name, index, ptr, attr_values_int8);
                    }
                }
              else
                {
                  throw runtime_error("Unsupported integer attribute size");
                };
              break;
            case H5T_FLOAT:
              {
                vector<float> attr_values_float;
                status = hdf5::read_node_attribute(comm, file, attr_path,
                                                   index, ptr, attr_values_float,
                                                   offset, numitems);
                attr_values.insert(attr_name, index, ptr, attr_values_float);
              }
              break;
            case H5T_ENUM:
              if (attr_size == 1)
                {
                  if (H5Tget_sign( attr_h5type ) == H5T_SGN_NONE)
                    {
                      vector<uint8_t> attr_values_uint8;
                      status = hdf5::read_node_attribute(comm, file, attr_path,
                                                         index, ptr, attr_values_uint8,
                                                         offset, numitems);
                      attr_values.insert(attr_name, index, ptr, attr_values_uint8);
                    }
                  else
                    {
                      vector<int8_t> attr_values_int8;
                      status = hdf5::read_node_attribute(comm, file, attr_path,
                                                         index, ptr, attr_values_int8,
                                                         offset, numitems);
                      attr_values.insert(attr_name, index, ptr, attr_values_int8);
                    }
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

      status = H5Fclose(file);
      throw_assert_nomsg(status == 0);
      status = H5Pclose(fapl);
      throw_assert_nomsg(status == 0);
    }


    void append_rank_attr_map
    (
     const map<NODE_IDX_T,size_t> &node_rank_map,
     const data::NamedAttrMap   &attr_values,
     map <rank_t, data::AttrMap> &rank_attr_map)
    {
      const vector<map< NODE_IDX_T, vector<float> > > &all_float_values     = attr_values.attr_maps<float>();
      const vector<map< NODE_IDX_T, vector<int8_t> > > &all_int8_values     = attr_values.attr_maps<int8_t>();
      const vector<map< NODE_IDX_T, vector<uint8_t> > > &all_uint8_values   = attr_values.attr_maps<uint8_t>();
      const vector<map< NODE_IDX_T, vector<uint16_t> > > &all_uint16_values = attr_values.attr_maps<uint16_t>();
      const vector<map< NODE_IDX_T, vector<int16_t> > > &all_int16_values   = attr_values.attr_maps<int16_t>();
      const vector<map< NODE_IDX_T, vector<uint32_t> > > &all_uint32_values = attr_values.attr_maps<uint32_t>();
      const vector<map< NODE_IDX_T, vector<int32_t> > > &all_int32_values   = attr_values.attr_maps<int32_t>();
    
      for (size_t i=0; i<all_float_values.size(); i++)
        {
          const map< NODE_IDX_T, vector<float> > &float_values = all_float_values[i];
          for (auto const& element : float_values)
            {
              const NODE_IDX_T index = element.first;
              const vector<float> &v = element.second;
              auto it = node_rank_map.find(index);
              if(it == node_rank_map.end())
                {
                  printf("index %u not in node rank map\n", index);
                }
              throw_assert_nomsg(it != node_rank_map.end());
              size_t dst_rank = it->second;
              data::AttrMap &attr_map = rank_attr_map[dst_rank];
              attr_map.insert(i, index, v);
            }
        }

      for (size_t i=0; i<all_uint8_values.size(); i++)
        {
          const map< NODE_IDX_T, vector<uint8_t> > &uint8_values = all_uint8_values[i];
          for (auto const& element : uint8_values)
            {
              const NODE_IDX_T index = element.first;
              const vector<uint8_t> &v = element.second;
              auto it = node_rank_map.find(index);
              throw_assert_nomsg(it != node_rank_map.end());
              size_t dst_rank = it->second;
              data::AttrMap &attr_map = rank_attr_map[dst_rank];
              attr_map.insert(i, index, v);
            }
        }

      for (size_t i=0; i<all_int8_values.size(); i++)
        {
          const map< NODE_IDX_T, vector<int8_t> > &int8_values = all_int8_values[i];
          for (auto const& element : int8_values)
            {
              const NODE_IDX_T index = element.first;
              const vector<int8_t> &v = element.second;
              auto it = node_rank_map.find(index);
              throw_assert_nomsg(it != node_rank_map.end());
              size_t dst_rank = it->second;
              data::AttrMap &attr_map = rank_attr_map[dst_rank];
              attr_map.insert(i, index, v);
            }
        }

      for (size_t i=0; i<all_uint16_values.size(); i++)
        {
          const map< NODE_IDX_T, vector<uint16_t> > &uint16_values = all_uint16_values[i];
          for (auto const& element : uint16_values)
            {
              const NODE_IDX_T index = element.first;
              const vector<uint16_t> &v = element.second;
              auto it = node_rank_map.find(index);
              throw_assert_nomsg(it != node_rank_map.end());
              size_t dst_rank = it->second;
              data::AttrMap &attr_map = rank_attr_map[dst_rank];
              attr_map.insert(i, index, v);
            }
        }

      for (size_t i=0; i<all_int16_values.size(); i++)
        {
          const map< NODE_IDX_T, vector<int16_t> > &int16_values = all_int16_values[i];
          for (auto const& element : int16_values)
            {
              const NODE_IDX_T index = element.first;
              const vector<int16_t> &v = element.second;
              auto it = node_rank_map.find(index);
              throw_assert_nomsg(it != node_rank_map.end());
              size_t dst_rank = it->second;
              data::AttrMap &attr_map = rank_attr_map[dst_rank];
              attr_map.insert(i, index, v);
            }
        }

      for (size_t i=0; i<all_uint32_values.size(); i++)
        {
          const map< NODE_IDX_T, vector<uint32_t> > &uint32_values = all_uint32_values[i];
          for (auto const& element : uint32_values)
            {
              const NODE_IDX_T index = element.first;
              const vector<uint32_t> &v = element.second;
              auto it = node_rank_map.find(index);
              throw_assert_nomsg(it != node_rank_map.end());
              size_t dst_rank = it->second;
              data::AttrMap &attr_map = rank_attr_map[dst_rank];
              attr_map.insert(i, index, v);
            }
        }

      for (size_t i=0; i<all_int32_values.size(); i++)
        {
          const map< NODE_IDX_T, vector<int32_t> > &int32_values = all_int32_values[i];
          for (auto const& element : int32_values)
            {
              const NODE_IDX_T index = element.first;
              const vector<int32_t> &v = element.second;
              auto it = node_rank_map.find(index);
              throw_assert_nomsg(it != node_rank_map.end());
              size_t dst_rank = it->second;
              data::AttrMap &attr_map = rank_attr_map[dst_rank];
              attr_map.insert(i, index, v);
            }
        }

    }


    int scatter_read_node_attributes
    (
     MPI_Comm                      all_comm,
     const string                 &file_name,
     const int                     io_size,
     const string                 &attr_name_space,
     // A vector that maps nodes to compute ranks
     const map<NODE_IDX_T,size_t> &node_rank_map,
     const NODE_IDX_T              pop_start,
     data::NamedAttrMap           &attr_map,
     // if positive, these arguments specify offset and number of entries to read
     // from the entries available to the current rank
     size_t offset   = 0,
     size_t numitems = 0
     )
    {
      int srank, ssize; size_t rank, size;
      throw_assert_nomsg(MPI_Comm_size(all_comm, &ssize) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Comm_rank(all_comm, &srank) == MPI_SUCCESS);
      throw_assert_nomsg(ssize > 0);
      throw_assert_nomsg(srank >= 0);
      size = ssize;
      rank = srank;

      vector< size_t > num_attrs;
      num_attrs.resize(data::AttrMap::num_attr_types);
      vector< vector<string> > attr_names;
      attr_names.resize(data::AttrMap::num_attr_types);

      // MPI Communicator for I/O ranks
      MPI_Comm io_comm;
      // MPI group color value used for I/O ranks
      int io_color = 1;

      throw_assert_nomsg(io_size > 0);
    
      vector<char> sendbuf; 
      vector<int> sendcounts, sdispls, recvcounts, rdispls;

      sendcounts.resize(size,0);
      sdispls.resize(size,0);
      recvcounts.resize(size,0);
      rdispls.resize(size,0);
        
      if (srank < io_size)
        {
          // Am I an I/O rank?
          MPI_Comm_split(all_comm,io_color,rank,&io_comm);
          MPI_Comm_set_errhandler(io_comm, MPI_ERRORS_RETURN);

          map <rank_t, data::AttrMap > rank_attr_map;
          {
            data::NamedAttrMap  attr_values;
            read_node_attributes(io_comm, file_name, attr_name_space, attr_values,
                                 offset, numitems);
            append_rank_attr_map(node_rank_map, attr_values, rank_attr_map);
            attr_values.num_attrs(num_attrs);
            attr_values.attr_names(attr_names);
          }

          data::serialize_rank_attr_map (size, rank, rank_attr_map, sendcounts, sendbuf, sdispls);
        }
      else
        {
          MPI_Comm_split(all_comm,0,rank,&io_comm);
        }
      MPI_Barrier(all_comm);
    
      vector<size_t> num_attrs_bcast(num_attrs.size());
      for (size_t i=0; i<num_attrs.size(); i++)
        {
          num_attrs_bcast[i] = num_attrs[i];
        }
      // 4. Broadcast the number of attributes of each type to all ranks
      throw_assert_nomsg(MPI_Bcast(&num_attrs_bcast[0], num_attrs_bcast.size(), MPI_SIZE_T, 0, all_comm) == MPI_SUCCESS);
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

        throw_assert_nomsg(MPI_Bcast(&sendbuf_size, 1, MPI_SIZE_T, 0, all_comm) == MPI_SUCCESS);
        sendbuf.resize(sendbuf_size);
        throw_assert_nomsg(MPI_Bcast(&sendbuf[0], sendbuf_size, MPI_CHAR, 0, all_comm) == MPI_SUCCESS);
        
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
                          &recvcounts[0], 1, MPI_INT, all_comm) == MPI_SUCCESS);
    
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
      if (recvbuf_size > 0) recvbuf.resize(recvbuf_size);
    
      // 8. Each ALL_COMM rank participates in the MPI_Alltoallv
      throw_assert_nomsg(mpi::alltoallv_vector<char>(all_comm, MPI_CHAR, sendcounts, sdispls, sendbuf,
                                         recvcounts, rdispls, recvbuf) >= 0);
    
      sendbuf.clear();

      data::deserialize_rank_attr_map (size, recvbuf, recvcounts, rdispls, attr_map);
      recvbuf.clear();
      
      throw_assert_nomsg(MPI_Comm_free(&io_comm) == MPI_SUCCESS);

      return 0;
    }
  }
}
