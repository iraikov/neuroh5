// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file cell_attributes.cc
///
///  Routines for manipulation of scalar and vector attributes associated with a cell id.
///
///  Copyright (C) 2016-2017 Project Neurograph.
//==============================================================================

#include "neuroh5_types.hh"
#include "pack_tree.hh"
#include "bcast_string_vector.hh"
#include "path_names.hh"
#include "read_template.hh"
#include "write_template.hh"
#include "hdf5_cell_attributes.hh"
#include "dataset_num_elements.hh"
#include "create_group.hh"
#include "attr_map.hh"
#include "infer_datatype.hh"

#include <hdf5.h>
#include <mpi.h>

#include <cassert>
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
    void size_cell_attributes
    (
     MPI_Comm         comm,
     hid_t            loc,
     const string&    path,
     hsize_t&         ptr_size,
     hsize_t&         index_size,
     hsize_t&         value_size
     )
    {
      ptr_size   = hdf5::dataset_num_elements(comm, loc, path + "/" + hdf5::ATTR_PTR);
      index_size = hdf5::dataset_num_elements(comm, loc, path + "/" + hdf5::CELL_INDEX);
      value_size = hdf5::dataset_num_elements(comm, loc, path + "/" + hdf5::ATTR_VAL);
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
      assert(in_file >= 0);
      out_name_spaces.clear();
    
      string path = "/" + hdf5::POPULATIONS + "/" + pop_name;

      hid_t grp = H5Gopen2(in_file, path.c_str(), H5P_DEFAULT);
      assert(grp >= 0);
    
      hsize_t idx = 0;
      ierr = H5Literate(grp, H5_INDEX_NAME, H5_ITER_NATIVE, &idx,
                        &name_space_iterate_cb, (void*) &out_name_spaces);
    
      assert(H5Gclose(grp) >= 0);
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
      string value_path = string(name) + "/" + hdf5::ATTR_VAL;
      hid_t dset = H5Dopen2(grp, value_path.c_str(), H5P_DEFAULT);
      if (dset < 0) // skip the link, if this is not a dataset
        {
          return 0;
        }
    
      hid_t ftype = H5Dget_type(dset);
      assert(ftype >= 0);
    
      vector< pair<string,hid_t> >* ptr =
        (vector< pair<string,hid_t> >*) op_data;
      ptr->push_back(make_pair(name, ftype));

      assert(H5Dclose(dset) >= 0);
    
      return 0;
    }

  
    herr_t get_cell_attributes
    (
     const string&                 file_name,
     const string&                 name_space,
     const string&                 pop_name,
     vector< pair<string,hid_t> >& out_attributes
     )
    {
      hid_t in_file;
      herr_t ierr;
    
      in_file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      assert(in_file >= 0);
      out_attributes.clear();
    
      string path = hdf5::cell_attribute_prefix(name_space, pop_name);

      hid_t grp = H5Gopen2(in_file, path.c_str(), H5P_DEFAULT);
      assert(grp >= 0);
    
      hsize_t idx = 0;
      ierr = H5Literate(grp, H5_INDEX_NAME, H5_ITER_NATIVE, &idx,
                        &cell_attribute_cb, (void*) &out_attributes);
    
      assert(H5Gclose(grp) >= 0);
      ierr = H5Fclose(in_file);
    
      return ierr;
    }
  

    herr_t num_cell_attributes
    (
     const vector< pair<string,hid_t> >& attributes,
     vector <uint32_t> &num_attrs
     )
    {
      herr_t ierr = 0;
      num_attrs.resize(AttrMap::num_attr_types);
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
                      num_attrs[AttrMap::attr_index_uint32]++;
                    }
                  else
                    {
                      num_attrs[AttrMap::attr_index_int32]++;
                    }
                }
              else if (attr_size == 2)
                {
                  if (H5Tget_sign( attr_h5type ) == H5T_SGN_NONE)
                    {
                      num_attrs[AttrMap::attr_index_uint16]++;
                    }
                  else
                    {
                      num_attrs[AttrMap::attr_index_int16]++;
                    }
                }
              else if (attr_size == 1)
                {
                  if (H5Tget_sign( attr_h5type ) == H5T_SGN_NONE)
                    {
                      num_attrs[AttrMap::attr_index_uint8]++;
                    }
                  else
                    {
                      num_attrs[AttrMap::attr_index_int8]++;
                    }
                }
              else
                {
                  throw runtime_error("Unsupported integer attribute size");
                };
              break;
            case H5T_FLOAT:
              num_attrs[AttrMap::attr_index_float]++;
              break;
            case H5T_ENUM:
              if (attr_size == 1)
                {
                  num_attrs[AttrMap::attr_index_uint8]++;
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

          assert(ierr >= 0);
        }

      return ierr;
    }


    void append_rank_attr_map
    (
     const map<CELL_IDX_T, rank_t> &node_rank_map,
     const data::NamedAttrMap   &attr_values,
     map <size_t, data::AttrMap> &rank_attr_map)
    {
      const vector<map< CELL_IDX_T, vector<float> > > &all_float_values     = attr_values.attr_maps<float>();
      const vector<map< CELL_IDX_T, vector<int8_t> > > &all_int8_values     = attr_values.attr_maps<int8_t>();
      const vector<map< CELL_IDX_T, vector<uint8_t> > > &all_uint8_values   = attr_values.attr_maps<uint8_t>();
      const vector<map< CELL_IDX_T, vector<uint16_t> > > &all_uint16_values = attr_values.attr_maps<uint16_t>();
      const vector<map< CELL_IDX_T, vector<int16_t> > > &all_int16_values   = attr_values.attr_maps<int16_t>();
      const vector<map< CELL_IDX_T, vector<uint32_t> > > &all_uint32_values = attr_values.attr_maps<uint32_t>();
      const vector<map< CELL_IDX_T, vector<int32_t> > > &all_int32_values   = attr_values.attr_maps<int32_t>();
    
      for (size_t i=0; i<all_float_values.size(); i++)
        {
          const map< CELL_IDX_T, vector<float> > &float_values = all_float_values[i];
          for (auto const& element : float_values)
            {
              const CELL_IDX_T index = element.first;
              const vector<float> &v = element.second;
              auto it = node_rank_map.find(index);
              if(it == node_rank_map.end())
                {
                  printf("index %u not in node rank map\n", index);
                }
              assert(it != node_rank_map.end());
              size_t dst_rank = it->second;
              data::AttrMap &attr_map = rank_attr_map[dst_rank];
              attr_map.insert(i, index, v);
            }
        }

      for (size_t i=0; i<all_uint8_values.size(); i++)
        {
          const map< CELL_IDX_T, vector<uint8_t> > &uint8_values = all_uint8_values[i];
          for (auto const& element : uint8_values)
            {
              const CELL_IDX_T index = element.first;
              const vector<uint8_t> &v = element.second;
              auto it = node_rank_map.find(index);
              assert(it != node_rank_map.end());
              size_t dst_rank = it->second;
              data::AttrMap &attr_map = rank_attr_map[dst_rank];
              attr_map.insert(i, index, v);
            }
        }

      for (size_t i=0; i<all_int8_values.size(); i++)
        {
          const map< CELL_IDX_T, vector<int8_t> > &int8_values = all_int8_values[i];
          for (auto const& element : int8_values)
            {
              const CELL_IDX_T index = element.first;
              const vector<int8_t> &v = element.second;
              auto it = node_rank_map.find(index);
              assert(it != node_rank_map.end());
              size_t dst_rank = it->second;
              data::AttrMap &attr_map = rank_attr_map[dst_rank];
              attr_map.insert(i, index, v);
            }
        }

      for (size_t i=0; i<all_uint16_values.size(); i++)
        {
          const map< CELL_IDX_T, vector<uint16_t> > &uint16_values = all_uint16_values[i];
          for (auto const& element : uint16_values)
            {
              const CELL_IDX_T index = element.first;
              const vector<uint16_t> &v = element.second;
              auto it = node_rank_map.find(index);
              assert(it != node_rank_map.end());
              size_t dst_rank = it->second;
              data::AttrMap &attr_map = rank_attr_map[dst_rank];
              attr_map.insert(i, index, v);
            }
        }

      for (size_t i=0; i<all_int16_values.size(); i++)
        {
          const map< CELL_IDX_T, vector<int16_t> > &int16_values = all_int16_values[i];
          for (auto const& element : int16_values)
            {
              const CELL_IDX_T index = element.first;
              const vector<int16_t> &v = element.second;
              auto it = node_rank_map.find(index);
              assert(it != node_rank_map.end());
              size_t dst_rank = it->second;
              data::AttrMap &attr_map = rank_attr_map[dst_rank];
              attr_map.insert(i, index, v);
            }
        }

      for (size_t i=0; i<all_uint32_values.size(); i++)
        {
          const map< CELL_IDX_T, vector<uint32_t> > &uint32_values = all_uint32_values[i];
          for (auto const& element : uint32_values)
            {
              const CELL_IDX_T index = element.first;
              const vector<uint32_t> &v = element.second;
              auto it = node_rank_map.find(index);
              assert(it != node_rank_map.end());
              size_t dst_rank = it->second;
              data::AttrMap &attr_map = rank_attr_map[dst_rank];
              attr_map.insert(i, index, v);
            }
        }

      for (size_t i=0; i<all_int32_values.size(); i++)
        {
          const map< CELL_IDX_T, vector<int32_t> > &int32_values = all_int32_values[i];
          for (auto const& element : int32_values)
            {
              const CELL_IDX_T index = element.first;
              const vector<int32_t> &v = element.second;
              auto it = node_rank_map.find(index);
              assert(it != node_rank_map.end());
              size_t dst_rank = it->second;
              data::AttrMap &attr_map = rank_attr_map[dst_rank];
              attr_map.insert(i, index, v);
            }
        }

    }

    void create_cell_attribute_datasets
    (
     const hid_t&   file,
     const string&  attr_namespace,
     const string&  pop_name,
     const string&  attr_name,
     const hid_t&   ftype,
     CellIndex index_type = IndexOwner,
     CellPtr ptr_type = PtrOwner,
     const string   shared_ptr_name = "",
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
      assert(status == 0);

      hsize_t value_cdims[1]   = {value_chunk_size}; /* chunking dimensions for value dataset */		
      hid_t value_plist = H5Pcreate (H5P_DATASET_CREATE);
      status = H5Pset_chunk(value_plist, 1, value_cdims);
      assert(status == 0);
    
      hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
      assert(lcpl >= 0);
      assert(H5Pset_create_intermediate_group(lcpl, 1) >= 0);
    
      if (!(H5Lexists (file, ("/" + hdf5::POPULATIONS).c_str(), H5P_DEFAULT) > 0))
        {
          hdf5::create_group(file, ("/" + hdf5::POPULATIONS).c_str());
        }

      if (!(H5Lexists (file, hdf5::population_path(pop_name).c_str(), H5P_DEFAULT) > 0))
        {
          hdf5::create_group(file, hdf5::population_path(pop_name));
        }

      string attr_prefix = hdf5::cell_attribute_prefix(attr_namespace, pop_name);
      if (!(H5Lexists (file, attr_prefix.c_str(), H5P_DEFAULT) > 0))
        {
          hdf5::create_group(file, attr_prefix);
        }

      string attr_prefix = hdf5::cell_attribute_prefix(attr_namespace, pop_name);
      string attr_path = hdf5::cell_attribute_path(attr_namespace, pop_name, attr_name);
      hid_t mspace, dset;
      
      switch (index_type)
        {
        case IndexOwner:
          {
            mspace = H5Screate_simple(1, &initial_size, maxdims);
            assert(mspace >= 0);
            dset = H5Dcreate2(file, (attr_path + "/" + hdf5::CELL_INDEX).c_str(),
                              CELL_IDX_H5_FILE_T,
                              mspace, lcpl, plist, H5P_DEFAULT);
            assert(H5Dclose(dset) >= 0);
            assert(H5Sclose(mspace) >= 0);
          }
          break;
        case IndexShared:
          {
            dset = H5Dopen2(file, (attr_prefix + "/" + hdf5::CELL_INDEX).c_str(), H5P_DEFAULT);
            assert(dset >= 0);
            status = H5Olink(dset, file, (attr_path + "/" + hdf5::CELL_INDEX).c_str(), H5P_DEFAULT, H5P_DEFAULT);
            assert(status >= 0);
            assert(H5Dclose(dset) >= 0);
          }
          break;
        }

      switch (ptr_type)
        {
        case PtrOwner:
          {
            mspace = H5Screate_simple(1, &initial_size, maxdims);
            assert(mspace >= 0);
            dset = H5Dcreate2(file, (attr_path + "/" + hdf5::ATTR_PTR).c_str(), ATTR_PTR_H5_FILE_T,
                              mspace, lcpl, plist, H5P_DEFAULT);
            assert(H5Dclose(dset) >= 0);
            assert(H5Sclose(mspace) >= 0);
          }
          break;
        case PtrShared:
          {
            dset = H5Dopen2(file, (attr_prefix + "/" + shared_ptr_name).c_str(), H5P_DEFAULT);
            assert(dset >= 0);
            status = H5Olink(dset, file, (attr_path + "/" + hdf5::ATTR_PTR).c_str(), H5P_DEFAULT, H5P_DEFAULT);
            assert(status >= 0);
            assert(H5Dclose(dset) >= 0);
          }
          break;
        }
      
      mspace = H5Screate_simple(1, &initial_size, maxdims);
      dset = H5Dcreate2(file, (attr_path + "/" + hdf5::ATTR_VAL).c_str(), ftype, mspace,
                        lcpl, value_plist, H5P_DEFAULT);
      assert(H5Dclose(dset) >= 0);
      assert(H5Sclose(mspace) >= 0);
    
      assert(H5Pclose(lcpl) >= 0);
    
      status = H5Pclose(plist);
      assert(status == 0);
      status = H5Pclose(value_plist);
      assert(status == 0);
    
    }

  
    void read_cell_attributes
    (
     MPI_Comm      comm,
     const string& file_name,
     const string& name_space,
     const string& pop_name,
     const CELL_IDX_T pop_start,
     data::NamedAttrMap& attr_values,
     size_t offset = 0,
     size_t numitems = 0
     )
    {
      herr_t status; 

      unsigned int rank, size;
      assert(MPI_Comm_size(comm, (int*)&size) >= 0);
      assert(MPI_Comm_rank(comm, (int*)&rank) >= 0);

      vector< pair<string,hid_t> > attr_info;
    
      status = get_cell_attributes (file_name, name_space,
                                    pop_name, attr_info);

      // get a file handle and retrieve the MPI info
      hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
      assert(H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL) >= 0);
      hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, fapl);
      assert(file >= 0);

      for (size_t i=0; i<attr_info.size(); i++)
        {
          vector<CELL_IDX_T>  index;
          vector<ATTR_PTR_T>  ptr;

          string attr_name  = attr_info[i].first;
          hid_t attr_h5type = attr_info[i].second;
          size_t attr_size  = H5Tget_size(attr_h5type);
          string attr_path  = hdf5::cell_attribute_path (name_space, pop_name, attr_name);

          switch (H5Tget_class(attr_h5type))
            {
            case H5T_INTEGER:
              if (attr_size == 4)
                {
                  if (H5Tget_sign( attr_h5type ) == H5T_SGN_NONE)
                    {
                      vector<uint32_t> attr_values_uint32;
                      status = hdf5::read_cell_attribute(comm, file, attr_path, pop_start,
                                                         index, ptr, attr_values_uint32,
                                                         offset, numitems);
                      attr_values.insert(attr_name, index, ptr, attr_values_uint32);
                    }
                  else
                    {
                      vector<int32_t> attr_values_int32;
                      status = hdf5::read_cell_attribute(comm, file, attr_path, pop_start,
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
                      status = hdf5::read_cell_attribute(comm, file, attr_path, pop_start,
                                                        index, ptr, attr_values_uint16,
                                                        offset, numitems);
                      attr_values.insert(attr_name, index, ptr, attr_values_uint16);
                    }
                  else
                    {
                      vector<int16_t> attr_values_int16;
                      status = hdf5::read_cell_attribute(comm, file, attr_path, pop_start,
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
                      status = hdf5::read_cell_attribute(comm, file, attr_path, pop_start,
                                                        index, ptr, attr_values_uint8,
                                                        offset, numitems);
                      attr_values.insert(attr_name, index, ptr, attr_values_uint8);
                    }
                  else
                    {
                      vector<int8_t> attr_values_int8;
                      status = hdf5::read_cell_attribute(comm, file, attr_path, pop_start,
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
                status = hdf5::read_cell_attribute(comm, file, attr_path, pop_start,
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
                      status = hdf5::read_cell_attribute(comm, file, attr_path, pop_start,
                                                        index, ptr, attr_values_uint8,
                                                        offset, numitems);
                      attr_values.insert(attr_name, index, ptr, attr_values_uint8);
                    }
                  else
                    {
                      vector<int8_t> attr_values_int8;
                      status = hdf5::read_cell_attribute(comm, file, attr_path, pop_start,
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
      assert(status == 0);
      status = H5Pclose(fapl);
      assert(status == 0);
    }



    int scatter_read_cell_attributes
    (
     MPI_Comm                      all_comm,
     const string                 &file_name,
     const int                     io_size,
     const string                 &attr_name_space,
     // A vector that maps nodes to compute ranks
     const map<CELL_IDX_T, rank_t> &node_rank_map,
     const string                 &pop_name,
     const CELL_IDX_T              pop_start,
     data::NamedAttrMap           &attr_map,
     // if positive, these arguments specify offset and number of entries to read
     // from the entries available to the current rank
     size_t offset   = 0,
     size_t numitems = 0
     )
    {
      int srank, ssize; size_t rank, size;
      assert(MPI_Comm_size(all_comm, &ssize) >= 0);
      assert(MPI_Comm_rank(all_comm, &srank) >= 0);
      assert(ssize > 0);
      assert(srank >= 0);
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

      assert(io_size > 0);
    
      vector<uint8_t> sendbuf; int sendpos = 0;
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

          map <size_t, data::AttrMap > rank_attr_map;
          {
            data::NamedAttrMap  attr_values;
            read_cell_attributes(io_comm, file_name, attr_name_space, pop_name, pop_start,
                                 attr_values, offset, numitems);
            append_rank_attr_map(node_rank_map, attr_values, rank_attr_map);
            attr_values.num_attrs(num_attrs);
            attr_values.attr_names(attr_names);
          }
            
          vector<int> rank_sequence;
          // Recommended all-to-all communication pattern: start at the current rank, then wrap around;
          // (as opposed to starting at rank 0)
          for (size_t dst_rank = rank; dst_rank < size; dst_rank++)
            {
              rank_sequence.push_back(dst_rank);
            }
          for (size_t dst_rank = 0; dst_rank < rank; dst_rank++)
            {
              rank_sequence.push_back(dst_rank);
            }
      
          for (const int& dst_rank : rank_sequence)
            {
              auto it1 = rank_attr_map.find(dst_rank); 
              sdispls[dst_rank] = sendpos;
	    
              if (it1 != rank_attr_map.end())
                {
                  data::AttrMap &m = it1->second;
		
                  // Create MPI_PACKED object with the number of gids for this rank
                  int packsize=0;
                  uint32_t rank_numitems = m.index_set.size();
                  assert(MPI_Pack_size(1, MPI_UINT32_T, all_comm, &packsize) == MPI_SUCCESS);
                  sendbuf.resize(sendbuf.size() + packsize);
                  assert(MPI_Pack(&rank_numitems, 1, MPI_UINT32_T, &sendbuf[0],
                                  (int)sendbuf.size(), &sendpos, all_comm) == MPI_SUCCESS);
		
                  for (auto it2 = m.index_set.begin(); it2 != m.index_set.end(); ++it2)
                    {
                      int packsize = 0;
                      CELL_IDX_T index = *it2;
		    

                      const vector<vector<float>>    float_values  = m.find<float>(index);
                      const vector<vector<uint8_t>>  uint8_values  = m.find<uint8_t>(index);
                      const vector<vector<int8_t>>   int8_values   = m.find<int8_t>(index);
                      const vector<vector<uint16_t>> uint16_values = m.find<uint16_t>(index);
                      const vector<vector<int16_t>>  int16_values  = m.find<int16_t>(index);
                      const vector<vector<uint32_t>> uint32_values = m.find<uint32_t>(index);
                      const vector<vector<int32_t>>  int32_values  = m.find<int32_t>(index);
		    
                      mpi::pack_size_index(all_comm, packsize);
                      mpi::pack_size_attr<float>(all_comm, MPI_FLOAT, index, float_values, packsize);
                      mpi::pack_size_attr<uint8_t>(all_comm, MPI_UINT8_T, index, uint8_values, packsize);
                      mpi::pack_size_attr<int8_t>(all_comm, MPI_INT8_T, index, int8_values, packsize);
                      mpi::pack_size_attr<uint16_t>(all_comm, MPI_UINT16_T, index, uint16_values, packsize);
                      mpi::pack_size_attr<int16_t>(all_comm, MPI_INT16_T, index, int16_values, packsize);
                      mpi::pack_size_attr<uint32_t>(all_comm, MPI_UINT32_T, index, uint32_values, packsize);
                      mpi::pack_size_attr<int32_t>(all_comm, MPI_INT32_T, index, int32_values, packsize);
                    
                      sendbuf.resize(sendbuf.size()+packsize);
                      int sendbuf_size = sendbuf.size();
                    
                      mpi::pack_index(all_comm, index, sendbuf_size, sendbuf, sendpos);
                      mpi::pack_attr<float>(all_comm, MPI_FLOAT, index, float_values, sendbuf_size, sendpos, sendbuf);
                      mpi::pack_attr<uint8_t>(all_comm, MPI_UINT8_T, index, uint8_values, sendbuf_size, sendpos, sendbuf);
                      mpi::pack_attr<int8_t>(all_comm, MPI_INT8_T, index, int8_values, sendbuf_size, sendpos, sendbuf);
                      mpi::pack_attr<uint16_t>(all_comm, MPI_UINT16_T, index, uint16_values, sendbuf_size, sendpos, sendbuf);
                      mpi::pack_attr<int16_t>(all_comm, MPI_INT16_T, index, int16_values, sendbuf_size, sendpos, sendbuf);
                      mpi::pack_attr<uint32_t>(all_comm, MPI_UINT32_T, index, uint32_values, sendbuf_size, sendpos, sendbuf);
                      mpi::pack_attr<int32_t>(all_comm, MPI_INT32_T, index, int32_values, sendbuf_size, sendpos, sendbuf);
                    }
                }
              sendcounts[dst_rank] = sendpos - sdispls[dst_rank];
            }
        }
      else
        {
          MPI_Comm_split(all_comm,0,rank,&io_comm);
        }
      MPI_Barrier(all_comm);
    
      vector<uint32_t> num_attrs_bcast(num_attrs.size());
      for (size_t i=0; i<num_attrs.size(); i++)
        {
          num_attrs_bcast[i] = num_attrs[i];
        }
      // 4. Broadcast the number of attributes of each type to all ranks
      assert(MPI_Bcast(&num_attrs_bcast[0], num_attrs_bcast.size(), MPI_UINT32_T, 0, all_comm) >= 0);
      for (size_t i=0; i<num_attrs.size(); i++)
        {
          num_attrs[i] = num_attrs_bcast[i];
        }
    
      // 5. Broadcast the names of each attributes of each type to all ranks
      assert(mpi::bcast_string_vector(all_comm, 0, MAX_ATTR_NAME_LEN, attr_names[data::AttrMap::attr_index_float]) >= 0);
      assert(mpi::bcast_string_vector(all_comm, 0, MAX_ATTR_NAME_LEN, attr_names[data::AttrMap::attr_index_uint8]) >= 0);
      assert(mpi::bcast_string_vector(all_comm, 0, MAX_ATTR_NAME_LEN, attr_names[data::AttrMap::attr_index_int8]) >= 0);
      assert(mpi::bcast_string_vector(all_comm, 0, MAX_ATTR_NAME_LEN, attr_names[data::AttrMap::attr_index_uint16]) >= 0);
      assert(mpi::bcast_string_vector(all_comm, 0, MAX_ATTR_NAME_LEN, attr_names[data::AttrMap::attr_index_int16]) >= 0);
      assert(mpi::bcast_string_vector(all_comm, 0, MAX_ATTR_NAME_LEN, attr_names[data::AttrMap::attr_index_uint32]) >= 0);
      assert(mpi::bcast_string_vector(all_comm, 0, MAX_ATTR_NAME_LEN, attr_names[data::AttrMap::attr_index_int32]) >= 0);
      for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_float]; i++)
        {
          attr_map.insert_name<float>(attr_names[data::AttrMap::attr_index_float][i],i);
        }
      for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_uint8]; i++)
        {
          attr_map.insert_name<uint8_t>(attr_names[data::AttrMap::attr_index_uint8][i],i);
        }
      for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_int8]; i++)
        {
          attr_map.insert_name<int8_t>(attr_names[data::AttrMap::attr_index_int8][i],i);
        }
      for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_uint16]; i++)
        {
          attr_map.insert_name<uint16_t>(attr_names[data::AttrMap::attr_index_uint16][i],i);
        }
      for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_int16]; i++)
        {
          attr_map.insert_name<int16_t>(attr_names[data::AttrMap::attr_index_int16][i],i);
        }
      for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_uint32]; i++)
        {
          attr_map.insert_name<uint32_t>(attr_names[data::AttrMap::attr_index_uint32][i],i);
        }
      for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_int32]; i++)
        {
          attr_map.insert_name<int32_t>(attr_names[data::AttrMap::attr_index_int32][i],i);
        }
    
      // 6. Each ALL_COMM rank sends an attribute set size to
      //    every other ALL_COMM rank (non IO_COMM ranks pass zero)
    
      assert(MPI_Alltoall(&sendcounts[0], 1, MPI_INT,
                          &recvcounts[0], 1, MPI_INT, all_comm) >= 0);
    
      // 7. Each ALL_COMM rank accumulates the vector sizes and allocates
      //    a receive buffer, recvcounts, and rdispls
      size_t recvbuf_size;
      vector<uint8_t> recvbuf;

      recvbuf_size = recvcounts[0];
      for (int p = 1; p < ssize; ++p)
        {
          rdispls[p] = rdispls[p-1] + recvcounts[p-1];
          recvbuf_size += recvcounts[p];
        }
      if (recvbuf_size > 0) recvbuf.resize(recvbuf_size);
    
      // 8. Each ALL_COMM rank participates in the MPI_Alltoallv
      assert(MPI_Alltoallv(&sendbuf[0], &sendcounts[0], &sdispls[0], MPI_PACKED,
                           &recvbuf[0], &recvcounts[0], &rdispls[0], MPI_PACKED,
                           all_comm) >= 0);
    
      sendbuf.clear();
    
      MPI_Barrier(all_comm);
    
      int recvpos = 0; 
      while ((size_t)recvpos < recvbuf_size)
        {
          uint32_t num_recv_items=0;
          // Unpack number of received blocks for this rank
          assert(MPI_Unpack(&recvbuf[0], recvbuf_size, 
                            &recvpos, &num_recv_items, 1, MPI_UINT32_T, all_comm) ==
                 MPI_SUCCESS);
	
          for (size_t i=0; i<num_recv_items; i++)
            {
              CELL_IDX_T index;
              mpi::unpack_index(all_comm, index, recvbuf_size, recvbuf, recvpos);
              assert((size_t)recvpos <= recvbuf_size);
              mpi::unpack_attr<float>(all_comm, MPI_FLOAT, index, attr_map, recvbuf_size, recvbuf, recvpos);
              assert((size_t)recvpos <= recvbuf_size);
              mpi::unpack_attr<uint8_t>(all_comm, MPI_UINT8_T, index, attr_map, recvbuf_size, recvbuf, recvpos);
              assert((size_t)recvpos <= recvbuf_size);
              mpi::unpack_attr<int8_t>(all_comm, MPI_INT8_T, index, attr_map, recvbuf_size, recvbuf, recvpos);
              assert((size_t)recvpos <= recvbuf_size);
              mpi::unpack_attr<uint16_t>(all_comm, MPI_UINT16_T, index, attr_map, recvbuf_size, recvbuf, recvpos);
              assert((size_t)recvpos <= recvbuf_size);
              mpi::unpack_attr<int16_t>(all_comm, MPI_INT16_T, index, attr_map, recvbuf_size, recvbuf, recvpos);
              assert((size_t)recvpos <= recvbuf_size);
              mpi::unpack_attr<uint32_t>(all_comm, MPI_UINT32_T, index, attr_map, recvbuf_size, recvbuf, recvpos);
              assert((size_t)recvpos <= recvbuf_size);
              mpi::unpack_attr<int32_t>(all_comm, MPI_INT32_T, index, attr_map, recvbuf_size, recvbuf, recvpos);
              assert((size_t)recvpos <= recvbuf_size);
            }
        }
      assert(MPI_Comm_free(&io_comm) == MPI_SUCCESS);

      return 0;
    }

  
    void bcast_cell_attributes
    (
     MPI_Comm      comm,
     const int     root,
     const string& file_name,
     const string& name_space,
     const string& pop_name,
     const CELL_IDX_T pop_start,
     data::NamedAttrMap& attr_map,
     size_t offset = 0,
     size_t numitems = 0
     )
    {
      herr_t status; 

    
      fflush(stdout);
      unsigned int rank, size;
      assert(MPI_Comm_size(comm, (int*)&size) >= 0);
      assert(MPI_Comm_rank(comm, (int*)&rank) >= 0);
      fflush(stdout);

      vector<uint8_t> sendrecvbuf; 
      vector< pair<string,hid_t> > attr_info;

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
      assert(H5Pset_fapl_mpio(fapl, io_comm, MPI_INFO_NULL) >= 0);

    
      hid_t file;
      if (rank == (unsigned int)root)
        {

          status = cell::get_cell_attributes (file_name, name_space,
                                              pop_name, attr_info);
        
          file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, fapl);
          assert(file >= 0);
        
          for (size_t i=0; i<attr_info.size(); i++)
            {
              vector<CELL_IDX_T>  index;
              vector<ATTR_PTR_T>  ptr;
            
              string attr_name  = attr_info[i].first;
              hid_t attr_h5type = attr_info[i].second;
              size_t attr_size  = H5Tget_size(attr_h5type);
              string attr_path  = hdf5::cell_attribute_path (name_space, pop_name, attr_name);
            
              switch (H5Tget_class(attr_h5type))
                {
                case H5T_INTEGER:
                  if (attr_size == 4)
                    {
                      if (H5Tget_sign( attr_h5type ) == H5T_SGN_NONE)
                        {
                          vector<uint32_t> attr_map_uint32;
                          status = hdf5::read_cell_attribute(io_comm, file, attr_path, pop_start,
                                                            index, ptr, attr_map_uint32);
                          attr_map.insert(attr_name, index, ptr, attr_map_uint32);
                        }
                      else
                        {
                          vector<int32_t> attr_map_int32;
                          status = hdf5::read_cell_attribute(io_comm, file, attr_path, pop_start,
                                                            index, ptr, attr_map_int32);
                          attr_map.insert(attr_name, index, ptr, attr_map_int32);
                        }
                    }
                  else if (attr_size == 2)
                    {
                      if (H5Tget_sign( attr_h5type ) == H5T_SGN_NONE)
                        {
                          vector<uint16_t> attr_map_uint16;
                          status = hdf5::read_cell_attribute(io_comm, file, attr_path, pop_start,
                                                            index, ptr, attr_map_uint16);
                          attr_map.insert(attr_name, index, ptr, attr_map_uint16);
                        }
                      else
                        {
                          vector<int16_t> attr_map_int16;
                          status = hdf5::read_cell_attribute(io_comm, file, attr_path, pop_start,
                                                            index, ptr, attr_map_int16);
                          attr_map.insert(attr_name, index, ptr, attr_map_int16);
                        }
                    }
                  else if (attr_size == 1)
                    {
                      if (H5Tget_sign( attr_h5type ) == H5T_SGN_NONE)
                        {
                          vector<uint8_t> attr_map_uint8;
                          status = hdf5::read_cell_attribute(io_comm, file, attr_path, pop_start,
                                                            index, ptr, attr_map_uint8);
                          attr_map.insert(attr_name, index, ptr, attr_map_uint8);
                        }
                      else
                        {
                          vector<int8_t> attr_map_int8;
                          status = hdf5::read_cell_attribute(io_comm, file, attr_path, pop_start,
                                                            index, ptr, attr_map_int8);
                          attr_map.insert(attr_name, index, ptr, attr_map_int8);
                        }
                    }
                  else
                    {
                      throw runtime_error("Unsupported integer attribute size");
                    };
                  break;
                case H5T_FLOAT:
                  {
                    vector<float> attr_map_float;
                    status = hdf5::read_cell_attribute(io_comm, file, attr_path, pop_start,
                                                      index, ptr, attr_map_float);
                    attr_map.insert(attr_name, index, ptr, attr_map_float);
                  }
                  break;
                case H5T_ENUM:
                  if (attr_size == 1)
                    {
                      if (H5Tget_sign( attr_h5type ) == H5T_SGN_NONE)
                        {
                          vector<uint8_t> attr_map_uint8;
                          status = hdf5::read_cell_attribute(io_comm, file, attr_path, pop_start,
                                                            index, ptr, attr_map_uint8);
                          attr_map.insert(attr_name, index, ptr, attr_map_uint8);
                        }
                      else
                        {
                          vector<int8_t> attr_map_int8;
                          status = hdf5::read_cell_attribute(io_comm, file, attr_path, pop_start,
                                                            index, ptr, attr_map_int8);
                          attr_map.insert(attr_name, index, ptr, attr_map_int8);
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
          assert(status == 0);

          attr_map.num_attrs(num_attrs);
          attr_map.attr_names(attr_names);

          // Create MPI_PACKED object with the number of gids
          int sendpos = 0;
          int packsize=0;
          uint32_t numitems = attr_map.index_set.size();
          assert(MPI_Pack_size(1, MPI_UINT32_T, comm, &packsize) == MPI_SUCCESS);
          sendrecvbuf.resize(sendrecvbuf.size() + packsize);
          assert(MPI_Pack(&numitems, 1, MPI_UINT32_T, &sendrecvbuf[0],
                          (int)sendrecvbuf.size(), &sendpos, comm) == MPI_SUCCESS);
		
          for (auto it = attr_map.index_set.begin(); it != attr_map.index_set.end(); ++it)
            {
              int packsize = 0;
              CELL_IDX_T index = *it;

              const vector<vector<float>>    float_values = attr_map.find<float>(index);
              const vector<vector<uint8_t>>  uint8_values = attr_map.find<uint8_t>(index);
              const vector<vector<int8_t>>   int8_values = attr_map.find<int8_t>(index);
              const vector<vector<uint16_t>> uint16_values = attr_map.find<uint16_t>(index);
              const vector<vector<int16_t>>  int16_values = attr_map.find<int16_t>(index);
              const vector<vector<uint32_t>> uint32_values = attr_map.find<uint32_t>(index);
              const vector<vector<int32_t>>  int32_values = attr_map.find<int32_t>(index);
		    
              mpi::pack_size_index(comm, packsize);
              mpi::pack_size_attr<float>(comm, MPI_FLOAT, index, float_values, packsize);
              mpi::pack_size_attr<uint8_t>(comm, MPI_UINT8_T, index, uint8_values, packsize);
              mpi::pack_size_attr<int8_t>(comm, MPI_INT8_T, index, int8_values, packsize);
              mpi::pack_size_attr<uint16_t>(comm, MPI_UINT16_T, index, uint16_values, packsize);
              mpi::pack_size_attr<int16_t>(comm, MPI_INT16_T, index, int16_values, packsize);
              mpi::pack_size_attr<uint32_t>(comm, MPI_UINT32_T, index, uint32_values, packsize);
              mpi::pack_size_attr<int32_t>(comm, MPI_INT32_T, index, int32_values, packsize);
              
              sendrecvbuf.resize(sendrecvbuf.size()+packsize);
              int sendrecvbuf_size = sendrecvbuf.size();
              
              mpi::pack_index(comm, index, sendrecvbuf_size, sendrecvbuf, sendpos);
              mpi::pack_attr<float>(comm, MPI_FLOAT, index, float_values, sendrecvbuf_size, sendpos, sendrecvbuf);
              mpi::pack_attr<uint8_t>(comm, MPI_UINT8_T, index, uint8_values, sendrecvbuf_size, sendpos, sendrecvbuf);
              mpi::pack_attr<int8_t>(comm, MPI_INT8_T, index, int8_values, sendrecvbuf_size, sendpos, sendrecvbuf);
              mpi::pack_attr<uint16_t>(comm, MPI_UINT16_T, index, uint16_values, sendrecvbuf_size, sendpos, sendrecvbuf);
              mpi::pack_attr<int16_t>(comm, MPI_INT16_T, index, int16_values, sendrecvbuf_size, sendpos, sendrecvbuf);
              mpi::pack_attr<uint32_t>(comm, MPI_UINT32_T, index, uint32_values, sendrecvbuf_size, sendpos, sendrecvbuf);
              mpi::pack_attr<int32_t>(comm, MPI_INT32_T, index, int32_values, sendrecvbuf_size, sendpos, sendrecvbuf);
            }
        }

      assert(MPI_Comm_free(&io_comm) == MPI_SUCCESS);

      vector<uint32_t> num_attrs_bcast(num_attrs.size());
      for (size_t i=0; i<num_attrs.size(); i++)
        {
          num_attrs_bcast[i] = num_attrs[i];
        }
      // Broadcast the number of attributes of each type to all ranks
      assert(MPI_Bcast(&num_attrs_bcast[0], num_attrs_bcast.size(), MPI_UINT32_T, root, comm) >= 0);
      for (size_t i=0; i<num_attrs.size(); i++)
        {
          num_attrs[i] = num_attrs_bcast[i];
        }
    
      // Broadcast the names of each attributes of each type to all ranks
      assert(mpi::bcast_string_vector(comm, 0, MAX_ATTR_NAME_LEN, attr_names[data::AttrMap::attr_index_float]) >= 0);
      assert(mpi::bcast_string_vector(comm, 0, MAX_ATTR_NAME_LEN, attr_names[data::AttrMap::attr_index_uint8]) >= 0);
      assert(mpi::bcast_string_vector(comm, 0, MAX_ATTR_NAME_LEN, attr_names[data::AttrMap::attr_index_int8]) >= 0);
      assert(mpi::bcast_string_vector(comm, 0, MAX_ATTR_NAME_LEN, attr_names[data::AttrMap::attr_index_uint16]) >= 0);
      assert(mpi::bcast_string_vector(comm, 0, MAX_ATTR_NAME_LEN, attr_names[data::AttrMap::attr_index_int16]) >= 0);
      assert(mpi::bcast_string_vector(comm, 0, MAX_ATTR_NAME_LEN, attr_names[data::AttrMap::attr_index_uint32]) >= 0);
      assert(mpi::bcast_string_vector(comm, 0, MAX_ATTR_NAME_LEN, attr_names[data::AttrMap::attr_index_int32]) >= 0);
      for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_float]; i++)
        {
          attr_map.insert_name<float>(attr_names[data::AttrMap::attr_index_float][i],i);
        }
      for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_uint8]; i++)
        {
          attr_map.insert_name<uint8_t>(attr_names[data::AttrMap::attr_index_uint8][i],i);
        }
      for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_int8]; i++)
        {
          attr_map.insert_name<int8_t>(attr_names[data::AttrMap::attr_index_int8][i],i);
        }
      for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_uint16]; i++)
        {
          attr_map.insert_name<uint16_t>(attr_names[data::AttrMap::attr_index_uint16][i],i);
        }
      for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_int16]; i++)
        {
          attr_map.insert_name<int16_t>(attr_names[data::AttrMap::attr_index_int16][i],i);
        }
      for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_uint32]; i++)
        {
          attr_map.insert_name<uint32_t>(attr_names[data::AttrMap::attr_index_uint32][i],i);
        }
      for (size_t i=0; i<num_attrs[data::AttrMap::attr_index_int32]; i++)
        {
          attr_map.insert_name<int32_t>(attr_names[data::AttrMap::attr_index_int32][i],i);
        }

      uint32_t sendrecvbuf_size = sendrecvbuf.size();
      assert(MPI_Bcast(&sendrecvbuf_size, 1, MPI_UINT32_T, root, comm) == MPI_SUCCESS);
      sendrecvbuf.resize(sendrecvbuf_size);
      assert(MPI_Bcast(&sendrecvbuf[0], sendrecvbuf_size, MPI_PACKED, root, comm) == MPI_SUCCESS);
    
      if (rank != (unsigned int)root)
        {
          int recvpos = 0; 
          while ((size_t)recvpos < sendrecvbuf_size)
            {
              uint32_t num_recv_items=0;
              // Unpack number of received blocks for this rank
              assert(MPI_Unpack(&sendrecvbuf[0], sendrecvbuf_size, 
                                &recvpos, &num_recv_items, 1, MPI_UINT32_T, comm) ==
                     MPI_SUCCESS);
            
              for (size_t i=0; i<num_recv_items; i++)
                {
                  CELL_IDX_T index;
                  mpi::unpack_index(comm, index, sendrecvbuf_size, sendrecvbuf, recvpos);
                  assert((size_t)recvpos <= sendrecvbuf_size);
                  mpi::unpack_attr<float>(comm, MPI_FLOAT, index, attr_map, sendrecvbuf_size, sendrecvbuf, recvpos);
                  assert((size_t)recvpos <= sendrecvbuf_size);
                  mpi::unpack_attr<uint8_t>(comm, MPI_UINT8_T, index, attr_map, sendrecvbuf_size, sendrecvbuf, recvpos);
                  assert((size_t)recvpos <= sendrecvbuf_size);
                  mpi::unpack_attr<int8_t>(comm, MPI_INT8_T, index, attr_map, sendrecvbuf_size, sendrecvbuf, recvpos);
                  assert((size_t)recvpos <= sendrecvbuf_size);
                  mpi::unpack_attr<uint16_t>(comm, MPI_UINT16_T, index, attr_map, sendrecvbuf_size, sendrecvbuf, recvpos);
                  assert((size_t)recvpos <= sendrecvbuf_size);
                  mpi::unpack_attr<int16_t>(comm, MPI_INT16_T, index, attr_map, sendrecvbuf_size, sendrecvbuf, recvpos);
                  assert((size_t)recvpos <= sendrecvbuf_size);
                  mpi::unpack_attr<uint32_t>(comm, MPI_UINT32_T, index, attr_map, sendrecvbuf_size, sendrecvbuf, recvpos);
                  assert((size_t)recvpos <= sendrecvbuf_size);
                  mpi::unpack_attr<int32_t>(comm, MPI_INT32_T, index, attr_map, sendrecvbuf_size, sendrecvbuf, recvpos);
                  assert((size_t)recvpos <= sendrecvbuf_size);
                }
            }
        }
    
      status = H5Pclose(fapl);
      assert(status == 0);

    }
  }
}
