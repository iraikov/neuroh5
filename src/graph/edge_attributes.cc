// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file edge_attributes.cc
///
///  Functions for edge attribute read/write.
///
///  Copyright (C) 2016-2020 Project NeuroH5.
//==============================================================================

#include "attr_val.hh"
#include "attr_kind_datatype.hh"
#include "edge_attributes.hh"
#include "exists_dataset.hh"
#include "exists_group.hh"
#include "path_names.hh"
#include "serialize_data.hh"
#include "read_template.hh"
#include "mpi_debug.hh"
#include "throw_assert.hh"

#include <iostream>

using namespace std;

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
     )
    {
      string path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name,
                                              attr_namespace, attr_name);
      value_size = hdf5::dataset_num_elements(loc, path);
    }

    
    void create_projection_groups
    (
     const hid_t&   file,
     const string&  src_pop_name,
     const string&  dst_pop_name
     )
    {
      string path = "/" + hdf5::PROJECTIONS;
      if (!(hdf5::exists_dataset (file, path) > 0))
        {
          hdf5::create_group(file, path.c_str());
        }
      
      path = "/" + hdf5::PROJECTIONS + "/" + dst_pop_name;
            
      if (!(hdf5::exists_dataset (file, path) > 0))
        {
          hdf5::create_group(file, path.c_str());
        }

      path = hdf5::projection_prefix(src_pop_name, dst_pop_name);

      if (!(hdf5::exists_dataset (file, path) > 0))
        {
          hdf5::create_group(file, path.c_str());
        }
    
    }

    void create_edge_attribute_datasets
    (
     const hid_t&   file,
     const string&  src_pop_name,
     const string&  dst_pop_name,
     const string&  attr_namespace,
     const string&  attr_name,
     const hid_t&   ftype,
     const size_t   chunk_size
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
      status = H5Pset_deflate(plist, 9);
      throw_assert_nomsg(status == 0);
#endif
      
      hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
      throw_assert_nomsg(lcpl >= 0);
      throw_assert_nomsg(H5Pset_create_intermediate_group(lcpl, 1) >= 0);

      create_projection_groups(file, src_pop_name, dst_pop_name);

      string attr_path = hdf5::edge_attribute_path(src_pop_name,
                                                   dst_pop_name,
                                                   attr_namespace,
                                                   attr_name);
      
      hid_t mspace = H5Screate_simple(1, &initial_size, maxdims);
      hid_t dset = H5Dcreate2(file, attr_path.c_str(), ftype, mspace,
                              lcpl, plist, H5P_DEFAULT);
      throw_assert_nomsg(H5Dclose(dset) >= 0);
      throw_assert_nomsg(H5Sclose(mspace) >= 0);
    
      throw_assert_nomsg(H5Pclose(lcpl) >= 0);
    
      status = H5Pclose(plist);
      throw_assert_nomsg(status == 0);
    
    }

  }

  namespace graph
  {
    //////////////////////////////////////////////////////////////////////////
    // Callback for H5Literate
    static herr_t edge_attribute_cb
    (
     hid_t             group_id,
     const char*       name,
     const H5L_info_t* info,
     void*             op_data
     )
    {
      hid_t dset = H5Dopen2(group_id, name, H5P_DEFAULT);

      if (dset < 0) // skip the link, if this is not a dataset
        {
          return 0;
        }
      
      hid_t ftype = H5Dget_type(dset);
      throw_assert_nomsg(ftype >= 0);
      
      vector< pair<string,AttrKind> >* ptr =
        (vector< pair<string,AttrKind> >*) op_data;
      ptr->push_back(make_pair(name, hdf5::h5type_attr_kind(ftype)));
      
      throw_assert_nomsg(H5Dclose(dset) >= 0);

      return 0;
    }

    /////////////////////////////////////////////////////////////////////////
    herr_t has_edge_attribute_namespace
    (
     MPI_Comm                      comm,
     const string&                 file_name,
     const string&                 src_pop_name,
     const string&                 dst_pop_name,
     const string&                 name_space,
     bool &has_namespace
     )
    {
      herr_t ierr=0;
      int root=0;
      int rank, size;
      throw_assert_nomsg(MPI_Comm_size(comm, &size) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS);
      uint8_t has_namespace_flag = 0;

      if (rank == root)
        {
          hid_t in_file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
          throw_assert_nomsg(in_file >= 0);
          
          string path = hdf5::edge_attribute_prefix(src_pop_name, dst_pop_name, name_space);
          
          ierr = hdf5::exists_group (in_file, path.c_str());
          if (ierr > 0)
            {
              has_namespace_flag = 1;
            }
          else
            {
              has_namespace_flag = 0;
            }
          ierr = H5Fclose(in_file);
        }

      throw_assert_nomsg(MPI_Bcast(&has_namespace_flag, 1, MPI_UINT8_T, root, comm) == MPI_SUCCESS);
      
      if (has_namespace_flag > 0)
        {
          has_namespace = true;
        }
      else
        {
          has_namespace = false;
        }
      return ierr;
    }

    
    /////////////////////////////////////////////////////////////////////////
    herr_t get_edge_attributes
    (
     MPI_Comm                      comm,
     const string&                 file_name,
     const string&                 src_pop_name,
     const string&                 dst_pop_name,
     const string&                 name_space,
     vector< pair<string,AttrKind> >& out_attributes
     )
    {
      herr_t ierr=0;
      int root=0;
      int rank, size;
      throw_assert_nomsg(MPI_Comm_size(comm, &size) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS);

      if (rank == root)
        {
          hid_t in_file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
          throw_assert_nomsg(in_file >= 0);
          out_attributes.clear();
          
          string path = hdf5::edge_attribute_prefix(src_pop_name, dst_pop_name, name_space);
          
          ierr = hdf5::exists_dataset (in_file, path.c_str());
          if (ierr > 0)
            {
              hid_t grp = H5Gopen2(in_file, path.c_str(), H5P_DEFAULT);
              if (grp >= 0)
                {
                  hsize_t idx = 0;
                  ierr = H5Literate(grp, H5_INDEX_NAME, H5_ITER_NATIVE, &idx,
                                    &edge_attribute_cb, (void*) &out_attributes);
                  
                  throw_assert_nomsg(H5Gclose(grp) >= 0);
                }
            }
          ierr = H5Fclose(in_file);
        }

      vector<char> edge_attributes_sendbuf;  size_t edge_attributes_sendbuf_size=0;
      if (rank == root)
        {
          data::serialize_data(out_attributes, edge_attributes_sendbuf);
          edge_attributes_sendbuf_size = edge_attributes_sendbuf.size();
        }

      throw_assert_nomsg(MPI_Bcast(&edge_attributes_sendbuf_size, 1, MPI_SIZE_T, root, comm) == MPI_SUCCESS);
      edge_attributes_sendbuf.resize(edge_attributes_sendbuf_size);

      ierr = MPI_Bcast (&edge_attributes_sendbuf[0], edge_attributes_sendbuf_size,
                        MPI_CHAR, root, comm);
      throw_assert_nomsg(ierr == MPI_SUCCESS);

      out_attributes.clear();

      data::deserialize_data(edge_attributes_sendbuf, out_attributes);

      return ierr;
    }
    
    /////////////////////////////////////////////////////////////////////////
    herr_t num_edge_attributes
    (
     const vector< pair<string,AttrKind> >& attributes,
     vector <size_t> &num_attrs
     )
    {
      herr_t ierr = 0;
      num_attrs.resize(data::AttrVal::num_attr_types);
      for (size_t i = 0; i < attributes.size(); i++)
        {
          AttrKind attr_kind = attributes[i].second;
          size_t attr_size = attr_kind.size;
          switch (attr_kind.type)
            {
            case UIntVal:
              if (attr_size == 4)
                {
                  num_attrs[data::AttrVal::attr_index_uint32]++;
                }
              else if (attr_size == 2)
                {
                  num_attrs[data::AttrVal::attr_index_uint16]++;
                }
              else if (attr_size == 1)
                {
                  num_attrs[data::AttrVal::attr_index_uint8]++;
                }
              else
                {
                  throw runtime_error("Unsupported integer attribute size");
                };
              break;
            case SIntVal:
              if (attr_size == 4)
                {
                  num_attrs[data::AttrVal::attr_index_int32]++;
                }
              else if (attr_size == 2)
                {
                  num_attrs[data::AttrVal::attr_index_int16]++;
                }
              else if (attr_size == 1)
                {
                  num_attrs[data::AttrVal::attr_index_int8]++;
                }
              else
                {
                  throw runtime_error("Unsupported integer attribute size");
                };
              break;
            case FloatVal:
              num_attrs[data::AttrVal::attr_index_float]++;
              break;
            case EnumVal:
              if (attr_size == 1)
                {
                  num_attrs[data::AttrVal::attr_index_uint8]++;
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

    /////////////////////////////////////////////////////////////////////////
    herr_t read_edge_attributes
    (
     MPI_Comm              comm,
     const string&         file_name,
     const string&         src_pop_name,
     const string&         dst_pop_name,
     const string&         name_space,
     const string&         attr_name,
     const DST_PTR_T       edge_base,
     const DST_PTR_T       edge_count,
     const AttrKind        attr_kind,
     data::NamedAttrVal&   attr_values,
     bool collective
     )
    {
      hid_t file;
      herr_t ierr = 0;
      hsize_t block = edge_count, base = edge_base;

      hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
      throw_assert_nomsg(fapl >= 0);
      throw_assert_nomsg(H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL) >= 0);
      
      throw_assert_nomsg(MPI_Barrier(comm) == MPI_SUCCESS);

      file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, fapl);
      throw_assert_nomsg(file >= 0);

      /* Create property list for collective dataset operations. */
      hid_t rapl = H5Pcreate (H5P_DATASET_XFER);
#ifdef HDF5_IS_PARALLEL
      if (collective)
        {
          ierr = H5Pset_dxpl_mpio (rapl, H5FD_MPIO_COLLECTIVE);
          throw_assert(ierr >= 0,
                       "read_edge_attributes: error in H5Pset_dxpl_mpio");
        }
#endif
      mpi::MPI_DEBUG(comm, "read_edge_attributes: reading attributes for ", 
                     src_pop_name, " -> ", dst_pop_name, 
                     " namespace ", name_space, " attribute ", attr_name, 
                     " edge_base: ", edge_base, " edge count: ", edge_count);
      
      string dset_path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name, name_space, attr_name);
      ierr = hdf5::exists_dataset (file, dset_path.c_str());
      if (ierr > 0)
        {
          hsize_t one = 1;

          hid_t mspace = H5Screate_simple(1, &block, NULL);
          throw_assert_nomsg(mspace >= 0);
          ierr = H5Sselect_all(mspace);
          throw_assert_nomsg(ierr >= 0);
          
          hid_t dset = H5Dopen2 (file, dset_path.c_str(), H5P_DEFAULT);
          throw_assert_nomsg(dset >= 0);
          
          // make hyperslab selection
          hid_t fspace = H5Dget_space(dset);
          throw_assert_nomsg(fspace >= 0);
          
          if (block > 0)
            {
              ierr = H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &base, NULL, &one, &block);
            }
          else
            {
              ierr = H5Sselect_none(fspace);
            }
          throw_assert_nomsg(ierr >= 0);
          
          size_t attr_size = attr_kind.size;
          hid_t attr_h5type = hdf5::attr_kind_h5type(attr_kind);
          switch (attr_kind.type)
            {
            case UIntVal:
              if (attr_size == 4)
                {
                  vector <uint32_t> attr_values_uint32;
                  attr_values_uint32.resize(edge_count);
                  ierr = H5Dread(dset, attr_h5type, mspace, fspace, rapl,
                                 &attr_values_uint32[0]);
                  attr_values.insert(string(attr_name), attr_values_uint32);
                }
              else if (attr_size == 2)
                {
                  vector <uint16_t>    attr_values_uint16;
                  attr_values_uint16.resize(edge_count);
                  ierr = H5Dread(dset, attr_h5type, mspace, fspace, rapl,
                                 &attr_values_uint16[0]);
                  attr_values.insert(string(attr_name), attr_values_uint16);
                }
              else if (attr_size == 1)
                {
                  vector <uint8_t> attr_values_uint8;
                  attr_values_uint8.resize(edge_count);
                  ierr = H5Dread(dset, attr_h5type, mspace, fspace, rapl,
                                 &attr_values_uint8[0]);
                  attr_values.insert(string(attr_name), attr_values_uint8);
                }
              else
                {
                  throw runtime_error("Unsupported integer attribute size");
                };
              break;
            case SIntVal:
              if (attr_size == 4)
                {
                  vector <int32_t>  attr_values_int32;
                  attr_values_int32.resize(edge_count);
                  ierr = H5Dread(dset, attr_h5type, mspace, fspace, rapl,
                                 &attr_values_int32[0]);
                  attr_values.insert(string(attr_name), attr_values_int32);
                }
              else if (attr_size == 2)
                {
                  vector <int16_t>  attr_values_int16;
                  attr_values_int16.resize(edge_count);
                  ierr = H5Dread(dset, attr_h5type, mspace, fspace, rapl,
                                 &attr_values_int16[0]);
                  attr_values.insert(string(attr_name), attr_values_int16);
                }
              else if (attr_size == 1)
                {
                  vector <int8_t>  attr_values_int8;
                  attr_values_int8.resize(edge_count);
                  ierr = H5Dread(dset, attr_h5type, mspace, fspace, rapl,
                                 &attr_values_int8[0]);
                  attr_values.insert(string(attr_name), attr_values_int8);
                }
              else
                {
                  throw runtime_error("Unsupported integer attribute size");
                };
              break;
            case FloatVal:
              {
                vector <float>  attr_values_float;
                attr_values_float.resize(edge_count);
                ierr = H5Dread(dset, attr_h5type, mspace, fspace, rapl,
                               &attr_values_float[0]);
                attr_values.insert(string(attr_name), attr_values_float);
              }
              break;
            case EnumVal:
              if (attr_size == 1)
                {
                  vector <uint8_t>  attr_values_uint8;;
                  attr_values_uint8.resize(edge_count);
                  ierr = H5Dread(dset, attr_h5type, mspace, fspace, rapl,
                                 &attr_values_uint8[0]);
                  attr_values.insert(string(attr_name), attr_values_uint8);
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
          
          throw_assert_nomsg(H5Sclose(fspace) >= 0);
          throw_assert_nomsg(H5Dclose(dset) >= 0);
          throw_assert_nomsg(H5Sclose(mspace) >= 0);
        }

      throw_assert_nomsg(H5Fclose(file) >= 0);
      throw_assert_nomsg(H5Pclose(fapl) >= 0);
      throw_assert_nomsg(H5Pclose(rapl) >= 0);

      return ierr;
    }



    /////////////////////////////////////////////////////////////////////////
    int read_all_edge_attributes
    (
     MPI_Comm                            comm,
     const string&                       file_name,
     const string&                       src_pop_name,
     const string&                       dst_pop_name,
     const string&                       name_space,
     const DST_PTR_T                     edge_base,
     const DST_PTR_T                     edge_count,
     const vector< pair<string,AttrKind> >& edge_attr_info,
     data::NamedAttrVal&                 edge_attr_values
     )
    {
      int ierr = 0;
      vector<NODE_IDX_T> src_vec, dst_vec;

      for (size_t j = 0; j < edge_attr_info.size(); j++)
        {
          string attr_name   = edge_attr_info[j].first;
          AttrKind attr_kind = edge_attr_info[j].second;
          throw_assert_nomsg ((ierr = read_edge_attributes(comm, file_name, src_pop_name, dst_pop_name,
                                               name_space, attr_name, edge_base, edge_count,
                                               attr_kind, edge_attr_values))
                  >= 0);
        }

      return ierr;
    }
    

    /////////////////////////////////////////////////////////////////////////
    herr_t read_edge_attribute_selection
    (
     MPI_Comm              comm,
     const string&         file_name,
     const string&         src_pop_name,
     const string&         dst_pop_name,
     const string&         name_space,
     const string&         attr_name,
     const DST_PTR_T&       edge_base,
     const DST_PTR_T&       edge_count,
     const vector<NODE_IDX_T>&   selection_dst_idx,
     const vector<DST_PTR_T>&    selection_dst_ptr,
     const vector< pair<hsize_t,hsize_t> >& ranges,
     const AttrKind        attr_kind,
     data::NamedAttrVal&   attr_values,
     bool collective
     )
    {
      hid_t file;
      herr_t ierr = 0;
      hsize_t block = 0;
      hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
      throw_assert_nomsg(fapl >= 0);
      throw_assert_nomsg(H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL) >= 0);

      file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, fapl);
      throw_assert_nomsg(file >= 0);

      /* Create property list for collective dataset operations. */
      hid_t rapl = H5Pcreate (H5P_DATASET_XFER);
#ifdef HDF5_IS_PARALLEL
      if (collective)
        {
          ierr = H5Pset_dxpl_mpio (rapl, H5FD_MPIO_COLLECTIVE);
          throw_assert(ierr >= 0,
                       "read_edge_attribute_selection: error in H5Pset_dxpl_mpio");
        }
#endif

      for ( const std::pair<hsize_t,hsize_t> &range : ranges )
        {
          hsize_t start = range.first;
          hsize_t count = range.second;


        }
      
      string dset_path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name, name_space, attr_name);
      ierr = hdf5::exists_dataset (file, dset_path.c_str());
      if (ierr > 0)
        {
          size_t attr_size = attr_kind.size;
          hid_t attr_h5type = hdf5::attr_kind_h5type(attr_kind);
          switch (attr_kind.type)
            {
            case UIntVal:
              if (attr_size == 4)
                {
                  vector <uint32_t> attr_values_uint32;
                  ierr = hdf5::read_selection<uint32_t>(file, dset_path, attr_h5type, ranges,
                                                        attr_values_uint32, rapl);
                  attr_values.insert(string(attr_name), attr_values_uint32);
                }
              else if (attr_size == 2)
                {
                  vector <uint16_t>    attr_values_uint16;
                  ierr = hdf5::read_selection<uint16_t>(file, dset_path, attr_h5type, ranges,
                                                        attr_values_uint16, rapl);
                  attr_values.insert(string(attr_name), attr_values_uint16);
                }
              else if (attr_size == 1)
                {
                  vector <uint8_t> attr_values_uint8;
                  ierr = hdf5::read_selection<uint8_t>(file, dset_path, attr_h5type, ranges,
                                                       attr_values_uint8, rapl);
                  attr_values.insert(string(attr_name), attr_values_uint8);
                }
              else
                {
                  throw runtime_error("Unsupported integer attribute size");
                };
              break;
            case SIntVal:
              if (attr_size == 4)
                {
                  vector <int32_t>  attr_values_int32;
                  ierr = hdf5::read_selection<int32_t>(file, dset_path, attr_h5type, ranges,
                                                       attr_values_int32, rapl);
                  attr_values.insert(string(attr_name), attr_values_int32);
                }
              else if (attr_size == 2)
                {
                  vector <int16_t>  attr_values_int16;
                  ierr = hdf5::read_selection<int16_t>(file, dset_path, attr_h5type, ranges,
                                                       attr_values_int16, rapl);
                  attr_values.insert(string(attr_name), attr_values_int16);
                }
              else if (attr_size == 1)
                {
                  vector <int8_t>  attr_values_int8;
                  ierr = hdf5::read_selection<int8_t>(file, dset_path, attr_h5type, ranges,
                                                      attr_values_int8, rapl);
                  attr_values.insert(string(attr_name), attr_values_int8);
                }
              else
                {
                  throw runtime_error("Unsupported integer attribute size");
                };
              break;
            case FloatVal:
              {
                vector <float>  attr_values_float;
                ierr = hdf5::read_selection<float>(file, dset_path, attr_h5type, ranges,
                                                   attr_values_float, rapl);
                attr_values.insert(string(attr_name), attr_values_float);
              }
              break;
            case EnumVal:
              if (attr_size == 1)
                {
                  vector <uint8_t>  attr_values_uint8;
                  ierr = hdf5::read_selection<uint8_t>(file, dset_path, attr_h5type, ranges,
                                                       attr_values_uint8, rapl);

                  attr_values.insert(string(attr_name), attr_values_uint8);
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

      throw_assert_nomsg(H5Fclose(file) >= 0);
      throw_assert_nomsg(H5Pclose(fapl) >= 0);
      throw_assert_nomsg(H5Pclose(rapl) >= 0);

      return ierr;
    }


    /////////////////////////////////////////////////////////////////////////
    int read_all_edge_attribute_selection
    (
     MPI_Comm                            comm,
     const string&                       file_name,
     const string&                       src_pop_name,
     const string&                       dst_pop_name,
     const string&                       name_space,
     const DST_PTR_T&       edge_base,
     const DST_PTR_T&       edge_count,
     const vector<NODE_IDX_T>&   selection_dst_idx,
     const vector<DST_PTR_T>&    selection_dst_ptr,
     const vector< pair<hsize_t,hsize_t> >& src_idx_ranges,
     const vector< pair<string,AttrKind> >& edge_attr_info,
     data::NamedAttrVal&                 edge_attr_values
     )
    {
      herr_t ierr = 0;
      vector<NODE_IDX_T> src_vec, dst_vec;

      for (size_t j = 0; j < edge_attr_info.size(); j++)
        {
          string attr_name   = edge_attr_info[j].first;
          AttrKind attr_kind = edge_attr_info[j].second;
          ierr = read_edge_attribute_selection(comm, file_name, src_pop_name, dst_pop_name,
                                               name_space, attr_name, edge_base, edge_count,
                                               selection_dst_idx, selection_dst_ptr, src_idx_ranges,
                                               attr_kind, edge_attr_values);
          throw_assert_nomsg (ierr >= 0);
        }

      return ierr;
    }

  }
}
