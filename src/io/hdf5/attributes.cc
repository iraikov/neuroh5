// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file attributes.cc
///
///  Functions for node and edge attribute discovery.
///
///  Copyright (C) 2016 Project Neurograph.
//==============================================================================

#include "attributes.hh"

#include "hdf5_path_names.hh"

#include <cassert>
#include <iostream>

using namespace std;

namespace ngh5
{
  namespace io
  {
    namespace hdf5
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
        assert(ftype >= 0);

        vector< pair<string,hid_t> >* ptr =
          (vector< pair<string,hid_t> >*) op_data;
        ptr->push_back(make_pair(name, ftype));

        assert(H5Dclose(dset) >= 0);

        return 0;
      }

      /////////////////////////////////////////////////////////////////////////
      herr_t get_edge_attributes
      (
       const string&                 file_name,
       const string&                 proj_name,
       vector< pair<string,hid_t> >& out_attributes
       )
      {
        hid_t in_file;
        herr_t ierr;

        in_file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        assert(in_file >= 0);
        out_attributes.clear();

        string path = io::hdf5::edge_attribute_path(proj_name);

        // TODO: Be more gentle if the group is not found!
        hid_t grp = H5Gopen2(in_file, path.c_str(), H5P_DEFAULT);
        assert(grp >= 0);

        hsize_t idx = 0;
        ierr = H5Literate(grp, H5_INDEX_NAME, H5_ITER_NATIVE, &idx,
                          &edge_attribute_cb, (void*) &out_attributes);

        assert(H5Gclose(grp) >= 0);
        ierr = H5Fclose(in_file);

        return ierr;
      }

      /////////////////////////////////////////////////////////////////////////
      herr_t num_edge_attributes
      (
       const vector< pair<string,hid_t> >& attributes,
       vector <size_t> &num_attrs
       )
      {
        herr_t ierr = 0;
        num_attrs.resize(4);
        for (size_t i = 0; i < attributes.size(); i++)
          {
            hid_t attr_h5type = attributes[i].second;
            size_t attr_size = H5Tget_size(attr_h5type);
            switch (H5Tget_class(attr_h5type))
              {
              case H5T_INTEGER:
                if (attr_size == 32)
                  {
                    num_attrs[3]++;
                  }
                else if (attr_size == 16)
                  {
                    num_attrs[2]++;
                  }
                else if (attr_size == 8)
                  {
                    num_attrs[1]++;
                  }
                else
                  {
                    throw runtime_error("Unsupported integer attribute size");
                  };
                break;
              case H5T_FLOAT:
                num_attrs[0]++;
                break;
              case H5T_ENUM:
                if (attr_size == 8)
                  {
                    num_attrs[1]++;
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

      /////////////////////////////////////////////////////////////////////////
      herr_t read_edge_attributes
      (
       MPI_Comm              comm,
       const string&         file_name,
       const string&         proj_name,
       const string&         attr_name,
       const DST_PTR_T       edge_base,
       const DST_PTR_T       edge_count,
       const hid_t           attr_h5type,
       model::EdgeNamedAttr& attr_values
       )
      {
        hid_t file;
        herr_t ierr = 0;
        hsize_t block = edge_count, base = edge_base;
        vector <float>    attr_values_float;
        vector <uint16_t> attr_values_uint16;
        vector <uint32_t> attr_values_uint32;
        vector <uint8_t>  attr_values_uint8;
        size_t attr_size = H5Tget_size(attr_h5type);

        file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        assert(file >= 0);

        if (edge_count > 0)
          {
            hsize_t one = 1;

            hid_t mspace = H5Screate_simple(1, &block, NULL);
            assert(mspace >= 0);
            ierr = H5Sselect_all(mspace);
            assert(ierr >= 0);

            hid_t dset = H5Dopen2
              (file, io::hdf5::edge_attribute_path(proj_name, attr_name).c_str(),
               H5P_DEFAULT);
            assert(dset >= 0);

            // make hyperslab selection
            hid_t fspace = H5Dget_space(dset);
            assert(fspace >= 0);
            ierr = H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &base, NULL, &one,
                                       &block);
            assert(ierr >= 0);

            switch (H5Tget_class(attr_h5type))
              {
              case H5T_INTEGER:
                if (attr_size == 32)
                  {
                    attr_values_uint32.resize(edge_count);
                    ierr = H5Dread(dset, attr_h5type, mspace, fspace, H5P_DEFAULT,
                                   &attr_values_uint32[0]);
                    attr_values.insert(string(attr_name), attr_values_uint32);
                  }
                else if (attr_size == 16)
                  {
                    attr_values_uint16.resize(edge_count);
                    ierr = H5Dread(dset, attr_h5type, mspace, fspace, H5P_DEFAULT,
                                   &attr_values_uint16[0]);
                    attr_values.insert(string(attr_name), attr_values_uint16);
                  }
                else if (attr_size == 8)
                  {
                    attr_values_uint8.resize(edge_count);
                    ierr = H5Dread(dset, attr_h5type, mspace, fspace, H5P_DEFAULT,
                                   &attr_values_uint8[0]);
                    attr_values.insert(string(attr_name), attr_values_uint8);
                  }
                else
                  {
                    throw runtime_error("Unsupported integer attribute size");
                  };
                break;
              case H5T_FLOAT:
                attr_values_float.resize(edge_count);
                ierr = H5Dread(dset, attr_h5type, mspace, fspace, H5P_DEFAULT,
                               &attr_values_float[0]);
                attr_values.insert(string(attr_name), attr_values_float);
                break;
              case H5T_ENUM:
                if (attr_size == 8)
                  {
                    attr_values_uint8.resize(edge_count);
                    ierr = H5Dread(dset, attr_h5type, mspace, fspace, H5P_DEFAULT,
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

            assert(ierr >= 0);

            assert(H5Sclose(fspace) >= 0);
            assert(H5Dclose(dset) >= 0);
            assert(H5Sclose(mspace) >= 0);
          }

        assert(H5Fclose(file) >= 0);

        return ierr;
      }

      /////////////////////////////////////////////////////////////////////////
      int read_all_edge_attributes
      (
       MPI_Comm                            comm,
       const string&                       file_name,
       const string&                       prj_name,
       const DST_PTR_T                     edge_base,
       const DST_PTR_T                     edge_count,
       const vector< pair<string,hid_t> >& edge_attr_info,
       model::EdgeNamedAttr&               edge_attr_values
       )
      {
        int ierr = 0;
        vector<NODE_IDX_T> src_vec, dst_vec;

        for (size_t j = 0; j < edge_attr_info.size(); j++)
          {
            string attr_name   = edge_attr_info[j].first;
            hid_t  attr_h5type = edge_attr_info[j].second;
            assert ((ierr = read_edge_attributes(comm, file_name, prj_name,
                                                 attr_name, edge_base, edge_count,
                                                 attr_h5type, edge_attr_values))
                    >= 0);
          }

        return ierr;
      }
    }
  }
}
