
#include "attributes.hh"

#include <cassert>
#include <iostream>

using namespace std;

namespace ngh5
{
  string ngh5_edge_attr_path (const char *dsetname, const char *attr_name)
  {
    string result;
    result = string("/Projections/") + dsetname + "/Attributes/Edge/" + attr_name;
    return result;
  }

  //////////////////////////////////////////////////////////////////////////////
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

    vector< pair<string,hid_t> >* ptr = (vector< pair<string,hid_t> >*) op_data;
    ptr->push_back(make_pair(name, ftype));

    assert(H5Dclose(dset) >= 0);

    return 0;
  }

  //////////////////////////////////////////////////////////////////////////////
  herr_t get_edge_attributes
  (
   const char *                  in_file_name,
   const string&                 in_projName,
   vector< pair<string,hid_t> >& out_attributes
   )
  {
    hid_t in_file;
    herr_t ierr;

    in_file = H5Fopen(in_file_name, H5F_ACC_RDONLY, H5P_DEFAULT);
    assert(in_file >= 0);
    out_attributes.clear();

    // TODO: Don't hardcode this!
     string path = "/Projections/" + in_projName + "/Attributes/Edge";

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


  
  herr_t read_edge_attributes
  (
   MPI_Comm            comm,
   const char*         fname, 
   const char*         dsetname, 
   const char*         attrname, 
   const DST_PTR_T     edge_base,
   const DST_PTR_T     edge_count,
   const hid_t         attr_h5type,
   EdgeNamedAttr      &attr_values
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
    
    file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    assert(file >= 0);
    
    if (edge_count > 0)
      {
        hsize_t one = 1;
        
        hid_t mspace = H5Screate_simple(1, &block, NULL);
        assert(mspace >= 0);
        ierr = H5Sselect_all(mspace);
        assert(ierr >= 0);
        
        hid_t dset = H5Dopen2(file, ngh5_edge_attr_path(dsetname, attrname).c_str(), H5P_DEFAULT);
        assert(dset >= 0);
        
        // make hyperslab selection
        hid_t fspace = H5Dget_space(dset);
        assert(fspace >= 0);
        ierr = H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &base, NULL, &one, &block);
        assert(ierr >= 0);
        
        switch (H5Tget_class(attr_h5type))
          {
          case H5T_INTEGER:
            if (attr_size == 32)
              {
                ierr = H5Dread(dset, attr_h5type, mspace, fspace, H5P_DEFAULT, &attr_values_uint32[0]);
                attr_values.insert<uint32_t>(std::string(attrname), attr_values_uint32);
              }
            else if (attr_size == 16)
              {
                ierr = H5Dread(dset, attr_h5type, mspace, fspace, H5P_DEFAULT, &attr_values_uint16[0]);
                attr_values.insert<uint16_t>(std::string(attrname), attr_values_uint16);
              }
            else if (attr_size == 8)
              {
                ierr = H5Dread(dset, attr_h5type, mspace, fspace, H5P_DEFAULT, &attr_values_uint8[0]);
                attr_values.insert<uint8_t>(std::string(attrname), attr_values_uint8);
              }
            else
              {
                throw std::runtime_error("Unsupported integer attribute size");
              };
            break;
          case H5T_FLOAT:
            ierr = H5Dread(dset, attr_h5type, mspace, fspace, H5P_DEFAULT, &attr_values_float[0]);
            attr_values.insert<float>(std::string(attrname), attr_values_float);
            break;
          case H5T_ENUM:
             if (attr_size == 8)
              {
                ierr = H5Dread(dset, attr_h5type, mspace, fspace, H5P_DEFAULT, &attr_values_uint8[0]);
                attr_values.insert<uint8_t>(std::string(attrname), attr_values_uint8);    
              }
            else
              {
                throw std::runtime_error("Unsupported enumerated attribute size");
              };
            break;
          default:
            throw std::runtime_error("Unsupported attribute type");
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

}
