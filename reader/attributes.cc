
#include "attributes.hh"

#include <cassert>
#include <iostream>

using namespace std;

namespace ngh5
{
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
   hid_t&                        in_file,
   const string&                 in_projName,
   vector< pair<string,hid_t> >& out_attributes
   )
  {
    herr_t ierr;

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

    return ierr;
  }
}
