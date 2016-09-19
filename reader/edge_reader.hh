#ifndef EDGE_READER_HH
#define EDGE_READER_HH

#include "ngh5types.hh"

#include "mpi.h"

#include "hdf5.h"

#include <map>
#include <vector>

std::string ngh5_edge_attr_path (const char *dsetname, const char *attr_name) 
{
  std::string result;
  result = std::string("/Projections/") + dsetname + "/Attributes/Edge/" + attr_name;
  return result;
}

using namespace std;

template<typename AttrType>
herr_t read_edge_attributes
(
 MPI_Comm            comm,
 const char*         fname, 
 const char*         dsetname, 
 const char*         attrname, 
 const DST_PTR_T     edge_base,
 const DST_PTR_T     edge_count,
 const hid_t         attr_h5type,
 std::vector<AttrType>&   attr_values
 );


/*****************************************************************************
 * Read edge attributes
 *****************************************************************************/


template<typename AttrType>
herr_t read_edge_attributes
(
 MPI_Comm            comm,
 const char*         fname, 
 const char*         dsetname, 
 const char*         attrname, 
 const DST_PTR_T     edge_base,
 const DST_PTR_T     edge_count,
 const hid_t         attr_h5type,
 vector<AttrType>&   attr_values
 )
{
  hid_t file;
  herr_t ierr = 0;
  hsize_t block = edge_count, base = edge_base;
  
  file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
  assert(file >= 0);

  if (edge_count > 0)
    {
      hsize_t one = 1;

      // allocate buffer and memory dataspace
      attr_values.resize(edge_count);
      assert(attr_values.size() > 0);
          
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
          
      ierr = H5Dread(dset, attr_h5type, mspace, fspace, H5P_DEFAULT, &attr_values[0]);
      assert(ierr >= 0);
      
      assert(H5Sclose(fspace) >= 0);
      assert(H5Dclose(dset) >= 0);
      assert(H5Sclose(mspace) >= 0);
    }

  assert(H5Fclose(file) >= 0);

  return ierr;
}


#endif
