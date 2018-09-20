#ifndef HDF5_READ_TEMPLATE
#define HDF5_READ_TEMPLATE

#include <hdf5.h>

#include <cassert>
#include <vector>
#include <utility>

#include "exists_dataset.hh"


namespace neuroh5
{
  namespace hdf5
  {
  
    template<class T>
    herr_t read
    (
     hid_t              loc,
     const std::string& name,
     const hsize_t&     start,
     const hsize_t&     len,
     hid_t              ntype,
     std::vector<T>&    v,
     hid_t rapl
     )
    {
      herr_t ierr = 0;

      ierr = exists_dataset (loc, name.c_str());
      if (ierr > 0)
	{
	  hid_t mspace = H5Screate_simple(1, &len, NULL);
	  assert(mspace >= 0);
	  ierr = H5Sselect_all(mspace);
	  assert(ierr >= 0);

	  hid_t dset = H5Dopen(loc, name.c_str(), H5P_DEFAULT);
	  assert(dset >= 0);
    
	  // make hyperslab selection
	  hid_t fspace = H5Dget_space(dset);
	  assert(fspace >= 0);
	  hsize_t one = 1;
	  if (len > 0)
	    {
	      ierr = H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL, &one, &len);
	    }
	  else
	    {
	      ierr = H5Sselect_none(fspace);
	    }
	  assert(ierr >= 0);
	  
	  ierr = H5Dread(dset, ntype, mspace, fspace, rapl, &v[0]);
	  assert(ierr >= 0);
	  
	  assert(H5Dclose(dset) >= 0);
	  assert(H5Sclose(fspace) >= 0);
	  assert(H5Sclose(mspace) >= 0);
	}
    
      return ierr;
    }

    
    template<class T>
    herr_t read_selection
    (
     hid_t              loc,
     const std::string& name,
     hid_t              ntype,
     const std::vector< std::pair<hsize_t,hsize_t> >& ranges,
     std::vector<T>&    v,
     hid_t rapl
     )
    {
      hsize_t len = 0;

      vector <hsize_t> coords;
      for ( const std::pair<hsize_t,hsize_t> &range : ranges )
        {
          hsize_t start = range.first;
          hsize_t count = range.second;

          for (hsize_t i = start; i<start+count; ++i)
            {
              coords.push_back(i);
            }
          
          len += count;
        }
      assert(coords.size() == len);
      herr_t ierr = 0;

      ierr = exists_dataset (loc, name.c_str());
      if (ierr > 0)
	{

	  hid_t mspace = H5Screate_simple(1, &len, NULL);
	  assert(mspace >= 0);
	  ierr = H5Sselect_all(mspace);
	  assert(ierr >= 0);

	  hid_t dset = H5Dopen(loc, name.c_str(), H5P_DEFAULT);
	  assert(dset >= 0);
	  
	  // make hyperslab selection
	  hid_t fspace = H5Dget_space(dset);
	  assert(fspace >= 0);
	  
	  if (len > 0)
	    {
	      ierr = H5Sselect_elements (fspace, H5S_SELECT_SET, len, (const hsize_t *)&coords[0]);
	    }
	  else
	    {
	      ierr = H5Sselect_none(fspace);
	    }
	  assert(ierr >= 0);

          v.resize(len);
	  ierr = H5Dread(dset, ntype, mspace, fspace, rapl, &v[0]);
	  assert(ierr >= 0);
	  
	  assert(H5Dclose(dset) >= 0);
	  assert(H5Sclose(fspace) >= 0);
	  assert(H5Sclose(mspace) >= 0);
	}
    
      return ierr;
    }

    
  }
}

#endif
