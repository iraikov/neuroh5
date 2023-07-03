
#include <hdf5.h>
#include <string>
#include <vector>
#include "throw_assert.hh"

namespace neuroh5
{
  namespace hdf5
  {

    /*****************************************************************************
     * Routines for opening and closing NeuroH5 for reading and writing.
     *****************************************************************************/
    hid_t open_file
    (
     MPI_Comm comm,
     const std::string& file_name,
     const bool collective = false,
     const bool rdwr = false,
     const size_t cache_size = 1*1024*1024
     )
    {
      hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
      throw_assert_nomsg(fapl >= 0);

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

      
#ifdef HDF5_IS_PARALLEL
      if (collective)
        {
          throw_assert_nomsg(H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL) >= 0);
        }
#endif
      
      hid_t file;

      if (rdwr)
        {
          file = H5Fopen(file_name.c_str(), H5F_ACC_RDWR, fapl);
        }
      else
        {
          file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, fapl);
        }
      
      throw_assert_nomsg(file >= 0);

      throw_assert(H5Pclose(fapl) == 0,
                   "open_file: unable to close HDF5 file properties list");
      
      return file;
    }

    herr_t close_file
    (
     hid_t& file
     )
    {
      herr_t status = H5Fclose(file);
      return status;
    }

    int what_is_open(hid_t fid, int mask) 
    {
      ssize_t cnt;
      int howmany;
      int i;
      H5I_type_t ot;
      hid_t anobj;
      hid_t *objs;
      char *name = new char [1024];
      herr_t status;

      cnt = H5Fget_obj_count(fid, mask);

      if (cnt <= 0) return cnt;

      objs = (hid_t *)malloc(sizeof(hid_t) * cnt);

      printf("%lu object(s) open\n", cnt);

      howmany = H5Fget_obj_ids(fid, mask, cnt, objs);
      
      printf("open objects:\n");

      if (howmany > 0)
        {
          for (i = 0; i < howmany; i++ ) 
            {
            anobj = *objs++;
            if (!H5Iis_valid(anobj))
              continue;
            ot = H5Iget_type(anobj);
            status = H5Iget_name(anobj, name, 1023);
            name[1023] = 0;
            switch (ot)
              {
              case H5I_FILE:
                printf(" %d: type FILE, name %s\n",i,name);
                break;
              case H5I_GROUP:
                printf(" %d: type GROUP, name %s\n",i,name);
                break;
              case H5I_DATATYPE:
                printf(" %d: type DATATYPE, name %s\n",i,name);
                break;
              case H5I_DATASPACE:
                printf(" %d: type DATASPACE, name %s\n",i,name);
                break;
              case H5I_DATASET:
                printf(" %d: type DATASET, name %s\n",i,name);
                break;
              case H5I_ATTR:
                printf(" %d: type ATTR, name %s\n",i,name);
                break;
              default:
                printf(" %d: type UNKNOWN, name %s\n",i,name);
                break;
              }
          }
        }
         
      delete [] name;
      return howmany;
    }
        
    
  }
}

