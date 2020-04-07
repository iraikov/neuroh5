
#include "hdf5.h"

#include <cassert>
#include <string>
#include <vector>

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
     const bool rdwr = false
     )
    {
      hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
      assert(fapl >= 0);

      if (collective)
        {
          assert(H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL) >= 0);
        }
      
      hid_t file;

      if (rdwr)
        {
          file = H5Fopen(file_name.c_str(), H5F_ACC_RDWR, fapl);
        }
      else
        {
          file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, fapl);
        }
      
      assert(file >= 0);
      
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
      char name[1024];
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
         
      return howmany;
    }
        
    
  }
}

