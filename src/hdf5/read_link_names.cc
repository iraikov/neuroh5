
#include <cstdint>
#include <cstring>

#include "debug.hh"

#include "read_link_names.hh"
#include "throw_assert.hh"

using namespace std;

#define MAX_NAME_LEN 1024

namespace neuroh5
{
  namespace hdf5
  {
    //////////////////////////////////////////////////////////////////////////
    herr_t link_iterate_cb
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

    //////////////////////////////////////////////////////////////////////////
    void read_link_names
    (
     MPI_Comm        comm,
     const string&   file_name,
     const string&   path,
     vector<string>& names
     )
    {
      int rank, size;

      throw_assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS,
                   "hdf5::read_link_names: unable to obtain MPI communicator size");
      throw_assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS,
                   "hdf5::read_link_names: unable to obtain MPI communicator size");

      hsize_t num_links=0;

      if (rank == 0)
        {
          hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
          throw_assert(file >= 0, "hdf5::read_link_names: unable to open file " << file_name);

          hid_t grp = H5Gopen2(file, path.c_str(), H5P_DEFAULT);
          throw_assert(grp >= 0, "hdf5::read_link_names: unable to open group " << path);

          H5G_info_t info;
          throw_assert(H5Gget_info(grp, &info) >= 0,
                       "hdf5::read_link_names: unable to get info for group " << path);
          num_links = info.nlinks;

          hsize_t idx = 0;
          throw_assert(H5Literate(grp, H5_INDEX_NAME, H5_ITER_NATIVE, &idx,
                                  &link_iterate_cb, (void*)&names ) >= 0,
                       "hdf5::read_link_names: unable to iterate over objects of group " << path);
          throw_assert(num_links == names.size(),
                       "hdf5::read_link_names: unable to obtain object names for all links in group " << path);
          throw_assert(H5Gclose(grp) >= 0,
                       "hdf5::read_link_names: unable to close group " << path);
          throw_assert(H5Fclose(file) >= 0,
                       "hdf5::read_link_names: unable to close file " << file_name);
        }

    throw_assert(MPI_Bcast(&num_links, 1, MPI_UINT64_T, 0, comm) == MPI_SUCCESS,
                 "hdf5::read_link_names: unable to complete MPI broadcast operation");

    vector<uint64_t> link_name_lengths(num_links);
    
    size_t link_names_total_length = 0;

      if (rank == 0)
        {
          for (size_t i = 0; i < names.size(); ++i)
            {
              size_t len = names[i].size();
              link_name_lengths[i] = len;
              link_names_total_length += len;
            }
        }
      else
        {
          names.resize(num_links);
        }

      // Broadcast link name lengths
    throw_assert(MPI_Bcast(&link_names_total_length, 1, MPI_UINT64_T, 0, comm)
                 == MPI_SUCCESS,
                 "hdf5::read_link_names: unable to complete MPI broadcast operation");
    throw_assert(MPI_Bcast(&link_name_lengths[0], num_links, MPI_UINT64_T, 0, comm)
                 == MPI_SUCCESS,
                 "hdf5::read_link_names: unable to complete MPI broadcast operation");
    
    // Broadcast link names
    size_t offset = 0;
    char* link_names_buf = new char [link_names_total_length];
    throw_assert(link_names_buf != NULL,
                 "hdf5::read_link_names: unable to allocate memory for link names");

      if (rank == 0)
        {
          for (size_t i = 0; i < num_links; ++i)
            {
              memcpy(link_names_buf+offset, names[i].c_str(),
                     link_name_lengths[i]);
              offset = offset + link_name_lengths[i];
            }
        }

    throw_assert(MPI_Bcast(link_names_buf, link_names_total_length, MPI_BYTE, 0,
                           comm) == MPI_SUCCESS,
                 "hdf5::read_link_names: unable to complete MPI broadcast operation");                 

      if (rank != 0)
        {
          // Copy link names into names
          char link_name[MAX_NAME_LEN];
          offset = 0;
          for (size_t i = 0; i < num_links; ++i)
            {
              size_t len = link_name_lengths[i];
              memcpy(link_name, link_names_buf+offset, len);
              link_name[len] = '\0';
              names[i] = string((const char*)link_name);
              offset = offset + len;
            }
        }

      delete [] link_names_buf;
    }
  }
}
