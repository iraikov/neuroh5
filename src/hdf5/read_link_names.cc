
#include "read_link_names.hh"

#include "debug.hh"

#include <cassert>
#include <cstdint>
#include <cstring>

using namespace std;

#define MAX_NAME_LEN 1024

namespace ngh5
{
  namespace io
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

        assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS);
        assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS);

        hsize_t num_links=0;

        if (rank == 0)
          {
            hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            assert(file >= 0);

            hid_t grp = H5Gopen2(file, path.c_str(), H5P_DEFAULT);
            assert(grp >= 0);

            H5G_info_t info;
            assert(H5Gget_info(grp, &info) >= 0);
            num_links = info.nlinks;

            hsize_t idx = 0;
            assert(H5Literate(grp, H5_INDEX_NAME, H5_ITER_NATIVE, &idx,
                              &link_iterate_cb, (void*)&names ) >= 0);
            assert(num_links == names.size());
            assert(H5Gclose(grp) >= 0);
            assert(H5Fclose(file) >= 0);
          }

        assert(MPI_Bcast(&num_links, 1, MPI_UINT64_T, 0, comm) == MPI_SUCCESS);
        DEBUG("num_links = ",num_links,"\n");

        vector<uint64_t> link_name_lengths(num_links);

        size_t link_names_total_length = 0;

        if (rank == 0)
          {
            for (size_t i = 0; i < names.size(); ++i)
              {
                DEBUG("Link ",i," is named ",names[i],"\n");
                size_t len = names[i].size();
                link_name_lengths[i] = len;
                link_names_total_length += len;
              }
          }
        else
          {
            names.resize(num_links);
          }

        DEBUG("link_names_total_length = ",link_names_total_length,"\n");
        // Broadcast link name lengths
        assert(MPI_Bcast(&link_names_total_length, 1, MPI_UINT64_T, 0, comm)
               == MPI_SUCCESS);
        assert(MPI_Bcast(&link_name_lengths[0], num_links, MPI_UINT64_T, 0,
                         comm) == MPI_SUCCESS);

        // Broadcast link names
        size_t offset = 0;
        char* link_names_buf = new char [link_names_total_length];
        assert(link_names_buf != NULL);

        if (rank == 0)
          {
            for (size_t i = 0; i < num_links; ++i)
              {
                memcpy(link_names_buf+offset, names[i].c_str(),
                       link_name_lengths[i]);
                offset = offset + link_name_lengths[i];
              }
          }

        assert(MPI_Bcast(link_names_buf, link_names_total_length, MPI_BYTE, 0,
                         comm) == MPI_SUCCESS);

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
}
