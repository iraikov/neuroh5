#ifndef NODE_ATTRIBUTES_HH
#define NODE_ATTRIBUTES_HH

#include "infer_datatype.hh"
#include "ngh5_types.hh"

#include "hdf5.h"
#include "mpi.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace ngh5
{
  namespace io
  {
    namespace hdf5
    {
      template <typename T>
      void write_node_attribute
      (
       hid_t                           loc,
       const std::string&              path,
       const std::vector<NODE_IDX_T>&  node_id,
       const std::vector<T>&           value
       )
      {
        assert(node_id.size() == value.size());

        // get a file handle and retrieve the MPI info
        hid_t file = H5Iget_file_id(loc);
        assert(file >= 0);

        MPI_Comm comm;
        MPI_Info info;
        assert(H5Pget_fapl_mpio(file, &comm, &info) >= 0);

        int size, rank;
        assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS);
        assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS);

        uint32_t my_count = (uint32_t)node_id.size();
        std::vector<uint32_t> all_counts(size);
        assert(MPI_Allgather(&my_count, 1, MPI_UINT32_T, &all_counts[0], 1,
                             MPI_UINT32_T, comm) == MPI_SUCCESS);

        // calculate the total dataset size and the offset of my piece
        hsize_t start = 0, total = 0, count = 1, block = my_count;
        for (size_t p = 0; p < size; ++p)
          {
            if (p < rank)
              {
                start += (hsize_t) all_counts[p];
              }
            total += (hsize_t) all_counts[p];
          }

        // create dataspaces and selections
        hid_t mspace = H5Screate_simple(1, &block, &block);
        assert(mspace >= 0);
        assert(H5Sselect_all(mspace) >= 0);
        hid_t fspace = H5Screate_simple(1, &total, &total);
        assert(fspace >= 0);
        assert(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                   &count, &block) >= 0);

        // figure the type

        T dummy;
        hid_t ftype = infer_datatype(dummy);
        assert(ftype >= 0);
        hid_t mtype = H5Tget_native_type(ftype, H5T_DIR_ASCEND);
        assert(mtype >= 0);

        // Ready to roll!

        hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
        assert(lcpl >= 0);
        assert(H5Pset_create_intermediate_group(lcpl, 1) >= 0);

        hid_t dset = H5Dcreate(loc, (path + "/node_id").c_str(), H5T_STD_U32LE,
                               fspace, lcpl, H5P_DEFAULT, H5P_DEFAULT);
        assert(dset >= 0);
        assert(H5Dwrite(dset, H5T_NATIVE_UINT32, mspace, fspace, H5P_DEFAULT,
                        &node_id[0]) >= 0);
        assert(H5Dclose(dset) >= 0);

        dset = H5Dcreate(loc, (path + "/value").c_str(), ftype, fspace, lcpl,
                         H5P_DEFAULT, H5P_DEFAULT);
        assert(dset >= 0);
        assert(H5Dwrite(dset, mtype, mspace, fspace, H5P_DEFAULT, &value[0])
               >= 0);
        assert(H5Dclose(dset) >= 0);

        // clean house

        assert(H5Pclose(lcpl) >= 0);
        assert(H5Tclose(mtype) >= 0);
        assert(H5Tclose(ftype) >= 0);
        assert(H5Sclose(fspace) >= 0);
        assert(H5Sclose(mspace) >= 0);
      }

      template <typename T>
      void read_node_attribute
      (
       hid_t                     loc,
       const std::string&        path,
       std::vector<NODE_IDX_T>&  node_id,
       std::vector<T>&           value
       )
      {
        // read node IDs

        hid_t dset = H5Dopen(loc, (path + "/node_id").c_str(), H5P_DEFAULT);
        assert(dset >= 0);
        hid_t fspace = H5Dget_space(dset);
        assert(fspace >= 0);
        hssize_t size = H5Sget_simple_extent_npoints(fspace);
        assert(size > 0);
        node_id.resize(size);
        assert(H5Dread(dset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                       &node_id[0]) >= 0);
        assert(H5Sclose(fspace) >= 0);
        assert(H5Dclose(dset) >= 0);

        // read values

        dset = H5Dopen(loc, (path + "/value").c_str(), H5P_DEFAULT);
        assert(dset >= 0);
        fspace = H5Dget_space(dset);
        assert(fspace >= 0);
        size = H5Sget_simple_extent_npoints(fspace);
        assert(size > 0 && size == node_id.size());
        value.resize(size);

        hid_t ftype = H5Dget_type(dset);
        assert(ftype >= 0);
        hid_t ntype = H5Tget_native_type(ftype, H5T_DIR_ASCEND);
        assert(H5Dread(dset, ntype, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                       &value[0]) >= 0);
        assert(H5Tclose(ntype) >= 0);
        assert(H5Tclose(ftype) >= 0);
        assert(H5Sclose(fspace) >= 0);
        assert(H5Dclose(dset) >= 0);
      }
    }
  }
}

#endif