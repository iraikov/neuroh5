#ifndef READ_EDGE_ATTRIBUTES_HH
#define READ_EDGE_ATTRIBUTES_HH

#include "neuroh5_types.hh"
#include "infer_datatype.hh"

#include "hdf5.h"
#include "mpi.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

namespace neuroh5
{
  namespace graph
  {
    //////////////////////////////////////////////////////////////////////////
    template <typename T>
    void read_edge_attribute
    (
     hid_t                    loc,
     const std::string&       path,
     std::vector<NODE_IDX_T>& edge_id,
     std::vector<T>&          value
     )
    {
      // read node IDs

      hid_t dset = H5Dopen(loc, (path + "/edge_id").c_str(), H5P_DEFAULT);
      assert(dset >= 0);
      hid_t fspace = H5Dget_space(dset);
      assert(fspace >= 0);
      hssize_t size = H5Sget_simple_extent_npoints(fspace);
      assert(size > 0);
      edge_id.resize(size);
      assert(H5Dread(dset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     &edge_id[0]) >= 0);
      assert(H5Sclose(fspace) >= 0);
      assert(H5Dclose(dset) >= 0);

      // read values

      dset = H5Dopen(loc, (path + "/value").c_str(), H5P_DEFAULT);
      assert(dset >= 0);
      fspace = H5Dget_space(dset);
      assert(fspace >= 0);
      size = H5Sget_simple_extent_npoints(fspace);
      assert(size > 0 && 2*size == edge_id.size());
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

    //////////////////////////////////////////////////////////////////////////
    template <typename T>
    void read_edge_attribute_bal
    (
     hid_t                     loc,
     const std::string&        path,
     std::vector<NODE_IDX_T>&  edge_id,
     std::vector<T>&           value
     )
    {
      // get a file handle and retrieve the MPI info
      hid_t file = H5Iget_file_id(loc);
      assert(file >= 0);

      MPI_Comm comm;
      MPI_Info info;
      hid_t fapl = H5Fget_access_plist(file);
      assert(H5Pget_fapl_mpio(fapl, &comm, &info) >= 0);

      int size, rank;
      assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS);
      assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS);

      // divide up the work
      hid_t dset = H5Dopen(loc, (path + "/edge_id").c_str(), H5P_DEFAULT);
      assert(dset >= 0);
      hid_t fspace = H5Dget_space(dset);
      assert(fspace >= 0);
      hssize_t dsize = H5Sget_simple_extent_npoints(fspace);
      assert(dsize > 0 && dsize%2 == 0);
      dsize /= 2;

      hsize_t part = dsize / size;
      hsize_t start = rank*2*part, stop = (rank+1)*2*part;
      if (rank == size-1)
        {
          stop = dsize;
        }
      hsize_t block = stop - start, one = 1;

      // create dataspaces and selections
      hid_t mspace = H5Screate_simple(1, &block, &block);
      assert(mspace >= 0);
      assert(H5Sselect_all(mspace) >= 0);
      assert(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                 &one, &block) >= 0);

      // read edge IDs
      edge_id.resize(block);
      assert(H5Dread(dset, H5T_NATIVE_UINT32, mspace, fspace, H5P_DEFAULT,
                     &edge_id[0]) >= 0);
      assert(H5Dclose(dset) >= 0);
      assert(H5Sclose(mspace) >= 0);
      assert(H5Sclose(fspace) >= 0);

      // get the values
      dset = H5Dopen(loc, (path + "/value").c_str(), H5P_DEFAULT);
      assert(dset >= 0);

      fspace = H5Dget_space(dset);
      assert(fspace >= 0);
      hssize_t dsize1 = H5Sget_simple_extent_npoints(fspace);
      assert(dsize > 0 && dsize1 == dsize/2);

      start = rank*part;
      stop = (rank+1)*part;
      if (rank == size-1)
        {
          stop = dsize1;
        }
      block = stop - start;

      // create dataspaces and selections
      mspace = H5Screate_simple(1, &block, &block);
      assert(mspace >= 0);
      assert(H5Sselect_all(mspace) >= 0);
      assert(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                 &one, &block) >= 0);

      // figure the type
      T dummy;
      hid_t ftype = infer_datatype(dummy);
      assert(ftype >= 0);
      hid_t mtype = H5Tget_native_type(ftype, H5T_DIR_ASCEND);
      assert(mtype >= 0);
      value.resize(block);
      assert(H5Dread(dset, mtype, mspace, fspace, H5P_DEFAULT,
                     &value[0]) >= 0);

      assert(H5Tclose(mtype) >= 0);
      assert(H5Tclose(ftype) >= 0);
      assert(H5Sclose(mspace) >= 0);
      assert(H5Sclose(fspace) >= 0);
      assert(H5Dclose(dset) >= 0);
    }
  }
}

#endif
