
#include "write_connectivity.hh"
#include "hdf5_types.hh"
#include "hdf5_path_names.hh"

#include <algorithm>
#include <cassert>
#include <map>

using namespace std;

namespace ngh5
{
  namespace io
  {
    namespace hdf5
    {
      void write_connectivity
      (
       hid_t                     file,
       const string&             projection_name,
       const POP_IDX_T&          src_pop_idx,
       const POP_IDX_T&          dst_pop_idx,
       const NODE_IDX_T&         src_start,
       const NODE_IDX_T&         src_end,
       const NODE_IDX_T&         dst_start,
       const NODE_IDX_T&         dst_end,
       const vector<NODE_IDX_T>& edges
       )
      {
        // do a sanity check on the input
        assert(src_start < src_end);
        assert(dst_start < dst_end);
        assert(edges.size()%2 == 0);

        uint64_t num_edges = edges.size()/2;
        assert(num_edges > 0);

        // build destination->source(s) map as a side-effect
        map<NODE_IDX_T, vector<NODE_IDX_T> > dst_src_map;
        for (NODE_IDX_T inode = dst_start; inode < dst_end; inode++)
          {
            dst_src_map.insert(make_pair(inode, vector<NODE_IDX_T>()));
          }

        map<NODE_IDX_T, vector<NODE_IDX_T> >::iterator iter;

        for (size_t i = 1; i < edges.size(); i += 2)
          {
            // all source/destination node IDs must be in range
            assert(dst_start <= edges[i] && edges[i] < dst_end);
            assert(src_start <= edges[i-1] && edges[i-1] < src_end);

            iter = dst_src_map.find(edges[i] - dst_start);
            assert (iter != dst_src_map.end());
            iter->second.push_back(edges[i-1] - src_start);
          }

        uint64_t num_dest = dst_src_map.size();
        assert(num_dest > 0 && num_dest < (dst_end - dst_start + 1));

        // sort the source arrays and create relative destination pointers
        // and source index
        vector<uint64_t> dst_ptr(1, 0);
        vector<uint32_t> src_idx;
        size_t pos = 0;
        for (iter = dst_src_map.begin(); iter != dst_src_map.end(); ++iter)
          {
            sort(iter->second.begin(), iter->second.end());
            dst_ptr.push_back(dst_ptr[pos++] + iter->second.size());
            copy(iter->second.begin(), iter->second.end(),
                 back_inserter(src_idx));
            // save memory
            iter->second.clear();
          }

        // get the I/O communicator
        MPI_Comm comm;
        MPI_Info info;
        hid_t fapl = H5Fget_access_plist(file);
        assert(H5Pget_fapl_mpio(fapl, &comm, &info) >= 0);

        int size, rank;
        assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS);
        assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS);

        assert(H5Pclose(fapl) >= 0);

        // exchange allocation data

        vector<uint64_t> sendbuf_num_dest(size, num_dest);
        vector<uint64_t> recvbuf_num_dest(size);
        assert(MPI_Allgather(&sendbuf_num_dest[0], 1, MPI_UINT64_T,
                             &recvbuf_num_dest[0], 1, MPI_UINT64_T, comm)
               == MPI_SUCCESS);

        vector<uint64_t> sendbuf_num_edge(size, num_edges);
        vector<uint64_t> recvbuf_num_edge(size);
        assert(MPI_Allgather(&sendbuf_num_edge[0], 1, MPI_UINT64_T,
                             &recvbuf_num_edge[0], 1, MPI_UINT64_T, comm)
               == MPI_SUCCESS);

        hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
        assert(lcpl >= 0);
        assert(H5Pset_create_intermediate_group(lcpl, 1) >= 0);

        // write destination block index (= first_node)

        string path = io::hdf5::projection_path_join(projection_name, "/Connectivity/Destination Block Index");
        hsize_t dims = (hsize_t)size, one = 1;
        hid_t fspace = H5Screate_simple(1, &dims, &dims);
        assert(fspace >= 0);
        hid_t dset = H5Dcreate2(file, path.c_str(), NODE_IDX_H5_FILE_T, fspace,
                                lcpl, H5P_DEFAULT, H5P_DEFAULT);
        assert(dset >= 0);
        dims = 1;
        hid_t mspace = H5Screate_simple(1, &dims, &dims);
        assert(mspace >= 0);
        assert(H5Sselect_all(mspace) >= 0);
        hsize_t start = (hsize_t)rank, block = dims;
        assert(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                   &one, &block) >= 0);
        assert(H5Dwrite(dset, NODE_IDX_H5_NATIVE_T, mspace, fspace, H5P_DEFAULT,
                        &dst_start) >= 0);
        assert(H5Dclose(dset) >= 0);
        assert(H5Sclose(mspace) >= 0);
        assert(H5Sclose(fspace) >= 0);

        // write destination block pointer

        path = projection_path_join(projection_name, "/Connectivity/Destination Block Pointer");
        dims = (hsize_t)(size + 1);
        fspace = H5Screate_simple(1, &dims, &dims);
        assert(fspace >= 0);
        dset = H5Dcreate2(file, path.c_str(), DST_BLK_PTR_H5_FILE_T,
                          fspace, lcpl, H5P_DEFAULT, H5P_DEFAULT);
        assert(dset >= 0);

        dims = 1;
        if (rank == size-1) // the last rank writes an extra element
          {
            dims = 2;
          }
        mspace = H5Screate_simple(1, &dims, &dims);
        assert(mspace >= 0);
        assert(H5Sselect_all(mspace) >= 0);
        start = (hsize_t)rank;
        block = dims;
        assert(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                   &one, &block) >= 0);

        vector<uint64_t> dbp(2,0); // only the last rank writes two elements

        for (int p = 0; p < rank; ++p)
          {
            dbp[0] += recvbuf_num_dest[p];
          }

        if (rank == size-1) // last rank writes the total destination count
          {
            dbp[1] = dbp[0] + recvbuf_num_dest[rank];
          }

        assert(H5Dwrite(dset, DST_BLK_PTR_H5_NATIVE_T, mspace, fspace,
                        H5P_DEFAULT, &dbp[0]) >= 0);

        assert(H5Dclose(dset) >= 0);
        assert(H5Sclose(mspace) >= 0);
        assert(H5Sclose(fspace) >= 0);

        // write destination pointers
        // # dest. pointers = number of destinations + 1

        path = projection_path_join(projection_name, "/Connectivity/Destination Pointer");
        dims = 0;
        for (int p = 0; p < size; ++p)
          {
            dims += recvbuf_num_dest[p];
          }
        ++dims; // one extra element

        fspace = H5Screate_simple(1, &dims, &dims);
        assert(fspace >= 0);
        dset = H5Dcreate2(file, path.c_str(), DST_PTR_H5_FILE_T,
                          fspace, lcpl, H5P_DEFAULT, H5P_DEFAULT);
        assert(dset >= 0);

        uint64_t s = 0;
        for (int p = 0; p < rank; ++p)
          {
            s += recvbuf_num_edge[p];
          }

        for (size_t idst = 0; idst < dst_ptr.size(); ++idst)
          {
            dst_ptr[idst] += s;
          }

        if (rank == size-1) // only the last rank writes an additional element
          {
            dst_ptr.back() += recvbuf_num_edge[rank];
          }
        else
          {
            dst_ptr.resize(num_dest);
          }

        dims = (hsize_t) dst_ptr.size();
        mspace = H5Screate_simple(1, &dims, &dims);
        assert(mspace >= 0);
        assert(H5Sselect_all(mspace) >= 0);
        start = (hsize_t)dbp[0];
        block = dims;
        assert(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                   &one, &block) >= 0);

        assert(H5Dwrite(dset, DST_PTR_H5_NATIVE_T, mspace, fspace,
                        H5P_DEFAULT, &dst_ptr[0]) >= 0);

        assert(H5Dclose(dset) >= 0);
        assert(H5Sclose(mspace) >= 0);
        assert(H5Sclose(fspace) >= 0);

        // write source index
        // # source indexes = number of edges

        path = projection_path_join(projection_name, "/Connectivity/Source Index");
        dims = 0;
        for (int p = 0; p < size; ++p)
          {
            dims += recvbuf_num_edge[p];
          }

        fspace = H5Screate_simple(1, &dims, &dims);
        assert(fspace >= 0);
        dset = H5Dcreate2(file, path.c_str(), NODE_IDX_H5_FILE_T,
                          fspace, lcpl, H5P_DEFAULT, H5P_DEFAULT);
        assert(dset >= 0);

        dims = (hsize_t) src_idx.size();
        mspace = H5Screate_simple(1, &dims, &dims);
        assert(mspace >= 0);
        assert(H5Sselect_all(mspace) >= 0);
        start = (hsize_t)dst_ptr[0];
        block = dims;
        assert(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                   &one, &block) >= 0);

        assert(H5Dwrite(dset, NODE_IDX_H5_NATIVE_T, mspace, fspace,
                        H5P_DEFAULT, &src_idx[0]) >= 0);

        assert(H5Dclose(dset) >= 0);
        assert(H5Sclose(mspace) >= 0);
        assert(H5Sclose(fspace) >= 0);

        // write out source and destination population indices
        dims = 1;
        path = projection_path_join(projection_name, "Source Population");
        mspace = H5Screate_simple(1, &dims, &dims);
        assert(mspace >= 0);
        fspace = H5Screate_simple(1, &dims, &dims);
        assert(fspace >= 0);
        dset = H5Dcreate2(file, path.c_str(), POP_IDX_H5_NATIVE_T,
                          fspace, lcpl, H5P_DEFAULT, H5P_DEFAULT);
        assert(dset >= 0);
        assert(H5Sselect_all(fspace) >= 0);
        assert(H5Dwrite(dset, POP_IDX_H5_NATIVE_T, mspace, fspace,
                        H5P_DEFAULT, &src_pop_idx) >= 0);

        assert(H5Dclose(dset) >= 0);
        assert(H5Sclose(mspace) >= 0);
        assert(H5Sclose(fspace) >= 0);

        path = projection_path_join(projection_name, "Destination Population");
        mspace = H5Screate_simple(1, &dims, &dims);
        assert(mspace >= 0);
        fspace = H5Screate_simple(1, &dims, &dims);
        assert(fspace >= 0);
        dset = H5Dcreate2(file, path.c_str(), POP_IDX_H5_NATIVE_T,
                          fspace, lcpl, H5P_DEFAULT, H5P_DEFAULT);
        assert(dset >= 0);
        assert(H5Sselect_all(fspace) >= 0);
        assert(H5Dwrite(dset, POP_IDX_H5_NATIVE_T, mspace, fspace,
                        H5P_DEFAULT, &dst_pop_idx) >= 0);

        assert(H5Dclose(dset) >= 0);
        assert(H5Sclose(mspace) >= 0);
        assert(H5Sclose(fspace) >= 0);

        
        // clean-up
        assert(H5Pclose(lcpl) >= 0);
      }
    }
  }
}
