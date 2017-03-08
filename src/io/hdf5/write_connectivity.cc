
#include "write_connectivity.hh"
#include "hdf5_types.hh"
#include "hdf5_path_names.hh"
#include "write_template.hh"

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
       const uint64_t&           num_edges,
       const map<NODE_IDX_T, vector<NODE_IDX_T> >& adj_map,
       hsize_t            cdim
       )
      {
        // do a sanity check on the input
        assert(src_start < src_end);
        assert(dst_start < dst_end);

        uint64_t num_dest = adj_map.size();
        assert(num_dest > 0 && num_dest < (dst_end - dst_start + 1));

        // create relative destination pointers and source index
        vector<uint64_t> dst_ptr(1, 0);
        vector<uint32_t> src_idx;
        size_t pos = 0;
        for (auto iter = adj_map.begin(); iter != adj_map.end(); ++iter)
          {
            dst_ptr.push_back(dst_ptr[pos++] + iter->second.size());
            copy(iter->second.begin(), iter->second.end(),
                 back_inserter(src_idx));
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

        //hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
        //assert(lcpl >= 0);
        //assert(H5Pset_create_intermediate_group(lcpl, 1) >= 0);

        // write destination block index (= first_node)

        string path = io::hdf5::projection_path_join(projection_name, "/Connectivity/Destination Block Index");
        /*
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
        */

        /* Dataset creation property list to enable chunking */
        hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
        assert(dcpl >= 0);
        hsize_t chunk = cdim;
        assert(H5Pset_chunk(dcpl, 1, &chunk ) >= 0);
        assert(H5Pset_deflate(dcpl, 6) >= 0);


        vector<NODE_IDX_T> v_dst_start(1, dst_start);         
        write(file, path, NODE_IDX_H5_FILE_T, v_dst_start, dcpl);

        // write destination block pointer

        path = projection_path_join(projection_name, "/Connectivity/Destination Block Pointer");
        /*
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
        */
        vector<uint64_t> dbp(1,0); // only the last rank writes two elements

        for (int p = 0; p < rank; ++p)
          {
            dbp[0] += recvbuf_num_dest[p];
          }

        if (rank == size-1) // last rank writes the total destination count
          {
            dbp.push_back(dbp[0] + recvbuf_num_dest[rank]);
          }

        /*
        assert(H5Dwrite(dset, DST_BLK_PTR_H5_NATIVE_T, mspace, fspace,
                        H5P_DEFAULT, &dbp[0]) >= 0);

        assert(H5Dclose(dset) >= 0);
        assert(H5Sclose(mspace) >= 0);
        assert(H5Sclose(fspace) >= 0);
        */

        write(file, path, DST_BLK_PTR_H5_FILE_T, dbp, dcpl);

        // write destination pointers
        // # dest. pointers = number of destinations + 1

        path = projection_path_join(projection_name, "/Connectivity/Destination Pointer");
        /*
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
        */
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

        /*
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
        */
        write(file, path, DST_PTR_H5_FILE_T, dst_ptr, dcpl);

        // write source index
        // # source indexes = number of edges

        path = projection_path_join(projection_name, "/Connectivity/Source Index");
        /*
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
        */
        write(file, path, NODE_IDX_H5_FILE_T, src_idx, dcpl);

        // write out source and destination population indices
        //dims = 1;
        path = projection_path_join(projection_name, "Source Population");
/*
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
        */
        vector<POP_IDX_T> v_src_pop_idx(1, src_pop_idx);         
        write(file, path, POP_IDX_H5_FILE_T, v_src_pop_idx, dcpl);

        path = projection_path_join(projection_name, "Destination Population");
        /*
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
        */

        vector<POP_IDX_T> v_dst_pop_idx(1, dst_pop_idx);         
        write(file, path, POP_IDX_H5_FILE_T, v_dst_pop_idx, dcpl);
        
        // clean-up
        assert(H5Pclose(dcpl) >= 0);
        //assert(H5Pclose(lcpl) >= 0);
      }
    }
  }
}
