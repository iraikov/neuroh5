

#include "debug.hh"
#include "hdf5_types.hh"
#include "hdf5_path_names.hh"
#include "write_projection.hh"
#include "write_edge_attributes.hh"
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
      void write_projection
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
       const model::edge_map_t&  prj_edge_map,
       const vector<vector<string>>& edge_attr_names,
       hsize_t                   cdim
       )
      {
        // do a sanity check on the input
        assert(src_start < src_end);
        assert(dst_start < dst_end);

        uint64_t num_dest = prj_edge_map.size();

        // create relative destination pointers and source index
        vector<uint64_t> dbp(1,0); // only the last rank writes two elements
        vector<uint64_t> dst_ptr(1, 0);
        vector<NODE_IDX_T> dst_blk_idx, src_idx;
        NODE_IDX_T last_idx;
        size_t pos = 0; 
        for (auto iter = prj_edge_map.begin(); iter != prj_edge_map.end(); ++iter)
          {
            NODE_IDX_T dst = iter->first;
            model::edge_tuple_t et = iter->second;
            vector<NODE_IDX_T> v = get<0>(et);
            model::EdgeAttr a = get<1>(et);

            if (!dst_blk_idx.empty())
              {
                // creates new block if non-contiguous dst indices
                if ((dst-1) > last_idx)
                  {
                    dst_blk_idx.push_back(dst - dst_start);
                    dbp.push_back(dst_ptr.size());
                  }
                last_idx = dst;
              }
            else
              {
                dst_blk_idx.push_back(dst - dst_start);
                last_idx = dst;
              }

            dst_ptr.push_back(dst_ptr[pos++] + v.size());
            copy(v.begin(), v.end(), back_inserter(src_idx));
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
                        &dst_blk_idx[0]) >= 0);
        assert(H5Dclose(dset) >= 0);
        assert(H5Sclose(mspace) >= 0);
        assert(H5Sclose(fspace) >= 0);

        /* Dataset creation property list to enable chunking */
        hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
        assert(dcpl >= 0);
        hsize_t chunk = cdim;
        assert(H5Pset_chunk(dcpl, 1, &chunk ) >= 0);
	//assert(H5Pset_deflate(dcpl, 6) >= 0);

        /*
        vector<NODE_IDX_T> v_dst_start(1, dst_start);         
        write(file, path, NODE_IDX_H5_FILE_T, v_dst_start);
        */

        for (int p = 0; p < rank; ++p)
          {
            dbp[0] += recvbuf_num_dest[p];
          }
        for (size_t i = 1; i < dbp.size(); i++)
          {
            dbp[i] += dbp[0];
          }
        if (rank == size-1) // last rank writes the total destination count
          {
            dbp.push_back(dbp[0] + recvbuf_num_dest[rank]);
          }

        path = projection_path_join(projection_name, "/Connectivity/Destination Block Pointer");
        dims = (hsize_t)(size + 1);
        fspace = H5Screate_simple(1, &dims, &dims);
        assert(fspace >= 0);
        dset = H5Dcreate2(file, path.c_str(), DST_BLK_PTR_H5_FILE_T,
                          fspace, lcpl, H5P_DEFAULT, H5P_DEFAULT);
        assert(dset >= 0);

        dims = dbp.size();
        mspace = H5Screate_simple(1, &dims, &dims);
        assert(mspace >= 0);
        assert(H5Sselect_all(mspace) >= 0);
        start = (hsize_t)rank;
        block = dims;
        assert(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                   &one, &block) >= 0);
        assert(H5Dwrite(dset, DST_BLK_PTR_H5_NATIVE_T, mspace, fspace,
                        H5P_DEFAULT, &dbp[0]) >= 0);

        assert(H5Dclose(dset) >= 0);
        assert(H5Sclose(mspace) >= 0);
        assert(H5Sclose(fspace) >= 0);

        /*
	if (rank == 0)
	  {
	    DEBUG("writing dbp\n");
	  }

        write(file, path, DST_BLK_PTR_H5_FILE_T, dbp);
        */
        

        // write destination pointers
        // # dest. pointers = number of destinations + 1
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

        /*
	if (rank == 0)
	  {
	    DEBUG("writing dst_ptr\n");
	  }
        write(file, path, DST_PTR_H5_FILE_T, dst_ptr);
        */

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

        /*
        write(file, path, NODE_IDX_H5_FILE_T, src_idx);
        */
        
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
        /*
        vector<POP_IDX_T> v_src_pop_idx(1, src_pop_idx);         
	if (rank == 0)
	  {
	    DEBUG("writing src_pop_idx\n");
	  }
        write(file, path, POP_IDX_H5_FILE_T, v_src_pop_idx);
        */
        
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
        
        model::EdgeNamedAttr edge_attr_values;
        vector< vector<float> > float_attrs (edge_attr_names[model::EdgeAttr::attr_index_float].size());

        for (auto iter = prj_edge_map.begin(); iter != prj_edge_map.end(); ++iter)
          {
            model::edge_tuple_t et = iter->second;
            model::EdgeAttr a = get<1>(et);
            edge_attr_values.append(a);
          }
        
        for (size_t i=0; i<edge_attr_values.float_values.size(); i++)
          {
            const string& attr_name = edge_attr_names[model::EdgeAttr::attr_index_float][i];
            string path = io::hdf5::edge_attribute_path(projection_name, attr_name);
            io::hdf5::write_sparse_edge_attribute<float>(file, path, edge_attr_values.float_values[i]);
          }
        
        float_attrs.clear();
        
        
        // clean-up
        assert(H5Pclose(dcpl) >= 0);
        assert(H5Pclose(lcpl) >= 0);
      }
    }
  }
}
