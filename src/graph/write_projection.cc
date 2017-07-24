

#include "debug.hh"
#include "neuroh5_types.hh"
#include "path_names.hh"
#include "write_projection.hh"
#include "write_template.hh"
#include "edge_attributes.hh"

#include <algorithm>
#include <cassert>
#include <map>

using namespace std;

namespace neuroh5
{
  namespace graph
  {
    void write_projection
    (
     hid_t                     file,
     const string&             src_pop_name,
     const string&             dst_pop_name,
     const NODE_IDX_T&         src_start,
     const NODE_IDX_T&         src_end,
     const NODE_IDX_T&         dst_start,
     const NODE_IDX_T&         dst_end,
     const uint64_t&           num_edges,
     const edge_map_t&         prj_edge_map,
     const vector<vector<string>>& edge_attr_names,
     hsize_t                   cdim
     )
    {
      // do a sanity check on the input
      assert(src_start < src_end);
      assert(dst_start < dst_end);
        

      //assert(prj_edge_map.size() > 0);
      
      // get the I/O communicator
      MPI_Comm comm;
      MPI_Info info;
      hid_t fapl = H5Fget_access_plist(file);
      assert(H5Pget_fapl_mpio(fapl, &comm, &info) >= 0);

      int ssize, srank;
      assert(MPI_Comm_size(comm, &ssize) == MPI_SUCCESS);
      assert(MPI_Comm_rank(comm, &srank) == MPI_SUCCESS);
      size_t size, rank;
      size = (size_t)ssize;
      rank = (size_t)srank;
        
      assert(H5Pclose(fapl) >= 0);

      uint64_t num_dest = prj_edge_map.size();
      uint64_t num_blocks = num_dest > 0 ? 1 : 0;
      if (rank == size-1)
        {
          num_blocks++;
        }
        
      // create relative destination pointers and source index
      vector<uint64_t> dst_blk_ptr(1, 0); // only the last rank writes two elements
      vector<uint64_t> dst_ptr(1, 0);
      vector<NODE_IDX_T> dst_blk_idx, src_idx;
      NODE_IDX_T last_idx = 0;
      size_t pos = 0; 
      for (auto iter = prj_edge_map.begin(); iter != prj_edge_map.end(); ++iter)
        {
          NODE_IDX_T dst = iter->first;
          edge_tuple_t et = iter->second;
          vector<NODE_IDX_T> v = get<0>(et);
          data::AttrVal a = get<1>(et);

          if (!dst_blk_idx.empty())
            {
              // creates new block if non-contiguous dst indices
              if ((dst-1) > last_idx)
                {
                  dst_blk_idx.push_back(dst - dst_start);
                  dst_blk_ptr.push_back(dst_ptr.size()-1);
                  num_blocks++;
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
      printf("dst_ptr.back = %u\n", dst_ptr.back());
      assert(num_edges == src_idx.size());


      // exchange allocation data

      vector<uint64_t> sendbuf_num_blocks(size, num_blocks);
      vector<uint64_t> recvbuf_num_blocks(size);
      assert(MPI_Allgather(&sendbuf_num_blocks[0], 1, MPI_UINT64_T,
                           &recvbuf_num_blocks[0], 1, MPI_UINT64_T, comm)
             == MPI_SUCCESS);

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

      /* Dataset creation property list to enable chunking */
      hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
      assert(dcpl >= 0);
      hsize_t chunk = cdim;
      assert(H5Pset_chunk(dcpl, 1, &chunk ) >= 0);
      //assert(H5Pset_deflate(dcpl, 6) >= 0);

      uint64_t total_num_blocks=0;
      for (size_t p=0; p<size; p++)
        {
          total_num_blocks = total_num_blocks + recvbuf_num_blocks[p];
        }

      uint64_t total_num_dests=0;
      for (size_t p=0; p<size; p++)
        {
          total_num_dests = total_num_dests + recvbuf_num_dest[p];
        }

      uint64_t total_num_edges=0;
      for (size_t p=0; p<size; p++)
        {
          total_num_edges = total_num_edges + recvbuf_num_edge[p];
        }
        
      string path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::DST_BLK_IDX);
      hsize_t dims = (hsize_t)total_num_blocks-1, one = 1;
      hid_t fspace = H5Screate_simple(1, &dims, &dims);
      assert(fspace >= 0);
      if (chunk < dims)
        {
          assert(H5Pset_chunk(dcpl, 1, &chunk ) >= 0);
        }
      else
        {
          assert(H5Pset_chunk(dcpl, 1, &dims ) >= 0);
        }
      hid_t dset = H5Dcreate2(file, path.c_str(), NODE_IDX_H5_FILE_T, fspace,
                              lcpl, dcpl, H5P_DEFAULT);
      assert(dset >= 0);
      if ((rank == size-1) && (num_blocks > 0))
        {
          dims = num_blocks-1;
        }
      else
        {
          dims = num_blocks;
        }
      hid_t mspace = H5Screate_simple(1, &dims, &dims);
      assert(mspace >= 0);
      assert(H5Sselect_all(mspace) >= 0);
      hsize_t start = 0;
      for (size_t p = 0; p < rank; ++p)
        {
          start += recvbuf_num_blocks[p];
        }
        
      hsize_t block = dims;
      if (block > 0)
        {
          assert(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                     &one, &block) >= 0);
        }
      else
        {
          assert(H5Sselect_none(fspace) >= 0);
        }
      assert(H5Dwrite(dset, NODE_IDX_H5_NATIVE_T, mspace, fspace, H5P_DEFAULT,
                      &dst_blk_idx[0]) >= 0);

      assert(H5Dclose(dset) >= 0);
      assert(H5Sclose(mspace) >= 0);
      assert(H5Sclose(fspace) >= 0);

      /*
        vector<NODE_IDX_T> v_dst_start(1, dst_start);         
        write(file, path, NODE_IDX_H5_FILE_T, v_dst_start);
      */

      for (size_t p = 0; p < rank; ++p)
        {
          dst_blk_ptr[0] += recvbuf_num_dest[p];
        }
      for (size_t i = 1; i < dst_blk_ptr.size(); i++)
        {
          dst_blk_ptr[i] += dst_blk_ptr[0];
        }
      if (rank == size-1) // last rank writes the total destination count
        {
          dst_blk_ptr.push_back(dst_blk_ptr[0] + recvbuf_num_dest[rank]);
        }

      path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::DST_BLK_PTR);
      dims = (hsize_t)total_num_blocks;
      fspace = H5Screate_simple(1, &dims, &dims);
      assert(fspace >= 0);
      if (chunk < dims)
        {
          assert(H5Pset_chunk(dcpl, 1, &chunk ) >= 0);
        }
      else
        {
          assert(H5Pset_chunk(dcpl, 1, &dims ) >= 0);
        }
      dset = H5Dcreate2(file, path.c_str(), DST_BLK_PTR_H5_FILE_T,
                        fspace, lcpl, dcpl, H5P_DEFAULT);
      assert(dset >= 0);
      dims = num_blocks;
      mspace = H5Screate_simple(1, &dims, &dims);
      assert(mspace >= 0);
      assert(H5Sselect_all(mspace) >= 0);

      start = 0;
      for (size_t p = 0; p < rank; ++p)
        {
          start += recvbuf_num_blocks[p];
        }

      block = dims;
      if (block > 0)
        {
          assert(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                     &one, &block) >= 0);
        }
      else
        {
          assert(H5Sselect_none(fspace) >= 0);
        }
      assert(H5Dwrite(dset, DST_BLK_PTR_H5_NATIVE_T, mspace, fspace,
                      H5P_DEFAULT, &dst_blk_ptr[0]) >= 0);

      assert(H5Dclose(dset) >= 0);
      assert(H5Sclose(mspace) >= 0);
      assert(H5Sclose(fspace) >= 0);

      /*
	if (rank == 0)
        {
        DEBUG("writing dst_blk_ptr\n");
        }

        write(file, path, DST_BLK_PTR_H5_FILE_T, dbp);
      */
        

      // write destination pointers
      // # dest. pointers = number of destinations + 1
      uint64_t s = 0;
      for (size_t p = 0; p < rank; ++p)
        {
          s += recvbuf_num_edge[p];
        }

      for (size_t idst = 0; idst < dst_ptr.size(); ++idst)
        {
          dst_ptr[idst] += s;
        }

      if (rank < size-1) // only the last rank writes an additional element
        {
          //dst_ptr.back() += recvbuf_num_edge[rank];
          dst_ptr.resize(num_dest);
        }
        
      path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::DST_PTR);
      dims = total_num_dests;
      ++dims; // one extra element

      fspace = H5Screate_simple(1, &dims, &dims);
      assert(fspace >= 0);
      if (chunk < dims)
        {
          assert(H5Pset_chunk(dcpl, 1, &chunk ) >= 0);
        }
      else
        {
          assert(H5Pset_chunk(dcpl, 1, &dims ) >= 0);
        }
      dset = H5Dcreate2(file, path.c_str(), DST_PTR_H5_FILE_T,
                        fspace, lcpl, dcpl, H5P_DEFAULT);
      assert(dset >= 0);
      dims = (hsize_t) dst_ptr.size();
      mspace = H5Screate_simple(1, &dims, &dims);
      assert(mspace >= 0);
      assert(H5Sselect_all(mspace) >= 0);
      start = (hsize_t)dst_blk_ptr[0];
      block = dims;
      if (block > 0)
        {
          assert(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                     &one, &block) >= 0);
        }
      else
        {
          assert(H5Sselect_none(fspace) >= 0);
        }

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

      path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::SRC_IDX);
      dims = total_num_edges;

      fspace = H5Screate_simple(1, &dims, &dims);
      assert(fspace >= 0);
      if (chunk < dims)
        {
          assert(H5Pset_chunk(dcpl, 1, &chunk ) >= 0);
        }
      else
        {
          assert(H5Pset_chunk(dcpl, 1, &dims ) >= 0);
        }
      dset = H5Dcreate2(file, path.c_str(), NODE_IDX_H5_FILE_T,
                        fspace, lcpl, dcpl, H5P_DEFAULT);
      assert(dset >= 0);

      dims = (hsize_t) src_idx.size();
      mspace = H5Screate_simple(1, &dims, &dims);
      assert(mspace >= 0);
      assert(H5Sselect_all(mspace) >= 0);
      start = (hsize_t)dst_ptr[0];
      block = dims;
      if (block > 0)
        {
          assert(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                     &one, &block) >= 0);
        }
      else
        {
          assert(H5Sselect_none(fspace) >= 0);
        }

      assert(H5Dwrite(dset, NODE_IDX_H5_NATIVE_T, mspace, fspace,
                      H5P_DEFAULT, &src_idx[0]) >= 0);

      assert(H5Dclose(dset) >= 0);
      assert(H5Sclose(mspace) >= 0);
      assert(H5Sclose(fspace) >= 0);

      /*
        write(file, path, NODE_IDX_H5_FILE_T, src_idx);
      */
        
        
      data::NamedAttrVal edge_attr_values;
      edge_attr_values.float_values.resize(edge_attr_names[data::AttrVal::attr_index_float].size());
      edge_attr_values.uint8_values.resize(edge_attr_names[data::AttrVal::attr_index_uint8].size());
      edge_attr_values.uint16_values.resize(edge_attr_names[data::AttrVal::attr_index_uint16].size());
      edge_attr_values.uint32_values.resize(edge_attr_names[data::AttrVal::attr_index_uint32].size());
        
      for (auto iter = prj_edge_map.begin(); iter != prj_edge_map.end(); ++iter)
        {
          edge_tuple_t et = iter->second;
          data::AttrVal a = get<1>(et);
          edge_attr_values.append(a);
        }
        
      for (size_t i=0; i<edge_attr_values.float_values.size(); i++)
        {
          const string& attr_name = edge_attr_names[data::AttrVal::attr_index_float][i];
          string path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name, attr_name);
          graph::write_edge_attribute<float>(file, path, edge_attr_values.float_values[i]);
        }
        
      for (size_t i=0; i<edge_attr_values.uint8_values.size(); i++)
        {
          const string& attr_name = edge_attr_names[data::AttrVal::attr_index_uint8][i];
          string path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name, attr_name);
          graph::write_edge_attribute<uint8_t>(file, path, edge_attr_values.uint8_values[i]);
        }
        
      for (size_t i=0; i<edge_attr_values.uint16_values.size(); i++)
        {
          const string& attr_name = edge_attr_names[data::AttrVal::attr_index_uint16][i];
          string path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name, attr_name);
          graph::write_edge_attribute<uint16_t>(file, path, edge_attr_values.uint16_values[i]);
        }
        
      for (size_t i=0; i<edge_attr_values.uint32_values.size(); i++)
        {
          const string& attr_name = edge_attr_names[data::AttrVal::attr_index_uint32][i];
          string path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name, attr_name);
          graph::write_edge_attribute<uint32_t>(file, path, edge_attr_values.uint32_values[i]);
        }
        
        
      // clean-up
      assert(H5Pclose(dcpl) >= 0);
      assert(H5Pclose(lcpl) >= 0);
    }
  }
}
