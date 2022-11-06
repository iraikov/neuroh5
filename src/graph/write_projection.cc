

#include "debug.hh"
#include "neuroh5_types.hh"
#include "path_names.hh"
#include "write_projection.hh"
#include "write_template.hh"
#include "edge_attributes.hh"
#include "throw_assert.hh"

#include <algorithm>
#include <map>

using namespace std;

namespace neuroh5
{
  namespace graph
  {


    
    void write_projection
    (
     MPI_Comm                  comm,
     hid_t                     file,
     const string&             src_pop_name,
     const string&             dst_pop_name,
     const NODE_IDX_T&         src_start,
     const NODE_IDX_T&         src_end,
     const NODE_IDX_T&         dst_start,
     const NODE_IDX_T&         dst_end,
     const size_t&             num_edges,
     const edge_map_t&         prj_edge_map,
     const std::map <std::string, std::pair <size_t, data::AttrIndex > >& edge_attr_index,
     hsize_t                   chunk_size,
     hsize_t                   block_size,
     const bool collective
     )
    {
      // do a sanity check on the input
      throw_assert_nomsg(src_start < src_end);
      throw_assert_nomsg(dst_start < dst_end);
      
      int ssize, srank;
      throw_assert_nomsg(MPI_Comm_size(comm, &ssize) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Comm_rank(comm, &srank) == MPI_SUCCESS);
      size_t size, rank;
      size = (size_t)ssize;
      rank = (size_t)srank;
        
      size_t num_dest = prj_edge_map.size();
      size_t num_blocks = num_dest > 0 ? 1 : 0;
        
      // create relative destination pointers and source index
      vector<DST_BLK_PTR_T> dst_blk_ptr; 
      vector<DST_PTR_T> dst_ptr;
      vector<NODE_IDX_T> dst_blk_idx, src_idx;
      NODE_IDX_T first_idx = 0, last_idx = 0;
      hsize_t num_block_edges = 0, num_prj_edges = 0;
      if (!prj_edge_map.empty())
        {
          first_idx = (prj_edge_map.begin())->first;
          last_idx  = first_idx;
          dst_blk_idx.push_back(first_idx - dst_start);
          dst_blk_ptr.push_back(0);
          for (auto iter = prj_edge_map.begin(); iter != prj_edge_map.end(); ++iter)
            {
              NODE_IDX_T dst = iter->first;
              edge_tuple_t et = iter->second;
              vector<NODE_IDX_T> &v = get<0>(et);
              
              // creates new block if non-contiguous dst indices
              if (((dst > 0) && ((dst-1) > last_idx)) || (num_block_edges > block_size))
                {
                  dst_blk_idx.push_back(dst - dst_start);
                  dst_blk_ptr.push_back(dst_ptr.size());
                  num_blocks++;
                  num_block_edges = 0;
                }
              last_idx = dst;
              
              copy(v.begin(), v.end(), back_inserter(src_idx));
              dst_ptr.push_back(num_prj_edges);
              num_prj_edges += v.size();
              num_block_edges += v.size();
            }
        }
      dst_ptr.push_back(num_prj_edges);
      throw_assert_nomsg(num_edges == src_idx.size());

      

      // exchange allocation data
      vector<size_t> sendbuf_num_blocks(size, num_blocks);
      vector<size_t> recvbuf_num_blocks(size);
      throw_assert_nomsg(MPI_Allgather(&sendbuf_num_blocks[0], 1, MPI_SIZE_T,
                           &recvbuf_num_blocks[0], 1, MPI_SIZE_T, comm)
             == MPI_SUCCESS);

      vector<size_t> sendbuf_num_dest(size, num_dest);
      vector<size_t> recvbuf_num_dest(size);
      throw_assert_nomsg(MPI_Allgather(&sendbuf_num_dest[0], 1, MPI_SIZE_T,
                           &recvbuf_num_dest[0], 1, MPI_SIZE_T, comm)
             == MPI_SUCCESS);

      vector<size_t> sendbuf_num_edge(size, num_edges);
      vector<size_t> recvbuf_num_edge(size);
      throw_assert_nomsg(MPI_Allgather(&sendbuf_num_edge[0], 1, MPI_SIZE_T,
                           &recvbuf_num_edge[0], 1, MPI_SIZE_T, comm)
             == MPI_SUCCESS);

      // determine last rank that has data
      size_t last_rank = size-1;

      for (size_t r=last_rank; r >= 0; r--)
	{
	  if (recvbuf_num_blocks[r] > 0)
	    {
	      last_rank = r;
	      break;
	    }
	}
      
      hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
      throw_assert_nomsg(lcpl >= 0);
      throw_assert_nomsg(H5Pset_create_intermediate_group(lcpl, 1) >= 0);

      /* Dataset creation property list to enable chunking */
      hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
      throw_assert_nomsg(dcpl >= 0);
      hsize_t chunk = chunk_size;

      hid_t wapl = H5P_DEFAULT;
#ifdef HDF5_IS_PARALLEL
      if (collective)
	{
	  wapl = H5Pcreate(H5P_DATASET_XFER);
	  throw_assert_nomsg(wapl >= 0);
	  throw_assert_nomsg(H5Pset_dxpl_mpio(wapl, H5FD_MPIO_COLLECTIVE) >= 0);
	}
#endif

      size_t total_num_blocks=0;
      for (size_t p=0; p<size; p++)
        {
          total_num_blocks = total_num_blocks + recvbuf_num_blocks[p];
        }
      throw_assert_nomsg(total_num_blocks > 0);

      size_t total_num_dests=0;
      for (size_t p=0; p<size; p++)
        {
          total_num_dests = total_num_dests + recvbuf_num_dest[p];
        }
      throw_assert_nomsg(total_num_dests > 0);

      size_t total_num_edges=0;
      for (size_t p=0; p<size; p++)
        {
          total_num_edges = total_num_edges + recvbuf_num_edge[p];
        }
      throw_assert_nomsg(total_num_edges > 0);

      hsize_t start = 0, block = 0;
      
      string path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_BLK_IDX);
      hsize_t dst_blk_idx_dims = (hsize_t)total_num_blocks, one = 1;
      hid_t fspace = H5Screate_simple(1, &dst_blk_idx_dims, &dst_blk_idx_dims);
      throw_assert_nomsg(fspace >= 0);
      if (chunk < dst_blk_idx_dims)
        {
          throw_assert_nomsg(H5Pset_chunk(dcpl, 1, &chunk ) >= 0);
        }
      else
        {
          throw_assert_nomsg(H5Pset_chunk(dcpl, 1, &dst_blk_idx_dims ) >= 0);
        }
#ifdef H5_HAS_PARALLEL_DEFLATE
      throw_assert_nomsg(H5Pset_deflate(dcpl, 9) >= 0);
#endif
      hid_t dset = H5Dcreate2(file, path.c_str(), NODE_IDX_H5_FILE_T, fspace,
                              lcpl, dcpl, H5P_DEFAULT);
      throw_assert_nomsg(dset >= 0);
      block = num_blocks;
      hid_t mspace = H5Screate_simple(1, &block, &block);
      throw_assert_nomsg(mspace >= 0);
      throw_assert_nomsg(H5Sselect_all(mspace) >= 0);
      for (size_t p = 0; p < rank; ++p)
        {
          start += recvbuf_num_blocks[p];
        }
        
      if (block > 0)
        {
          throw_assert_nomsg(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                     &one, &block) >= 0);
        }
      else
        {
          throw_assert_nomsg(H5Sselect_none(fspace) >= 0);
        }
      throw_assert_nomsg(H5Dwrite(dset, NODE_IDX_H5_NATIVE_T, mspace, fspace,
		      wapl, &dst_blk_idx[0]) >= 0);

      throw_assert_nomsg(H5Dclose(dset) >= 0);
      throw_assert_nomsg(H5Sclose(mspace) >= 0);
      throw_assert_nomsg(H5Sclose(fspace) >= 0);

      if (block > 0)
        {
          for (size_t p = 0; p < rank; ++p)
            {
              dst_blk_ptr[0] += recvbuf_num_dest[p];
            }
          for (size_t i = 1; i < dst_blk_ptr.size(); i++)
            {
              dst_blk_ptr[i] += dst_blk_ptr[0];
            }
          if (rank == last_rank) // last rank writes the total destination count
            {
              dst_blk_ptr.push_back(dst_blk_ptr[0] + recvbuf_num_dest[rank] + 1);
            }
        }
      
      path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_BLK_PTR);
      hsize_t dst_blk_ptr_dims = (hsize_t)total_num_blocks+1;
      fspace = H5Screate_simple(1, &dst_blk_ptr_dims, &dst_blk_ptr_dims);
      throw_assert_nomsg(fspace >= 0);
      if (chunk < dst_blk_ptr_dims)
        {
          throw_assert_nomsg(H5Pset_chunk(dcpl, 1, &chunk ) >= 0);
        }
      else
        {
          throw_assert_nomsg(H5Pset_chunk(dcpl, 1, &dst_blk_ptr_dims ) >= 0);
        }
#ifdef H5_HAS_PARALLEL_DEFLATE
      throw_assert_nomsg(H5Pset_deflate(dcpl, 9) >= 0);
#endif
      dset = H5Dcreate2(file, path.c_str(), DST_BLK_PTR_H5_FILE_T,
                        fspace, lcpl, dcpl, H5P_DEFAULT);
      throw_assert_nomsg(dset >= 0);
      if (rank == last_rank)
        {
          block = num_blocks+1;
        }
      else
        {
          block = num_blocks;
        }
      mspace = H5Screate_simple(1, &block, &block);
      throw_assert_nomsg(mspace >= 0);
      throw_assert_nomsg(H5Sselect_all(mspace) >= 0);

      start = 0;
      for (size_t p = 0; p < rank; ++p)
        {
          start += recvbuf_num_blocks[p];
        }

      if (block > 0)
        {
          throw_assert_nomsg(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                     &one, &block) >= 0);
        }
      else
        {
          throw_assert_nomsg(H5Sselect_none(fspace) >= 0);
        }
      throw_assert_nomsg(H5Dwrite(dset, DST_BLK_PTR_H5_NATIVE_T, mspace, fspace,
                      wapl, &dst_blk_ptr[0]) >= 0);

      throw_assert_nomsg(H5Dclose(dset) >= 0);
      throw_assert_nomsg(H5Sclose(mspace) >= 0);
      throw_assert_nomsg(H5Sclose(fspace) >= 0);

      /*
        write(file, path, DST_BLK_PTR_H5_FILE_T, dbp);
      */
        

      // write destination pointers
      // # dest. pointers = number of destinations + 1
      size_t s = 0;
      for (size_t p = 0; p < rank; ++p)
        {
          s += recvbuf_num_edge[p];
        }

      for (size_t idst = 0; idst < dst_ptr.size(); ++idst)
        {
          dst_ptr[idst] += s;
        }


      if (rank != last_rank) // only the last rank writes an additional element
        {
          dst_ptr.resize(num_dest);
        }

      path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_PTR);
      hsize_t dst_ptr_dims = total_num_dests+1;

      fspace = H5Screate_simple(1, &dst_ptr_dims, &dst_ptr_dims);
      throw_assert_nomsg(fspace >= 0);
      if (chunk < dst_ptr_dims)
        {
          throw_assert_nomsg(H5Pset_chunk(dcpl, 1, &chunk ) >= 0);
        }
      else
        {
          throw_assert_nomsg(H5Pset_chunk(dcpl, 1, &dst_ptr_dims ) >= 0);
        }
#ifdef H5_HAS_PARALLEL_DEFLATE
      throw_assert_nomsg(H5Pset_deflate(dcpl, 9) >= 0);
#endif
      dset = H5Dcreate2(file, path.c_str(), DST_PTR_H5_FILE_T,
                        fspace, lcpl, dcpl, H5P_DEFAULT);
      throw_assert_nomsg(dset >= 0);
      block = (hsize_t) dst_ptr.size();
      mspace = H5Screate_simple(1, &block, &block);
      throw_assert_nomsg(mspace >= 0);
      throw_assert_nomsg(H5Sselect_all(mspace) >= 0);
      if (block > 0)
        {
          start = (hsize_t)dst_blk_ptr[0];
          throw_assert_nomsg(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                     &one, &block) >= 0);
        }
      else
        {
          throw_assert_nomsg(H5Sselect_none(fspace) >= 0);
        }

      throw_assert_nomsg(H5Dwrite(dset, DST_PTR_H5_NATIVE_T, mspace, fspace,
                      wapl, &dst_ptr[0]) >= 0);

      throw_assert_nomsg(H5Dclose(dset) >= 0);
      throw_assert_nomsg(H5Sclose(mspace) >= 0);
      throw_assert_nomsg(H5Sclose(fspace) >= 0);

      /*
        write(file, path, DST_PTR_H5_FILE_T, dst_ptr);
      */

      // write source index
      // # source indexes = number of edges

      path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::SRC_IDX);
      hsize_t src_idx_dims = total_num_edges;

      fspace = H5Screate_simple(1, &src_idx_dims, &src_idx_dims);
      throw_assert_nomsg(fspace >= 0);
      if (chunk < src_idx_dims)
        {
          throw_assert_nomsg(H5Pset_chunk(dcpl, 1, &chunk ) >= 0);
        }
      else
        {
          throw_assert_nomsg(H5Pset_chunk(dcpl, 1, &src_idx_dims ) >= 0);
        }
#ifdef H5_HAS_PARALLEL_DEFLATE
      throw_assert_nomsg(H5Pset_deflate(dcpl, 9) >= 0);
#endif
      dset = H5Dcreate2(file, path.c_str(), NODE_IDX_H5_FILE_T,
                        fspace, lcpl, dcpl, H5P_DEFAULT);
      throw_assert_nomsg(dset >= 0);

      block = (hsize_t) src_idx.size();
      mspace = H5Screate_simple(1, &block, &block);
      throw_assert_nomsg(mspace >= 0);
      throw_assert_nomsg(H5Sselect_all(mspace) >= 0);
      if (block > 0)
        {
          start = (hsize_t)dst_ptr[0];
          throw_assert_nomsg(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                     &one, &block) >= 0);
        }
      else
        {
          throw_assert_nomsg(H5Sselect_none(fspace) >= 0);
        }

      throw_assert_nomsg(H5Dwrite(dset, NODE_IDX_H5_NATIVE_T, mspace, fspace,
                      wapl, &src_idx[0]) >= 0);

      throw_assert_nomsg(H5Dclose(dset) >= 0);
      throw_assert_nomsg(H5Sclose(mspace) >= 0);
      throw_assert_nomsg(H5Sclose(fspace) >= 0);

      /*
        write(file, path, NODE_IDX_H5_FILE_T, src_idx);
      */
      vector <string> edge_attr_name_spaces;
      map <string, data::NamedAttrVal> edge_attr_map;

      for (auto const& iter : edge_attr_index)
        {
          const string & attr_namespace = iter.first;
          const data::AttrIndex& attr_index  = iter.second.second;

          data::NamedAttrVal& edge_attr_values = edge_attr_map[attr_namespace];

          edge_attr_values.float_values.resize(attr_index.size_attr_index<float>());
          edge_attr_values.uint8_values.resize(attr_index.size_attr_index<uint8_t>());
          edge_attr_values.uint16_values.resize(attr_index.size_attr_index<uint16_t>());
          edge_attr_values.uint32_values.resize(attr_index.size_attr_index<uint32_t>());
          edge_attr_values.int8_values.resize(attr_index.size_attr_index<int8_t>());
          edge_attr_values.int16_values.resize(attr_index.size_attr_index<int16_t>());
          edge_attr_values.int32_values.resize(attr_index.size_attr_index<int32_t>());

          edge_attr_name_spaces.push_back(attr_namespace);
        }
        
      
      for (auto const& iter : prj_edge_map)
	{
	  const edge_tuple_t& et = iter.second;
          const vector<NODE_IDX_T>& v = get<0>(et);
          const vector<data::AttrVal>& a = get<1>(et);
	  if (v.size() > 0)
	    {
              size_t ni=0;
              for (auto & attr_values : a)
                {
                  throw_assert_nomsg(ni < edge_attr_name_spaces.size());
                  const string & attr_namespace = edge_attr_name_spaces[ni];
                  edge_attr_map[attr_namespace].append(attr_values);
                  ni++;
                }
            }
	}

      throw_assert_nomsg(MPI_Barrier(comm) == MPI_SUCCESS);

      write_edge_attribute_map<float>(comm, file, src_pop_name, dst_pop_name,
                                      edge_attr_map, edge_attr_index);
      write_edge_attribute_map<uint8_t>(comm, file, src_pop_name, dst_pop_name,
                                        edge_attr_map, edge_attr_index);
      write_edge_attribute_map<uint16_t>(comm, file, src_pop_name, dst_pop_name,
                                         edge_attr_map, edge_attr_index);
      write_edge_attribute_map<uint32_t>(comm, file, src_pop_name, dst_pop_name,
                                         edge_attr_map, edge_attr_index);
      write_edge_attribute_map<int8_t>(comm, file, src_pop_name, dst_pop_name,
                                       edge_attr_map, edge_attr_index);
      write_edge_attribute_map<int16_t>(comm, file, src_pop_name, dst_pop_name,
                                        edge_attr_map, edge_attr_index);
      write_edge_attribute_map<int32_t>(comm, file, src_pop_name, dst_pop_name,
                                        edge_attr_map, edge_attr_index);
      
      // clean-up
      throw_assert_nomsg(H5Pclose(dcpl) >= 0);
      throw_assert_nomsg(H5Pclose(lcpl) >= 0);
      throw_assert_nomsg(H5Pclose(wapl) >= 0);

      throw_assert_nomsg(MPI_Barrier(comm) == MPI_SUCCESS);
    }
  }
}
