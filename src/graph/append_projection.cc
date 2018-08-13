

#include "debug.hh"
#include "neuroh5_types.hh"
#include "path_names.hh"
#include "create_group.hh"
#include "exists_dataset.hh"
#include "append_projection.hh"
#include "write_template.hh"
#include "edge_attributes.hh"
#include "mpe_seq.hh"
#include "mpi_debug.hh"

#include <algorithm>
#include <cassert>
#include <map>

using namespace std;

namespace neuroh5
{
  namespace graph
  {

    void append_dst_blk_idx
    (
     size_t                    rank,
     hid_t                     file,
     const string&             src_pop_name,
     const string&             dst_pop_name,
     const NODE_IDX_T&         src_start,
     const NODE_IDX_T&         src_end,
     const NODE_IDX_T&         dst_start,
     const NODE_IDX_T&         dst_end,
     const hsize_t&            num_blocks,
     const hsize_t&            total_num_blocks,
     const hsize_t&            dst_blk_idx_size,
     const hsize_t             chunk_size,
     const hsize_t             block_size,
     const bool                collective,
     vector<size_t>&           recvbuf_num_blocks,
     vector<NODE_IDX_T>&       dst_blk_idx
     )
    {
      hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
      assert(lcpl >= 0);
      assert(H5Pset_create_intermediate_group(lcpl, 1) >= 0);
      hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
      assert(H5Pset_layout(dcpl, H5D_CHUNKED) >= 0);
      hid_t wapl = H5P_DEFAULT;
      if (collective)
	{
	  wapl = H5Pcreate(H5P_DATASET_XFER);
	  assert(wapl >= 0);
	  assert(H5Pset_dxpl_mpio(wapl, H5FD_MPIO_COLLECTIVE) >= 0);
	}

      hsize_t chunk = chunk_size;
      assert(H5Pset_chunk(dcpl, 1, &chunk ) >= 0);
      hsize_t maxdims[1] = {H5S_UNLIMITED};
      hsize_t zerodims[1] = {0};

      // do a sanity check on the input
      assert(src_start < src_end);
      assert(dst_start < dst_end);

      
      string path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_BLK_IDX);
      hsize_t dst_blk_idx_dims = total_num_blocks, one=1;

      hid_t dset, fspace;
      
      if (!(hdf5::exists_dataset (file, path) > 0))
        {
          fspace = H5Screate_simple(1, zerodims, maxdims);
          assert(fspace >= 0);
	  assert(H5Pset_chunk(dcpl, 1, &chunk ) >= 0);
          dset = H5Dcreate2(file, path.c_str(), NODE_IDX_H5_FILE_T, fspace,
                            lcpl, dcpl, H5P_DEFAULT);
          assert(dset >= 0);
        }
      else
        {
          dset = H5Dopen2 (file, path.c_str(), H5P_DEFAULT);
          assert(dset >= 0);
      
          fspace = H5Dget_space(dset);
          assert(fspace >= 0);
        }
                                                            
      hsize_t dst_blk_idx_start = dst_blk_idx_size;
      hsize_t dst_blk_idx_newsize = dst_blk_idx_start + dst_blk_idx_dims;
      if (dst_blk_idx_newsize > 0)
        {
          herr_t ierr = H5Dset_extent (dset, &dst_blk_idx_newsize);
          assert(ierr >= 0);
        }
      assert(H5Sclose(fspace) >= 0);

      hsize_t block = num_blocks;
      
      hid_t mspace  = H5Screate_simple(1, &block, &block);
      assert(mspace >= 0);
      assert(H5Sselect_all(mspace) >= 0);
      hsize_t start = dst_blk_idx_start;
      for (size_t p = 0; p < rank; ++p)
        {
          start += recvbuf_num_blocks[p];
        }
        
      fspace = H5Dget_space(dset);
      assert(fspace >= 0);
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
		      wapl, &dst_blk_idx[0]) >= 0);

      // clean-up
      assert(H5Dclose(dset) >= 0);
      assert(H5Sclose(mspace) >= 0);
      assert(H5Sclose(fspace) >= 0);

      assert(H5Pclose(lcpl) >= 0);
      assert(H5Pclose(dcpl) >= 0);
      assert(H5Pclose(wapl) >= 0);

    }


    void append_dst_blk_ptr
    (
     size_t                    rank,
     hid_t                     file,
     const string&             src_pop_name,
     const string&             dst_pop_name,
     const NODE_IDX_T&         src_start,
     const NODE_IDX_T&         src_end,
     const NODE_IDX_T&         dst_start,
     const NODE_IDX_T&         dst_end,
     const hsize_t&            num_blocks,
     const hsize_t&            total_num_blocks,
     const hsize_t&            dst_blk_ptr_size,
     const hsize_t             chunk_size,
     const hsize_t             block_size,
     const bool                collective,
     vector<size_t>&           recvbuf_num_blocks,
     vector<DST_BLK_PTR_T>&    dst_blk_ptr
     )
    {
      hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
      assert(lcpl >= 0);
      assert(H5Pset_create_intermediate_group(lcpl, 1) >= 0);
      hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
      assert(H5Pset_layout(dcpl, H5D_CHUNKED) >= 0);
      hid_t wapl = H5P_DEFAULT;
      if (collective)
	{
	  wapl = H5Pcreate(H5P_DATASET_XFER);
	  assert(wapl >= 0);
	  assert(H5Pset_dxpl_mpio(wapl, H5FD_MPIO_COLLECTIVE) >= 0);
	}

      hsize_t chunk = chunk_size;
      assert(H5Pset_chunk(dcpl, 1, &chunk ) >= 0);
      hsize_t maxdims[1] = {H5S_UNLIMITED};
      hsize_t zerodims[1] = {0};


      hsize_t dst_blk_ptr_start = 0;
      if (dst_blk_ptr_size > 0)
        {
          dst_blk_ptr_start = dst_blk_ptr_size - 1;
        }
      
      string path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_BLK_PTR);

      hid_t dset, fspace;

      hsize_t dst_blk_ptr_dims = (hsize_t)total_num_blocks+1, one=1;
      if (!(hdf5::exists_dataset (file, path) > 0))
        {
          fspace = H5Screate_simple(1, zerodims, maxdims);
          assert(fspace >= 0);

	  assert(H5Pset_chunk(dcpl, 1, &chunk ) >= 0);

          dset = H5Dcreate2 (file, path.c_str(), DST_BLK_PTR_H5_FILE_T,
                             fspace, lcpl, dcpl, H5P_DEFAULT);
          assert(H5Sclose(fspace) >= 0);
        }
      else
        {
          dset = H5Dopen2 (file, path.c_str(), H5P_DEFAULT);
          assert(dset >= 0);
        }

      hsize_t dst_blk_ptr_newsize = dst_blk_ptr_start + dst_blk_ptr_dims;
      if (dst_blk_ptr_newsize > 0)
        {
          herr_t ierr = H5Dset_extent (dset, &dst_blk_ptr_newsize);
          assert(ierr >= 0);
        }

      hsize_t block = dst_blk_ptr.size();

      hid_t mspace  = H5Screate_simple(1, &block, &block);
      assert(mspace >= 0);
      assert(H5Sselect_all(mspace) >= 0);

      hsize_t start = dst_blk_ptr_start;
      for (size_t p = 0; p < rank; ++p)
        {
          start += recvbuf_num_blocks[p];
        }

      fspace = H5Dget_space(dset);
      assert(fspace >= 0);
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
                      wapl, &dst_blk_ptr[0]) >= 0);

      assert(H5Dclose(dset) >= 0);
      assert(H5Sclose(mspace) >= 0);
      assert(H5Sclose(fspace) >= 0);

      assert(H5Pclose(lcpl) >= 0);
      assert(H5Pclose(dcpl) >= 0);
      assert(H5Pclose(wapl) >= 0);
    }


    void append_dst_ptr
    (
     size_t                    rank,
     hid_t                     file,
     const string&             src_pop_name,
     const string&             dst_pop_name,
     const NODE_IDX_T&         src_start,
     const NODE_IDX_T&         src_end,
     const NODE_IDX_T&         dst_start,
     const NODE_IDX_T&         dst_end,
     const hsize_t&            total_num_dests,
     const hsize_t&            dst_ptr_size,
     const hsize_t             chunk_size,
     const hsize_t             block_size,
     const bool                collective,
     vector<size_t>&           recvbuf_num_dest,
     vector<DST_PTR_T>&    dst_ptr
     )
    {

      hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
      assert(lcpl >= 0);
      assert(H5Pset_create_intermediate_group(lcpl, 1) >= 0);
      hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
      assert(H5Pset_layout(dcpl, H5D_CHUNKED) >= 0);
      hid_t wapl = H5P_DEFAULT;
      if (collective)
	{
	  wapl = H5Pcreate(H5P_DATASET_XFER);
	  assert(wapl >= 0);
	  assert(H5Pset_dxpl_mpio(wapl, H5FD_MPIO_COLLECTIVE) >= 0);
	}

      hsize_t chunk = chunk_size;
      assert(H5Pset_chunk(dcpl, 1, &chunk ) >= 0);
      hsize_t maxdims[1] = {H5S_UNLIMITED};
      hsize_t zerodims[1] = {0};

      string path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_PTR);
      hsize_t dst_ptr_dims = total_num_dests+1, one=1;

      hid_t dset, fspace;
      
      if (!(hdf5::exists_dataset (file, path) > 0))
        {
          fspace = H5Screate_simple(1, zerodims, maxdims);
          assert(fspace >= 0);

	  assert(H5Pset_chunk(dcpl, 1, &chunk ) >= 0);

          dset = H5Dcreate2 (file, path.c_str(), DST_PTR_H5_FILE_T,
                             fspace, lcpl, dcpl, H5P_DEFAULT);
          assert(dset >= 0);
          assert(H5Sclose(fspace) >= 0);
        }
      else
        {
          dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
          assert(dset >= 0);

        }

      fspace = H5Dget_space(dset);
      assert(fspace >= 0);
      hsize_t dst_ptr_start = 0;
      if (dst_ptr_size > 0)
        {
          dst_ptr_start = dst_ptr_size-1;
        }
      
      assert(H5Sclose(fspace) >= 0);

      hsize_t dst_ptr_newsize = dst_ptr_start + dst_ptr_dims;
      if (dst_ptr_newsize > 0)
        {
          herr_t ierr = H5Dset_extent (dset, &dst_ptr_newsize);
          assert(ierr >= 0);
        }

      hsize_t block = (hsize_t) dst_ptr.size();
      hid_t mspace = H5Screate_simple(1, &block, &block);
      assert(mspace >= 0);
      assert(H5Sselect_all(mspace) >= 0);
      
      fspace = H5Dget_space(dset);
      assert(fspace >= 0);

      hsize_t start=0;
      if (block > 0)
        {
          start = dst_ptr_start;
          for (size_t p = 0; p < rank; ++p)
            {
              start += recvbuf_num_dest[p];
            }

          assert(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                     &one, &block) >= 0);
        }
      else
        {
          assert(H5Sselect_none(fspace) >= 0);
        }

      assert(H5Dwrite(dset, DST_PTR_H5_NATIVE_T, mspace, fspace,
                      wapl, &dst_ptr[0]) >= 0);

      assert(H5Dclose(dset) >= 0);
      assert(H5Sclose(mspace) >= 0);
      assert(H5Sclose(fspace) >= 0);

      assert(H5Pclose(lcpl) >= 0);
      assert(H5Pclose(dcpl) >= 0);
      assert(H5Pclose(wapl) >= 0);

    }


    void append_src_idx
    (
     size_t                    rank,
     hid_t                     file,
     const string&             src_pop_name,
     const string&             dst_pop_name,
     const NODE_IDX_T&         src_start,
     const NODE_IDX_T&         src_end,
     const NODE_IDX_T&         dst_start,
     const NODE_IDX_T&         dst_end,
     const hsize_t&            total_num_edges,
     const hsize_t&            src_idx_size,
     const hsize_t             chunk_size,
     const hsize_t             block_size,
     const bool                collective,
     vector<size_t>&           recvbuf_num_edge,
     vector<NODE_IDX_T>        src_idx
     )
    {

      hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
      assert(lcpl >= 0);
      assert(H5Pset_create_intermediate_group(lcpl, 1) >= 0);
      hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
      assert(H5Pset_layout(dcpl, H5D_CHUNKED) >= 0);
      hid_t wapl = H5P_DEFAULT;
      if (collective)
	{
	  wapl = H5Pcreate(H5P_DATASET_XFER);
	  assert(wapl >= 0);
	  assert(H5Pset_dxpl_mpio(wapl, H5FD_MPIO_COLLECTIVE) >= 0);
	}

      hsize_t chunk = chunk_size;
      assert(H5Pset_chunk(dcpl, 1, &chunk ) >= 0);
      hsize_t maxdims[1] = {H5S_UNLIMITED};
      hsize_t zerodims[1] = {0};
    

      string path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::SRC_IDX);
      hsize_t src_idx_dims = total_num_edges, one=1;

      hid_t dset, fspace;
      
      if (!(hdf5::exists_dataset (file, path) > 0))
        {
          fspace = H5Screate_simple(1, zerodims, maxdims);
          assert(fspace >= 0);

	  assert(H5Pset_chunk(dcpl, 1, &chunk ) >= 0);

          dset = H5Dcreate2 (file, path.c_str(), NODE_IDX_H5_FILE_T,
                             fspace, lcpl, dcpl, H5P_DEFAULT);
          assert(dset >= 0);
        }
      else
        {
          dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
          assert(dset >= 0);
          fspace = H5Dget_space(dset);
          assert(fspace >= 0);
        }

      hsize_t src_idx_start = src_idx_size;
      assert(H5Sclose(fspace) >= 0);

      hsize_t src_idx_newsize = src_idx_start + src_idx_dims;
      if (src_idx_newsize > 0)
        {
          herr_t ierr = H5Dset_extent (dset, &src_idx_newsize);
          assert(ierr >= 0);
        }

      hsize_t block = (hsize_t) src_idx.size();
      hid_t mspace = H5Screate_simple(1, &block, &block);
      assert(mspace >= 0);
      assert(H5Sselect_all(mspace) >= 0);

      fspace = H5Dget_space(dset);
      assert(fspace >= 0);

      hsize_t start = 0;
      if (block > 0)
        {
          start = src_idx_start;
          for (size_t p = 0; p < rank; ++p)
            {
              start += recvbuf_num_edge[p];
            }
          assert(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                     &one, &block) >= 0);
        }
      else
        {
          assert(H5Sselect_none(fspace) >= 0);
        }
      assert(H5Dwrite(dset, NODE_IDX_H5_NATIVE_T, mspace, fspace,
                      wapl, &src_idx[0]) >= 0);

      assert(H5Dclose(dset) >= 0);
      assert(H5Sclose(mspace) >= 0);
      assert(H5Sclose(fspace) >= 0);

      assert(H5Pclose(lcpl) >= 0);
      assert(H5Pclose(dcpl) >= 0);
      assert(H5Pclose(wapl) >= 0);
    }
    
    
    void append_projection
    (
     hid_t                     file,
     const string&             src_pop_name,
     const string&             dst_pop_name,
     const NODE_IDX_T&         src_start,
     const NODE_IDX_T&         src_end,
     const NODE_IDX_T&         dst_start,
     const NODE_IDX_T&         dst_end,
     const size_t&             num_edges,
     const edge_map_t&         prj_edge_map,
     const map<string, vector < vector<string> > >& edge_attr_names,
     const hsize_t             chunk_size,
     const hsize_t             block_size,
     const bool collective
     )
    {
      // do a sanity check on the input
      assert(src_start < src_end);
      assert(dst_start < dst_end);

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
      assert(num_edges == src_idx.size());

      size_t sum_num_edges = 0;
      assert(MPI_Allreduce(&num_edges, &sum_num_edges, 1,
                        MPI_SIZE_T, MPI_SUM, comm) == MPI_SUCCESS);

      if (sum_num_edges == 0)
        {
          assert(MPI_Comm_free(&comm) == MPI_SUCCESS);
          if (info != MPI_INFO_NULL)
            {
              assert(MPI_Info_free(&info) == MPI_SUCCESS);
            }
          return;
        }
      


      hsize_t dst_blk_ptr_size = 0;
      hdf5::size_edge_attributes(file,
                                 src_pop_name,
                                 dst_pop_name,
                                 hdf5::EDGES,
                                 hdf5::DST_BLK_PTR,
                                 dst_blk_ptr_size);

      hsize_t dst_blk_idx_size = 0;
      hdf5::size_edge_attributes(file,
                                 src_pop_name,
                                 dst_pop_name,
                                 hdf5::EDGES,
                                 hdf5::DST_BLK_IDX,
                                 dst_blk_idx_size);

      hsize_t dst_ptr_size = 0; 
      hdf5::size_edge_attributes(file,
                                 src_pop_name,
                                 dst_pop_name,
                                 hdf5::EDGES,
                                 hdf5::DST_PTR,
                                 dst_ptr_size);

      hsize_t src_idx_size = 0;
      hdf5::size_edge_attributes(file,
                                 src_pop_name,
                                 dst_pop_name,
                                 hdf5::EDGES,
                                 hdf5::SRC_IDX,
                                 src_idx_size);
      
      // exchange allocation data

      vector<size_t> sendbuf_num_blocks(size, num_blocks);
      vector<size_t> recvbuf_num_blocks(size);
      assert(MPI_Allgather(&sendbuf_num_blocks[0], 1, MPI_SIZE_T,
                           &recvbuf_num_blocks[0], 1, MPI_SIZE_T, comm)
             == MPI_SUCCESS);

      vector<size_t> sendbuf_num_dest(size, num_dest);
      vector<size_t> recvbuf_num_dest(size);
      assert(MPI_Allgather(&sendbuf_num_dest[0], 1, MPI_SIZE_T,
                           &recvbuf_num_dest[0], 1, MPI_SIZE_T, comm)
             == MPI_SUCCESS);

      vector<size_t> sendbuf_num_edge(size, num_edges);
      vector<size_t> recvbuf_num_edge(size);
      assert(MPI_Allgather(&sendbuf_num_edge[0], 1, MPI_SIZE_T,
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

      if (num_blocks > 0)
        {
          if (dst_ptr_size > 0)
            {
              dst_blk_ptr[0] += dst_ptr_size-1;
            }
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

      if (rank == last_rank) // only the last rank writes an additional element
        {
          dst_ptr.push_back(num_prj_edges);
        }

      size_t s = src_idx_size;
      for (size_t p = 0; p < rank; ++p)
        {
          s += recvbuf_num_edge[p];
        }

      for (size_t idst = 0; idst < dst_ptr.size(); ++idst)
        {
          dst_ptr[idst] += s;
        }
      
      size_t total_num_blocks=0;
      for (size_t p=0; p<size; p++)
        {
          total_num_blocks = total_num_blocks + recvbuf_num_blocks[p];
        }

      size_t total_num_dests=0;
      for (size_t p=0; p<size; p++)
        {
          total_num_dests = total_num_dests + recvbuf_num_dest[p];
        }

      size_t total_num_edges=0;
      for (size_t p=0; p<size; p++)
        {
          total_num_edges = total_num_edges + recvbuf_num_edge[p];
        }

      mpi::MPI_DEBUG(comm, "append_projection: ", src_pop_name, " -> ", dst_pop_name, ": ",
                     " total_num_dests = ", total_num_dests);


      append_dst_blk_idx
        (
         rank, file,
         src_pop_name, dst_pop_name,
         src_start, src_end, dst_start, dst_end,
         num_blocks, total_num_blocks, dst_blk_idx_size,
         chunk_size, block_size, collective,
         recvbuf_num_blocks,
         dst_blk_idx
         );

      /*
        vector<NODE_IDX_T> v_dst_start(1, dst_start);         
        write(file, path, NODE_IDX_H5_FILE_T, v_dst_start);
      */

      append_dst_blk_ptr
        (
         rank, file,
         src_pop_name, dst_pop_name,
         src_start, src_end, dst_start, dst_end,
         num_blocks, total_num_blocks, dst_blk_ptr_size,
         chunk_size, block_size, collective,
         recvbuf_num_blocks,
         dst_blk_ptr
         );
      

      // write destination pointers
      // # dest. pointers = number of destinations + 1
      append_dst_ptr
        (
         rank, file,
         src_pop_name, dst_pop_name,
         src_start, src_end,
         dst_start, dst_end,
         total_num_dests, dst_ptr_size,
         chunk_size, block_size, 
         collective,
         recvbuf_num_dest,
         dst_ptr
         );

      // write source index
      // # source indexes = number of edges

      append_src_idx
        (
         rank, file,
         src_pop_name, dst_pop_name,
         src_start, src_end,
         dst_start, dst_end,
         total_num_edges, src_idx_size,
         chunk_size, block_size, 
         collective,
         recvbuf_num_edge,
         src_idx
         );

      vector <string> edge_attr_name_spaces;
      map <string, data::NamedAttrVal> edge_attr_map;

      for (auto const& iter : edge_attr_names)
        {
          const string & attr_namespace = iter.first;
          const vector <vector <string> >& attr_names = iter.second;

          data::NamedAttrVal& edge_attr_values = edge_attr_map[attr_namespace];
          
          edge_attr_values.float_values.resize(attr_names[data::AttrVal::attr_index_float].size());
          edge_attr_values.uint8_values.resize(attr_names[data::AttrVal::attr_index_uint8].size());
          edge_attr_values.uint16_values.resize(attr_names[data::AttrVal::attr_index_uint16].size());
          edge_attr_values.uint32_values.resize(attr_names[data::AttrVal::attr_index_uint32].size());
          edge_attr_values.int8_values.resize(attr_names[data::AttrVal::attr_index_int8].size());
          edge_attr_values.int16_values.resize(attr_names[data::AttrVal::attr_index_int16].size());
          edge_attr_values.int32_values.resize(attr_names[data::AttrVal::attr_index_int32].size());

          edge_attr_name_spaces.push_back(attr_namespace);
        }
        
      for (auto const& iter : prj_edge_map)
	{
	  const edge_tuple_t& et = iter.second;
          const vector<NODE_IDX_T>& v = get<0>(et);
          const vector<data::AttrVal>& a = get<1>(et);
          size_t ni=0;
          for (auto const& attr_values : a)
            {
              const string & attr_namespace = edge_attr_name_spaces[ni];
              edge_attr_map[attr_namespace].append(attr_values);
              ni++;
            }
	}
      append_edge_attribute_map<float>(file, src_pop_name, dst_pop_name,
                                       edge_attr_map, edge_attr_names);
      append_edge_attribute_map<uint8_t>(file, src_pop_name, dst_pop_name,
                                         edge_attr_map, edge_attr_names);
      append_edge_attribute_map<uint16_t>(file, src_pop_name, dst_pop_name,
                                          edge_attr_map, edge_attr_names);
      append_edge_attribute_map<uint32_t>(file, src_pop_name, dst_pop_name,
                                          edge_attr_map, edge_attr_names);
      append_edge_attribute_map<int8_t>(file, src_pop_name, dst_pop_name,
                                        edge_attr_map, edge_attr_names);
      append_edge_attribute_map<int16_t>(file, src_pop_name, dst_pop_name,
                                         edge_attr_map, edge_attr_names);
      append_edge_attribute_map<int32_t>(file, src_pop_name, dst_pop_name,
                                         edge_attr_map, edge_attr_names);
        
      // clean-up

      assert(MPI_Comm_free(&comm) == MPI_SUCCESS);
      if (info != MPI_INFO_NULL)
        {
          assert(MPI_Info_free(&info) == MPI_SUCCESS);
        }
      assert(H5Pclose(fapl) >= 0);
    }
  }
}
