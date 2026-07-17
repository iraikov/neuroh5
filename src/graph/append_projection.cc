

#include "neuroh5_types.hh"
#include "path_names.hh"
#include "create_group.hh"
#include "exists_dataset.hh"
#include "append_projection.hh"
#include "write_template.hh"
#include "edge_attributes.hh"
#include "mpe_seq.hh"
#include "mpi_debug.hh"
#include "debug.hh"
#include "throw_assert.hh"

#include <algorithm>
#include <map>
#include <cstdlib>
#include <cstdio>

using namespace std;

namespace neuroh5
{
  namespace graph
  {

    // dst_blk_idx, dst_blk_ptr, and dst_ptr are small (block/pointer
    // metadata, O(num destinations)), so rather than have every I/O
    // rank independently extend and collectively write its own
    // hyperslab.  these three gather to rank 0 (of comm) via
    // MPI_Gatherv and are written by rank 0 alone, in a single
    // hyperslab.  All ranks still participate in the collective
    // create/open/extend calls (required when the file is opened via
    // H5Pset_fapl_mpio), but only rank 0 contributes a real
    // (non-empty) write selection.  src_idx (and edge attributes),
    // which can be large, use per-rank distributed collective
    // write. Each rank's hyperslab there is non-overlapping, which is
    // what collective I/O requires.
    void append_dst_blk_idx
    (
     MPI_Comm                  comm,
     size_t                    rank,
     size_t                    size,
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
      throw_assert_nomsg(lcpl >= 0);
      throw_assert_nomsg(H5Pset_create_intermediate_group(lcpl, 1) >= 0);
      hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
      throw_assert_nomsg(H5Pset_layout(dcpl, H5D_CHUNKED) >= 0);
      hid_t wapl = H5P_DEFAULT;
#ifdef HDF5_IS_PARALLEL
      if (collective)
	{
	  wapl = H5Pcreate(H5P_DATASET_XFER);
	  throw_assert_nomsg(wapl >= 0);
	  throw_assert_nomsg(H5Pset_dxpl_mpio(wapl, H5FD_MPIO_COLLECTIVE) >= 0);
	}
#endif

      hsize_t chunk = chunk_size;
      hsize_t maxdims[1] = {H5S_UNLIMITED};
      hsize_t zerodims[1] = {0};

      // do a sanity check on the input
      throw_assert_nomsg(src_start < src_end);
      throw_assert_nomsg(dst_start < dst_end);

      // Gather every rank's (already globally-offset) local dst_blk_idx
      // values onto rank 0.
      vector<int> counts(size, 0), displs(size + 1, 0);
      for (size_t p = 0; p < size; ++p)
        {
          counts[p] = (int)recvbuf_num_blocks[p];
          displs[p + 1] = displs[p] + counts[p];
        }
      vector<NODE_IDX_T> gathered;
      if (rank == 0)
        {
          gathered.resize(displs[size]);
        }
      throw_assert_nomsg(MPI_Gatherv(dst_blk_idx.data(), (int)dst_blk_idx.size(), MPI_NODE_IDX_T,
                                     rank == 0 ? gathered.data() : NULL,
                                     &counts[0], &displs[0], MPI_NODE_IDX_T,
                                     0, comm) == MPI_SUCCESS);

      string path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_BLK_IDX);
      hsize_t dst_blk_idx_dims = total_num_blocks, one=1;

      hid_t dset, fspace;

      if (!(hdf5::exists_dataset (file, path) > 0))
        {
          fspace = H5Screate_simple(1, zerodims, maxdims);
          throw_assert_nomsg(fspace >= 0);
	  throw_assert_nomsg(H5Pset_chunk(dcpl, 1, &chunk ) >= 0);
#ifdef H5_HAS_PARALLEL_DEFLATE
          throw_assert_nomsg(H5Pset_deflate(dcpl, 9) >= 0);
#endif

          dset = H5Dcreate2(file, path.c_str(), NODE_IDX_H5_FILE_T, fspace,
                            lcpl, dcpl, H5P_DEFAULT);
          throw_assert_nomsg(dset >= 0);
        }
      else
        {
          dset = H5Dopen2 (file, path.c_str(), H5P_DEFAULT);
          throw_assert_nomsg(dset >= 0);

          fspace = H5Dget_space(dset);
          throw_assert_nomsg(fspace >= 0);
        }

      hsize_t dst_blk_idx_start = dst_blk_idx_size;
      hsize_t dst_blk_idx_newsize = dst_blk_idx_start + dst_blk_idx_dims;
      if (dst_blk_idx_newsize > 0)
        {
          herr_t ierr = H5Dset_extent (dset, &dst_blk_idx_newsize);
          throw_assert_nomsg(ierr >= 0);
        }
      throw_assert_nomsg(H5Sclose(fspace) >= 0);

      hsize_t block = (rank == 0) ? (hsize_t)gathered.size() : 0;

      hid_t mspace  = H5Screate_simple(1, &block, &block);
      throw_assert_nomsg(mspace >= 0);
      throw_assert_nomsg(H5Sselect_all(mspace) >= 0);

      fspace = H5Dget_space(dset);
      throw_assert_nomsg(fspace >= 0);
      if (block > 0)
	{
	  hsize_t start = dst_blk_idx_start;
	  throw_assert_nomsg(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
				     &one, &block) >= 0);
	}
      else
	{
          throw_assert_nomsg(H5Sselect_none(fspace) >= 0);
	}
      NODE_IDX_T dummy = 0;
      throw_assert_nomsg(H5Dwrite(dset, NODE_IDX_H5_NATIVE_T, mspace, fspace,
		      wapl, block > 0 ? &gathered[0] : &dummy) >= 0);

      // clean-up
      throw_assert_nomsg(H5Dclose(dset) >= 0);
      throw_assert_nomsg(H5Sclose(mspace) >= 0);
      throw_assert_nomsg(H5Sclose(fspace) >= 0);

      throw_assert_nomsg(H5Pclose(lcpl) >= 0);
      throw_assert_nomsg(H5Pclose(dcpl) >= 0);
      throw_assert_nomsg(H5Pclose(wapl) >= 0);

    }


    void append_dst_blk_ptr
    (
     MPI_Comm                  comm,
     size_t                    rank,
     size_t                    size,
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
     vector<size_t>&           recvbuf_num_blocks_padded,
     vector<DST_BLK_PTR_T>&    dst_blk_ptr
     )
    {
      hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
      throw_assert_nomsg(lcpl >= 0);
      throw_assert_nomsg(H5Pset_create_intermediate_group(lcpl, 1) >= 0);
      hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
      throw_assert_nomsg(H5Pset_layout(dcpl, H5D_CHUNKED) >= 0);
      hid_t wapl = H5P_DEFAULT;
#ifdef HDF5_IS_PARALLEL
      if (collective)
	{
	  wapl = H5Pcreate(H5P_DATASET_XFER);
	  throw_assert_nomsg(wapl >= 0);
	  throw_assert_nomsg(H5Pset_dxpl_mpio(wapl, H5FD_MPIO_COLLECTIVE) >= 0);
	}
#endif

      hsize_t chunk = chunk_size;
      hsize_t maxdims[1] = {H5S_UNLIMITED};
      hsize_t zerodims[1] = {0};


      hsize_t dst_blk_ptr_start = 0;
      if (dst_blk_ptr_size > 0)
        {
          dst_blk_ptr_start = dst_blk_ptr_size - 1;
        }

      // Gather every rank's (already globally-offset) local dst_blk_ptr
      // values onto rank 0. recvbuf_num_blocks_padded already includes the
      // +1 sentinel element on whichever rank contributed it.
      vector<int> counts(size, 0), displs(size + 1, 0);
      for (size_t p = 0; p < size; ++p)
        {
          counts[p] = (int)recvbuf_num_blocks_padded[p];
          displs[p + 1] = displs[p] + counts[p];
        }
      vector<DST_BLK_PTR_T> gathered;
      if (rank == 0)
        {
          gathered.resize(displs[size]);
        }
      throw_assert_nomsg(MPI_Gatherv(dst_blk_ptr.data(), (int)dst_blk_ptr.size(), MPI_UINT64_T,
                                     rank == 0 ? gathered.data() : NULL,
                                     &counts[0], &displs[0], MPI_UINT64_T,
                                     0, comm) == MPI_SUCCESS);

      string path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_BLK_PTR);

      hid_t dset, fspace;

      hsize_t dst_blk_ptr_dims = (hsize_t)total_num_blocks+1, one=1;
      if (!(hdf5::exists_dataset (file, path) > 0))
        {
          fspace = H5Screate_simple(1, zerodims, maxdims);
          throw_assert_nomsg(fspace >= 0);

	  throw_assert_nomsg(H5Pset_chunk(dcpl, 1, &chunk ) >= 0);
#ifdef H5_HAS_PARALLEL_DEFLATE
          throw_assert_nomsg(H5Pset_deflate(dcpl, 9) >= 0);
#endif
          dset = H5Dcreate2 (file, path.c_str(), DST_BLK_PTR_H5_FILE_T,
                             fspace, lcpl, dcpl, H5P_DEFAULT);
          throw_assert_nomsg(H5Sclose(fspace) >= 0);
        }
      else
        {
          dset = H5Dopen2 (file, path.c_str(), H5P_DEFAULT);
          throw_assert_nomsg(dset >= 0);
        }

      hsize_t dst_blk_ptr_newsize = dst_blk_ptr_start + dst_blk_ptr_dims;
      if (dst_blk_ptr_newsize > 0)
        {
          herr_t ierr = H5Dset_extent (dset, &dst_blk_ptr_newsize);
          throw_assert_nomsg(ierr >= 0);
        }

      hsize_t block = (rank == 0) ? (hsize_t)gathered.size() : 0;

      hid_t mspace  = H5Screate_simple(1, &block, &block);
      throw_assert_nomsg(mspace >= 0);
      throw_assert_nomsg(H5Sselect_all(mspace) >= 0);

      fspace = H5Dget_space(dset);
      throw_assert_nomsg(fspace >= 0);
      if (block > 0)
        {
          hsize_t start = dst_blk_ptr_start;
          throw_assert_nomsg(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                     &one, &block) >= 0);
        }
      else
        {
          throw_assert_nomsg(H5Sselect_none(fspace) >= 0);
        }
      DST_BLK_PTR_T dummy = 0;
      throw_assert_nomsg(H5Dwrite(dset, DST_BLK_PTR_H5_NATIVE_T, mspace, fspace,
                      wapl, block > 0 ? &gathered[0] : &dummy) >= 0);

      throw_assert_nomsg(H5Dclose(dset) >= 0);
      throw_assert_nomsg(H5Sclose(mspace) >= 0);
      throw_assert_nomsg(H5Sclose(fspace) >= 0);

      throw_assert_nomsg(H5Pclose(lcpl) >= 0);
      throw_assert_nomsg(H5Pclose(dcpl) >= 0);
      throw_assert_nomsg(H5Pclose(wapl) >= 0);
    }


    void append_dst_ptr
    (
     MPI_Comm                  comm,
     size_t                    rank,
     size_t                    size,
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
     vector<size_t>&           recvbuf_num_dest_padded,
     vector<DST_PTR_T>&    dst_ptr
     )
    {

      hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
      throw_assert_nomsg(lcpl >= 0);
      throw_assert_nomsg(H5Pset_create_intermediate_group(lcpl, 1) >= 0);
      hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
      throw_assert_nomsg(H5Pset_layout(dcpl, H5D_CHUNKED) >= 0);
      hid_t wapl = H5P_DEFAULT;
#ifdef HDF5_IS_PARALLEL
      if (collective)
	{
	  wapl = H5Pcreate(H5P_DATASET_XFER);
	  throw_assert_nomsg(wapl >= 0);
	  throw_assert_nomsg(H5Pset_dxpl_mpio(wapl, H5FD_MPIO_COLLECTIVE) >= 0);
	}
#endif

      hsize_t chunk = chunk_size;
      hsize_t maxdims[1] = {H5S_UNLIMITED};
      hsize_t zerodims[1] = {0};

      // Gather every rank's (already globally-offset) local dst_ptr values
      // onto rank 0. recvbuf_num_dest_padded already includes the +1
      // sentinel element on whichever rank contributed it.
      vector<int> counts(size, 0), displs(size + 1, 0);
      for (size_t p = 0; p < size; ++p)
        {
          counts[p] = (int)recvbuf_num_dest_padded[p];
          displs[p + 1] = displs[p] + counts[p];
        }
      vector<DST_PTR_T> gathered;
      if (rank == 0)
        {
          gathered.resize(displs[size]);
        }
      throw_assert_nomsg(MPI_Gatherv(dst_ptr.data(), (int)dst_ptr.size(), MPI_UINT64_T,
                                     rank == 0 ? gathered.data() : NULL,
                                     &counts[0], &displs[0], MPI_UINT64_T,
                                     0, comm) == MPI_SUCCESS);

      string path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_PTR);
      hsize_t dst_ptr_dims = total_num_dests+1, one=1;

      hid_t dset, fspace;

      if (!(hdf5::exists_dataset (file, path) > 0))
        {
          fspace = H5Screate_simple(1, zerodims, maxdims);
          throw_assert_nomsg(fspace >= 0);

	  throw_assert_nomsg(H5Pset_chunk(dcpl, 1, &chunk ) >= 0);
#ifdef H5_HAS_PARALLEL_DEFLATE
          throw_assert_nomsg(H5Pset_deflate(dcpl, 9) >= 0);
#endif
          dset = H5Dcreate2 (file, path.c_str(), DST_PTR_H5_FILE_T,
                             fspace, lcpl, dcpl, H5P_DEFAULT);
          throw_assert_nomsg(dset >= 0);
          throw_assert_nomsg(H5Sclose(fspace) >= 0);
        }
      else
        {
          dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
          throw_assert_nomsg(dset >= 0);

        }

      fspace = H5Dget_space(dset);
      throw_assert_nomsg(fspace >= 0);
      hsize_t dst_ptr_start = 0;
      if (dst_ptr_size > 0)
        {
          dst_ptr_start = dst_ptr_size-1;
        }

      throw_assert_nomsg(H5Sclose(fspace) >= 0);

      hsize_t dst_ptr_newsize = dst_ptr_start + dst_ptr_dims;
      if (dst_ptr_newsize > 0)
        {
          herr_t ierr = H5Dset_extent (dset, &dst_ptr_newsize);
          throw_assert_nomsg(ierr >= 0);
        }

      hsize_t block = (rank == 0) ? (hsize_t)gathered.size() : 0;
      hid_t mspace = H5Screate_simple(1, &block, &block);
      throw_assert_nomsg(mspace >= 0);
      throw_assert_nomsg(H5Sselect_all(mspace) >= 0);

      fspace = H5Dget_space(dset);
      throw_assert_nomsg(fspace >= 0);

      if (block > 0)
        {
          hsize_t start = dst_ptr_start;
          throw_assert_nomsg(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                     &one, &block) >= 0);
        }
      else
        {
          throw_assert_nomsg(H5Sselect_none(fspace) >= 0);
        }

      DST_PTR_T dummy = 0;
      throw_assert_nomsg(H5Dwrite(dset, DST_PTR_H5_NATIVE_T, mspace, fspace,
                      wapl, block > 0 ? &gathered[0] : &dummy) >= 0);

      throw_assert_nomsg(H5Dclose(dset) >= 0);
      throw_assert_nomsg(H5Sclose(mspace) >= 0);
      throw_assert_nomsg(H5Sclose(fspace) >= 0);

      throw_assert_nomsg(H5Pclose(lcpl) >= 0);
      throw_assert_nomsg(H5Pclose(dcpl) >= 0);
      throw_assert_nomsg(H5Pclose(wapl) >= 0);

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
      throw_assert_nomsg(lcpl >= 0);
      throw_assert_nomsg(H5Pset_create_intermediate_group(lcpl, 1) >= 0);
      hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
      throw_assert_nomsg(H5Pset_layout(dcpl, H5D_CHUNKED) >= 0);
      hid_t wapl = H5P_DEFAULT;
#ifdef HDF5_IS_PARALLEL
      if (collective)
	{
	  wapl = H5Pcreate(H5P_DATASET_XFER);
	  throw_assert_nomsg(wapl >= 0);
	  throw_assert_nomsg(H5Pset_dxpl_mpio(wapl, H5FD_MPIO_COLLECTIVE) >= 0);
	}
#endif
      
      hsize_t chunk = chunk_size;
      hsize_t maxdims[1] = {H5S_UNLIMITED};
      hsize_t zerodims[1] = {0};
    

      string path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::SRC_IDX);
      hsize_t src_idx_dims = total_num_edges, one=1;

      hid_t dset, fspace;
      
      if (!(hdf5::exists_dataset (file, path) > 0))
        {
          fspace = H5Screate_simple(1, zerodims, maxdims);
          throw_assert_nomsg(fspace >= 0);

	  throw_assert_nomsg(H5Pset_chunk(dcpl, 1, &chunk ) >= 0);
#ifdef H5_HAS_PARALLEL_DEFLATE
          throw_assert_nomsg(H5Pset_deflate(dcpl, 9) >= 0);
#endif
          
          dset = H5Dcreate2 (file, path.c_str(), NODE_IDX_H5_FILE_T,
                             fspace, lcpl, dcpl, H5P_DEFAULT);
          throw_assert_nomsg(dset >= 0);
        }
      else
        {
          dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
          throw_assert_nomsg(dset >= 0);
          fspace = H5Dget_space(dset);
          throw_assert_nomsg(fspace >= 0);
        }

      hsize_t src_idx_start = src_idx_size;
      throw_assert_nomsg(H5Sclose(fspace) >= 0);

      hsize_t src_idx_newsize = src_idx_start + src_idx_dims;
      if (src_idx_newsize > 0)
        {
          herr_t ierr = H5Dset_extent (dset, &src_idx_newsize);
          throw_assert_nomsg(ierr >= 0);
        }

      hsize_t block = (hsize_t) src_idx.size();
      hid_t mspace = H5Screate_simple(1, &block, &block);
      throw_assert_nomsg(mspace >= 0);
      throw_assert_nomsg(H5Sselect_all(mspace) >= 0);

      fspace = H5Dget_space(dset);
      throw_assert_nomsg(fspace >= 0);

      hsize_t start = 0;
      if (block > 0)
        {
          start = src_idx_start;
          for (size_t p = 0; p < rank; ++p)
            {
              start += recvbuf_num_edge[p];
            }
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

      throw_assert_nomsg(H5Pclose(lcpl) >= 0);
      throw_assert_nomsg(H5Pclose(dcpl) >= 0);
      throw_assert_nomsg(H5Pclose(wapl) >= 0);
    }
    
    
    void append_projection
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
     const hsize_t             chunk_size,
     const hsize_t             block_size,
     const bool collective
     )
    {
      MPI_Request request;
      
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
      throw_assert_nomsg(num_edges == src_idx.size());

      size_t sum_num_edges = 0;
      throw_assert(MPI_Iallreduce(&num_edges, &sum_num_edges, 1,
                                  MPI_SIZE_T, MPI_SUM,
                                  comm, &request) == MPI_SUCCESS,
                   "append_projection: error in MPI_Iallreduce");
      throw_assert(MPI_Wait(&request, MPI_STATUS_IGNORE) == MPI_SUCCESS,
                   "append_projection: error in MPI_Wait");
      
      if (sum_num_edges == 0)
        {
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
      vector<size_t> recvbuf_num_blocks(size, 0);
      throw_assert(MPI_Iallgather(&sendbuf_num_blocks[0], 1, MPI_SIZE_T,
                                  &recvbuf_num_blocks[0], 1, MPI_SIZE_T,
                                  comm, &request)
                   == MPI_SUCCESS,
                   "append_projection: error in MPI_Iallgather");
      throw_assert(MPI_Wait(&request, MPI_STATUS_IGNORE) == MPI_SUCCESS,
                   "append_projection: error in MPI_Wait");

      vector<size_t> sendbuf_num_dest(size, num_dest);
      vector<size_t> recvbuf_num_dest(size, 0);
      throw_assert(MPI_Iallgather(&sendbuf_num_dest[0], 1, MPI_SIZE_T,
                                  &recvbuf_num_dest[0], 1, MPI_SIZE_T,
                                  comm, &request)
                   == MPI_SUCCESS,
                   "append_projection: error in MPI_Iallgather");
      throw_assert(MPI_Wait(&request, MPI_STATUS_IGNORE) == MPI_SUCCESS,
                   "append_projection: error in MPI_Wait");

      vector<size_t> sendbuf_num_edge(size, num_edges);
      vector<size_t> recvbuf_num_edge(size, 0);
      throw_assert(MPI_Iallgather(&sendbuf_num_edge[0], 1, MPI_SIZE_T,
                                  &recvbuf_num_edge[0], 1, MPI_SIZE_T,
                                  comm, &request)
                   == MPI_SUCCESS,
                   "append_projection: error in MPI_Iallgather");
      throw_assert(MPI_Wait(&request, MPI_STATUS_IGNORE) == MPI_SUCCESS,
                   "append_projection: error in MPI_Wait");

      // determine last rank that has data
      // (a backward-counting unsigned loop here previously underflowed
      // when no lower-numbered rank had data, reading past the start of
      // recvbuf_num_blocks; walking forward avoids that entirely)
      size_t last_rank = 0;
      for (size_t p = 0; p < size; ++p)
	{
	  if (recvbuf_num_blocks[p] > 0)
	    {
	      last_rank = p;
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

      // dst_blk_ptr and dst_ptr each carry one extra (+1 sentinel) element
      // on last_rank; pad the corresponding per-rank counts so the Gatherv
      // in append_dst_blk_ptr/append_dst_ptr accounts for it.
      vector<size_t> recvbuf_num_blocks_padded(recvbuf_num_blocks);
      recvbuf_num_blocks_padded[last_rank] += 1;
      vector<size_t> recvbuf_num_dest_padded(recvbuf_num_dest);
      recvbuf_num_dest_padded[last_rank] += 1;

      append_dst_blk_idx
        (
         comm, rank, size, file,
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
         comm, rank, size, file,
         src_pop_name, dst_pop_name,
         src_start, src_end, dst_start, dst_end,
         num_blocks, total_num_blocks, dst_blk_ptr_size,
         chunk_size, block_size, collective,
         recvbuf_num_blocks_padded,
         dst_blk_ptr
         );


      // write destination pointers
      // # dest. pointers = number of destinations + 1
      append_dst_ptr
        (
         comm, rank, size, file,
         src_pop_name, dst_pop_name,
         src_start, src_end,
         dst_start, dst_end,
         total_num_dests, dst_ptr_size,
         chunk_size, block_size,
         collective,
         recvbuf_num_dest_padded,
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

      for (auto const& iter : edge_attr_index)
        {
          const string & attr_namespace = iter.first;
          const data::AttrIndex& attr_index  = iter.second.second;

          data::NamedAttrVal& edge_attr = edge_attr_map[attr_namespace];

          edge_attr.float_values.resize(attr_index.size_attr_index<float>());
          edge_attr.uint8_values.resize(attr_index.size_attr_index<uint8_t>());
          edge_attr.uint16_values.resize(attr_index.size_attr_index<uint16_t>());
          edge_attr.uint32_values.resize(attr_index.size_attr_index<uint32_t>());
          edge_attr.int8_values.resize(attr_index.size_attr_index<int8_t>());
          edge_attr.int16_values.resize(attr_index.size_attr_index<int16_t>());
          edge_attr.int32_values.resize(attr_index.size_attr_index<int32_t>());

          edge_attr_name_spaces.push_back(attr_namespace);
        }
        
      for (auto const& iter : prj_edge_map)
	{
	  const edge_tuple_t& et = iter.second;
          //const vector<NODE_IDX_T>& v = get<0>(et);
          const vector<data::AttrVal>& va = get<1>(et);
          size_t ni=0;
          for (auto const& a : va)
            {
              const string & attr_namespace = edge_attr_name_spaces[ni];
              auto & edge_attr = edge_attr_map[attr_namespace];
              edge_attr.append(a);
              ni++;
            }
	}
      throw_assert_nomsg(MPI_Barrier(comm) == MPI_SUCCESS);

      append_edge_attribute_map<float>(comm, file, src_pop_name, dst_pop_name,
                                       edge_attr_map, edge_attr_index, chunk_size);
      append_edge_attribute_map<uint8_t>(comm, file, src_pop_name, dst_pop_name,
                                         edge_attr_map, edge_attr_index, chunk_size);
      append_edge_attribute_map<uint16_t>(comm, file, src_pop_name, dst_pop_name,
                                          edge_attr_map, edge_attr_index, chunk_size);
      append_edge_attribute_map<uint32_t>(comm, file, src_pop_name, dst_pop_name,
                                          edge_attr_map, edge_attr_index, chunk_size);
      append_edge_attribute_map<int8_t>(comm, file, src_pop_name, dst_pop_name,
                                        edge_attr_map, edge_attr_index, chunk_size);
      append_edge_attribute_map<int16_t>(comm, file, src_pop_name, dst_pop_name,
                                         edge_attr_map, edge_attr_index, chunk_size);
      append_edge_attribute_map<int32_t>(comm, file, src_pop_name, dst_pop_name,
                                         edge_attr_map, edge_attr_index, chunk_size);
        
      // clean-up
      throw_assert_nomsg(MPI_Barrier(comm) == MPI_SUCCESS);
    }
  }
}
