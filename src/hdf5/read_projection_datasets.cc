// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_projection_datasets.cc
///
///  Functions for reading edge information in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2020 Project NeuroH5.
//==============================================================================

#include <iostream>
#include <sstream>
#include <string>

#include "debug.hh"

#include "neuroh5_types.hh"
#include "dataset_num_elements.hh"
#include "read_template.hh"
#include "path_names.hh"
#include "rank_range.hh"
#include "read_projection_datasets.hh"
#include "mpi_debug.hh"
#include "throw_assert.hh"


using namespace std;

namespace neuroh5
{
  namespace hdf5
  {
    
    /**************************************************************************
     * Read the basic DBS graph structure
     *************************************************************************/

    herr_t read_projection_datasets
    (
     MPI_Comm                   comm,
     const std::string&         file_name,
     const std::string&         src_pop_name,
     const std::string&         dst_pop_name,
     DST_BLK_PTR_T&             block_base,
     DST_PTR_T&                 edge_base,
     vector<DST_BLK_PTR_T>&     dst_blk_ptr,
     vector<NODE_IDX_T>&        dst_idx,
     vector<DST_PTR_T>&         dst_ptr,
     vector<NODE_IDX_T>&        src_idx,
     size_t&                    total_num_edges,
     hsize_t&                   total_read_blocks,
     hsize_t&                   local_read_blocks,
     size_t                     offset,
     size_t                     numitems,
     bool collective
     )
    {
      herr_t ierr = 0;
      unsigned int rank, size;
      throw_assert_nomsg(MPI_Comm_size(comm, (int*)&size) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Comm_rank(comm, (int*)&rank) == MPI_SUCCESS);


      hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
      throw_assert_nomsg(fapl >= 0);
#ifdef HDF5_IS_PARALLEL
      throw_assert_nomsg(H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL) >= 0);
#endif
      
      hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, fapl);
      throw_assert_nomsg(file >= 0);

      // determine number of blocks in projection
      hsize_t num_blocks = hdf5::dataset_num_elements
         (file, hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_BLK_PTR));
      if (num_blocks > 0)
        num_blocks--;
      
      // determine number of edges in projection
      total_num_edges = hdf5::dataset_num_elements
        (file, hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::SRC_IDX));


      vector< pair<hsize_t,hsize_t> > bins;
      hsize_t read_blocks = 0;

      if (numitems > 0)
        {
          if (offset < num_blocks)
            {
              read_blocks = min((hsize_t)numitems, num_blocks-offset);
            }
        }
      else
        {
          read_blocks = num_blocks;
        }

      total_read_blocks = read_blocks;
      
      if (read_blocks > 0)
        {
          /* Create property list for collective dataset operations. */
          hid_t rapl = H5Pcreate (H5P_DATASET_XFER);
#ifdef HDF5_IS_PARALLEL
          if (collective)
            {
              ierr = H5Pset_dxpl_mpio (rapl, H5FD_MPIO_COLLECTIVE);
              throw_assert(ierr >= 0,
                           "read_projection_datasets: error in H5Pset_dxpl_mpio");
            }
#endif
          
          // determine which blocks of block_ptr are read by which rank
          mpi::rank_ranges(read_blocks, size, bins);

          // determine start and stop block for the current rank
          hsize_t start, stop;
          
          start = bins[rank].first + offset;
          stop  = start + bins[rank].second;
          block_base = start;
          
          hsize_t block;
          if (stop > start)
            block = stop - start + 1;
          else
            block = 0;
          if (block > 0)
            {
              local_read_blocks = block-1;
            }
          else
            {
              local_read_blocks = 0;
            }
          

          DST_BLK_PTR_T block_rebase = 0;

          // read destination block pointers

          // allocate buffer and memory dataspace
          if (block > 0)
            {
              dst_blk_ptr.resize(block, 0);
            }
          
          ierr = hdf5::read<DST_BLK_PTR_T>
            (
             file,
             hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_BLK_PTR),
             start,
             block,
             DST_BLK_PTR_H5_NATIVE_T,
             dst_blk_ptr,
             rapl
             );
          throw_assert_nomsg(ierr >= 0);
          
          // rebase the block_ptr array to local offsets
          // REBASE is going to be the start offset for the hyperslab
          
      
          if (block > 0)
            {
              block_rebase = dst_blk_ptr[0];
          
              for (size_t i = 0; i < dst_blk_ptr.size(); ++i)
                {
                  dst_blk_ptr[i] -= block_rebase;
                }
            }
          else
            {
              block_rebase = 0;
            }

          // read destination block indices
          hsize_t dst_idx_block=0;
          
          if (dst_blk_ptr.size() > 0)
            dst_idx_block = block-1;
          else
            dst_idx_block = 0;
          if (dst_idx_block > 0)
            {
              dst_idx.resize(dst_idx_block, 0);
            }
          
          
          ierr = hdf5::read<NODE_IDX_T>
            (
             file,
             hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_BLK_IDX),
             start,
             dst_idx_block,
             NODE_IDX_H5_NATIVE_T,
             dst_idx,
             rapl
             );
          throw_assert_nomsg(ierr >= 0);

          // read destination pointers
          hsize_t dst_ptr_block=0, dst_ptr_start=0;
          if (dst_blk_ptr.size() > 0)
            {
              dst_ptr_start = (hsize_t)block_rebase;
              dst_ptr_block = (hsize_t)(dst_blk_ptr.back() - dst_blk_ptr.front());
              if  (stop < num_blocks)
                {
                  dst_ptr_block ++;
                }
            }
          

          if (dst_ptr_block > 0)
            {
              dst_ptr.resize(dst_ptr_block, 0);
            }
          
          ierr = hdf5::read<DST_PTR_T>
            (
             file,
             hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_PTR),
             dst_ptr_start,
             dst_ptr_block,
             DST_PTR_H5_NATIVE_T,
             dst_ptr,
             rapl
             );
          throw_assert_nomsg(ierr >= 0);
          
          DST_PTR_T dst_rebase = 0;
          
          hsize_t src_idx_block=0, src_idx_start=0;
          if (dst_ptr_block > 0)
            {
              dst_rebase = dst_ptr[0];
              edge_base = dst_rebase;
              for (size_t i = 0; i < dst_ptr.size(); ++i)
                {
                  dst_ptr[i] -= dst_rebase;
                }
              
              // read source indices
              src_idx_start = dst_rebase;
              src_idx_block = (hsize_t)(dst_ptr.back() - dst_ptr.front());

              // allocate buffer and memory dataspace
              if (src_idx_block > 0)
                {
                  src_idx.resize(src_idx_block, 0);
                }
            }

          ierr = hdf5::read<NODE_IDX_T>
            (
             file,
             hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::SRC_IDX),
             src_idx_start,
             src_idx_block,
             NODE_IDX_H5_NATIVE_T,
             src_idx,
             rapl
             );
          throw_assert_nomsg(ierr >= 0);

          throw_assert_nomsg(H5Pclose(rapl) >= 0);
        }
      throw_assert_nomsg(H5Fclose(file) >= 0);
      throw_assert_nomsg(H5Pclose(fapl) >= 0);

      return ierr;
    }


    herr_t read_projection_node_datasets
    (
     MPI_Comm                   comm,
     const std::string&         file_name,
     const std::string&         src_pop_name,
     const std::string&         dst_pop_name,
     DST_BLK_PTR_T&             block_base,
     DST_PTR_T&                 edge_base,
     vector<DST_BLK_PTR_T>&     dst_blk_ptr,
     vector<NODE_IDX_T>&        dst_idx,
     vector<DST_PTR_T>&         dst_ptr,
     bool collective
     )
    {
      herr_t ierr = 0;
      unsigned int rank, size;
      throw_assert_nomsg(MPI_Comm_size(comm, (int*)&size) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Comm_rank(comm, (int*)&rank) == MPI_SUCCESS);


      hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
      throw_assert_nomsg(fapl >= 0);
#ifdef HDF5_IS_PARALLEL
      throw_assert_nomsg(H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL) >= 0);
#endif

      hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, fapl);
      throw_assert_nomsg(file >= 0);

      // determine number of blocks in projection
      hsize_t num_blocks = hdf5::dataset_num_elements
         (file, hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_BLK_PTR));
      if (num_blocks > 0)
        num_blocks--;

      vector< pair<hsize_t,hsize_t> > bins;
      hsize_t read_blocks = num_blocks;
      
      if (read_blocks > 0)
        {
          /* Create property list for collective dataset operations. */
          hid_t rapl = H5Pcreate (H5P_DATASET_XFER);
#ifdef HDF5_IS_PARALLEL
          if (collective)
            {
              ierr = H5Pset_dxpl_mpio (rapl, H5FD_MPIO_COLLECTIVE);
              throw_assert(ierr >= 0,
                           "read_projection_node_datasets: error in H5Pset_dxpl_mpio");
            }
#endif

          // determine which blocks of block_ptr are read by which rank
          mpi::rank_ranges(read_blocks, size, bins);

          // determine start and stop block for the current rank
          hsize_t start, stop;
          
          start = bins[rank].first;
          stop  = start + bins[rank].second;
          block_base = start;
          
          hsize_t block;
          if (stop > start)
            block = stop - start + 1;
          else
            block = 0;

          DST_BLK_PTR_T block_rebase = 0;

          // read destination block pointers

          // allocate buffer and memory dataspace
          if (block > 0)
            {
              dst_blk_ptr.resize(block, 0);
            }
          
          ierr = hdf5::read<DST_BLK_PTR_T>
            (
             file,
             hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_BLK_PTR),
             start,
             block,
             DST_BLK_PTR_H5_NATIVE_T,
             dst_blk_ptr,
             rapl
             );
          throw_assert_nomsg(ierr >= 0);
          
          // rebase the block_ptr array to local offsets
          // REBASE is going to be the start offset for the hyperslab
          
      
          if (block > 0)
            {
              block_rebase = dst_blk_ptr[0];
          
              for (size_t i = 0; i < dst_blk_ptr.size(); ++i)
                {
                  dst_blk_ptr[i] -= block_rebase;
                }
            }
          else
            {
              block_rebase = 0;
            }

          // read destination block indices
          hsize_t dst_idx_block=0;
          
          if (dst_blk_ptr.size() > 0)
            dst_idx_block = block-1;
          else
            dst_idx_block = 0;
          if (dst_idx_block > 0)
            {
              dst_idx.resize(dst_idx_block, 0);
            }
          
          
          ierr = hdf5::read<NODE_IDX_T>
            (
             file,
             hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_BLK_IDX),
             start,
             dst_idx_block,
             NODE_IDX_H5_NATIVE_T,
             dst_idx,
             rapl
             );
          throw_assert_nomsg(ierr >= 0);

          // read destination pointers
          hsize_t dst_ptr_block=0, dst_ptr_start=0;
          if (dst_blk_ptr.size() > 0)
            {
              dst_ptr_start = (hsize_t)block_rebase;
              dst_ptr_block = (hsize_t)(dst_blk_ptr.back() - dst_blk_ptr.front());
              if  (stop < num_blocks)
                {
                  dst_ptr_block ++;
                }
            }

          if (dst_ptr_block > 0)
            {
              dst_ptr.resize(dst_ptr_block, 0);
            }
          
          ierr = hdf5::read<DST_PTR_T>
            (
             file,
             hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_PTR),
             dst_ptr_start,
             dst_ptr_block,
             DST_PTR_H5_NATIVE_T,
             dst_ptr,
             rapl
             );
          throw_assert_nomsg(ierr >= 0);

          
          throw_assert_nomsg(H5Pclose(rapl) >= 0);
        }
      throw_assert_nomsg(H5Fclose(file) >= 0);
      throw_assert_nomsg(H5Pclose(fapl) >= 0);

      return ierr;
    }

    
  }
}

