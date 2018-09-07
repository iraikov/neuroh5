// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_projection_dataset_selection.cc
///
///  Functions for reading edge information in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2018 Project NeuroH5.
//==============================================================================

#include "debug.hh"

#include "neuroh5_types.hh"
#include "dataset_num_elements.hh"
#include "read_template.hh"
#include "path_names.hh"
#include "rank_range.hh"
#include "read_projection_datasets.hh"
#include "mpi_debug.hh"

#include <iostream>
#include <sstream>
#include <string>

#undef NDEBUG
#include <cassert>

using namespace std;

namespace neuroh5
{
  namespace hdf5
  {
    
    /**************************************************************************
     * Read a subset of the basic DBS graph structure
     *************************************************************************/

    herr_t read_projection_dataset_selection
    (
     MPI_Comm                   comm,
     const std::string&         file_name,
     const std::string&         src_pop_name,
     const std::string&         dst_pop_name,
     const NODE_IDX_T&          src_start,
     const NODE_IDX_T&          dst_start,
     const std::vector<NODE_IDX_T>&  selection,
     DST_PTR_T&                 edge_base,
     vector<NODE_IDX_T>&        selection_dst_idx,
     vector<DST_PTR_T>&         selection_dst_ptr,
     vector<NODE_IDX_T>&        src_idx,
     size_t&                    total_num_edges,
     bool collective
     )
    {
      herr_t ierr = 0;
      unsigned int rank, size;
      assert(MPI_Comm_size(comm, (int*)&size) == MPI_SUCCESS);
      assert(MPI_Comm_rank(comm, (int*)&rank) == MPI_SUCCESS);

      hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
      assert(fapl >= 0);
      assert(H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL) >= 0);

      hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, fapl);
      assert(file >= 0);

      // determine number of blocks in projection
      hsize_t num_blocks = hdf5::dataset_num_elements
         (file, hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_BLK_PTR));
      if (num_blocks > 0)
        num_blocks--;
      
      // determine number of edges in projection
      total_num_edges = hdf5::dataset_num_elements
        (file, hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::SRC_IDX));

      /* Create property list for collective dataset operations. */
      hid_t rapl = H5Pcreate (H5P_DATASET_XFER);
      if (collective)
        {
          ierr = H5Pset_dxpl_mpio (rapl, H5FD_MPIO_COLLECTIVE);
        }

      vector< pair<hsize_t,hsize_t> > bins;
      hsize_t read_blocks = num_blocks;
      
      if (read_blocks > 0)
        {
          // determine which blocks of block_ptr are read by which rank
          mpi::rank_ranges(read_blocks, size, bins);

          // determine start and stop block for the current rank
          hsize_t start, stop;
          
          start = bins[rank].first;
          stop  = start + bins[rank].second;
          
          hsize_t block;
          if (stop > start)
            block = stop - start + 1;
          else
            block = 0;

          DST_BLK_PTR_T block_rebase = 0;

          // read destination block pointers

          vector<DST_BLK_PTR_T> dst_blk_ptr(block, 0);
          
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
          assert(ierr >= 0);
          
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
          vector<NODE_IDX_T> dst_idx;

          
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
          assert(ierr >= 0);

          
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
          
          vector<DST_PTR_T> dst_ptr;
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
          assert(ierr >= 0);
          
          DST_PTR_T dst_rebase = 0;
          
          // Create source index ranges based on selection_dst_ptr
          vector< pair<hsize_t,hsize_t> > src_idx_ranges;
          ATTR_PTR_T selection_dst_ptr_pos = 0;
          if (dst_ptr_block > 0)
            {
              dst_rebase = dst_ptr[0];
              edge_base = dst_rebase;
              for (size_t i = 0; i < dst_ptr.size(); ++i)
                {
                  dst_ptr[i] -= dst_rebase;
                }

              // Create node index in order to determine which edges to read

              for (const NODE_IDX_T& s : selection) 
                {
                  if (s >= dst_start)
                    {
                      auto it = std::find(dst_idx.begin(), dst_idx.end(), s-dst_start);
                      if (it != dst_idx.end())
                        {
                          selection_dst_idx.push_back(s);
                          
                          ptrdiff_t pos = it - dst_idx.begin();
                          
                          hsize_t src_idx_start=dst_ptr[pos];
                          hsize_t src_idx_block=dst_ptr[pos+1]-src_idx_start;
                          
                          src_idx_ranges.push_back(make_pair(src_idx_start, src_idx_block));
                          selection_dst_ptr.push_back(selection_dst_ptr_pos);
                          selection_dst_ptr_pos += src_idx_block;
                        }
                    }
                }
              selection_dst_ptr.push_back(selection_dst_ptr_pos);
            }

          if (src_idx_ranges.size() > 0)
            {
              // allocate buffer and memory dataspace
              src_idx.resize(selection_dst_ptr_pos, 0);
            }

          ierr = hdf5::read_selection<NODE_IDX_T>
            (
             file,
             hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::SRC_IDX),
             NODE_IDX_H5_NATIVE_T,
             src_idx_ranges,
             src_idx,
             rapl
             );
          assert(ierr >= 0);
              
          assert(H5Fclose(file) >= 0);
          assert(H5Pclose(fapl) >= 0);
          ierr = H5Pclose(rapl);
          assert(ierr == 0);
          
        }


      return ierr;
    }

    
  }
}

