// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_projection.cc
///
///  Functions for reading edge information in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2017 Project Neurograph.
//==============================================================================

#include "neuroh5_types.hh"
#include "dataset_num_elements.hh"
#include "read_template.hh"
#include "path_names.hh"
#include "rank_range.hh"
#include "debug.hh"
#include "read_projection.hh"


#include <iostream>
#include <sstream>
#include <string>

#undef NDEBUG
#include <cassert>

using namespace std;

namespace neuroh5
{
  namespace graph
  {

    // Returns the number of blocks in a projection
    hsize_t projection_num_blocks
    (
     MPI_Comm                   comm,
     const std::string&         file_name,
     const std::string&         src_pop_name,
     const std::string&         dst_pop_name
     )
    {
      herr_t ierr = 0;
      unsigned int rank, size;
      assert(MPI_Comm_size(comm, (int*)&size) >= 0);
      assert(MPI_Comm_rank(comm, (int*)&rank) >= 0);

      hsize_t num_blocks = 0;

      if (rank == 0)
        {
          hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
          assert(file >= 0);
          
          // determine number of blocks in projection
          num_blocks = hdf5::dataset_num_elements
            (file, hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_BLK_PTR)) - 1;
        }

      assert(MPI_Bcast(&num_blocks, 1, MPI_UINT64_T, 0, comm) == MPI_SUCCESS);
      
      return num_blocks;
    }
    
    /**************************************************************************
     * Read the basic DBS graph structure
     *************************************************************************/

    herr_t read_projection
    (
     MPI_Comm                   comm,
     const std::string&         file_name,
     const std::string&         src_pop_name,
     const std::string&         dst_pop_name,
     const NODE_IDX_T&          dst_start,
     const NODE_IDX_T&          src_start,
     uint64_t&                  nedges,
     DST_BLK_PTR_T&             block_base,
     DST_PTR_T&                 edge_base,
     vector<DST_BLK_PTR_T>&     dst_blk_ptr,
     vector<NODE_IDX_T>&        dst_idx,
     vector<DST_PTR_T>&         dst_ptr,
     vector<NODE_IDX_T>&        src_idx,
     size_t                     offset,
     size_t                     numitems,
     bool collective
     )
    {
      herr_t ierr = 0;
      unsigned int rank, size;
      assert(MPI_Comm_size(comm, (int*)&size) >= 0);
      assert(MPI_Comm_rank(comm, (int*)&rank) >= 0);


      hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
      assert(fapl >= 0);
      assert(H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL) >= 0);

      hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, fapl);
      assert(file >= 0);

      // determine number of blocks in projection
      hsize_t num_blocks = hdf5::dataset_num_elements
        (file, hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_BLK_PTR)) - 1;

      
      // determine number of edges in projection
      nedges = hdf5::dataset_num_elements
        (file, hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::SRC_IDX));

      /* Create property list for collective dataset operations. */
      hid_t rapl = H5Pcreate (H5P_DATASET_XFER);
      if (collective)
        {
          ierr = H5Pset_dxpl_mpio (rapl, H5FD_MPIO_COLLECTIVE);
        }

      vector< pair<hsize_t,hsize_t> > bins;
      hsize_t read_blocks = 0;

      if (numitems > 0) 
        {
          if (offset < num_blocks)
            {
              read_blocks = min((hsize_t)numitems*size, num_blocks-offset);
            }
        }
      else
        {
          read_blocks = num_blocks;
        }

      if (read_blocks > 0)
        {
          // determine which blocks of block_ptr are read by which rank
          mpi::rank_ranges(read_blocks, size, bins);

          // determine start and stop block for the current rank
          hsize_t start, stop;
          
          start = bins[rank].first + offset;
          stop  = bins[rank].first + bins[rank].second;
          block_base = start;
          
          hsize_t block;
          if (stop > start)
            block = stop - start + 1;
          else
            block = 0;

          DEBUG("Task ",rank,": ","num_blocks = ", num_blocks, " read_blocks = ", read_blocks, 
                " start = ", start, " stop = ", stop, ", block = ", block, "\n");

          DST_BLK_PTR_T block_rebase = 0;

          // read destination block pointers

          // allocate buffer and memory dataspace
          dst_blk_ptr.resize(block, 0);
          
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
          
          DEBUG("Task ",rank,": ", "dst_blk_ptr.size() = ", dst_blk_ptr.size(), "\n");
          if (dst_blk_ptr.size() > 0)
            {
              DEBUG("Task ",rank,": ", "dst_blk_ptr.front() = ", dst_blk_ptr.front(), 
                    " dst_blk_ptr.back() = ", dst_blk_ptr.back(), "\n");
            }
      
          // rebase the block_ptr array to local offsets
          // REBASE is going to be the start offset for the hyperslab
      
          if (block > 0)
            {
              block_rebase = dst_blk_ptr[0];
              DEBUG("Task ",rank,": ","block_rebase = ", block_rebase, "\n");
          
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
          hsize_t dst_idx_block;
          
          if (dst_blk_ptr.size() > 0)
            dst_idx_block = block-1;
          else
            dst_idx_block = 0;
          dst_idx.resize(dst_idx_block, 0);
          
          DEBUG("Task ",rank,": ", "dst_idx: block = ", dst_idx_block, 
                " dst_idx: start = ", start, "\n");
          
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
          hsize_t dst_ptr_block, dst_ptr_start;
          if (dst_blk_ptr.size() > 0)
            {
              dst_ptr_start = (hsize_t)block_rebase;
              dst_ptr_block = (hsize_t)(dst_blk_ptr.back() - dst_blk_ptr.front());
              if ((num_blocks > 1) && (rank < size-1))
                {
                  dst_ptr_block ++;
                }
            }
          else
            {
              dst_ptr_start = 0;
              dst_ptr_block = 0;
            }
          dst_ptr.resize(dst_ptr_block, 0);
          
          DEBUG("Task ",rank,": ", "dst_ptr: start = ", dst_ptr_start, 
                " dst_ptr: block = ", dst_ptr_block, "\n");
          
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
          
          if (dst_ptr.size() > 0)
            {
              DEBUG("Task ",rank,": ", "dst_ptr.front() = ", dst_ptr.front(),
                    " dst_ptr.back() = ", dst_ptr.back(), "\n");
            }

          DST_PTR_T dst_rebase = 0;
          
          hsize_t src_idx_block=0, src_idx_start=0;
          if (dst_ptr_block > 0)
            {
              dst_rebase = dst_ptr[0];
              edge_base = dst_rebase;
              DEBUG("Task ",rank,": ", "dst_ptr: dst_rebase = ", dst_rebase, "\n");
              for (size_t i = 0; i < dst_ptr.size(); ++i)
                {
                  dst_ptr[i] -= dst_rebase;
                }
              
              // read source indices
              src_idx_start = dst_rebase;
              src_idx_block = (hsize_t)(dst_ptr.back() - dst_ptr.front());

              DEBUG("Task ",rank,": ", "src_idx: start = ", src_idx_start, " block = ", src_idx_block, "\n");

              // allocate buffer and memory dataspace
              src_idx.resize(src_idx_block, 0);
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
          assert(ierr >= 0);
              
          DEBUG("Task ",rank,": ", "src_idx: done\n");
          
          assert(H5Fclose(file) >= 0);
          assert(H5Pclose(fapl) >= 0);
          ierr = H5Pclose(rapl);
          assert(ierr == 0);
          
          DEBUG("Task ",rank,": ", "read_dbs_projection done\n");
        }
      return ierr;
    }

    
    herr_t read_projection_serial
    (
     const std::string&         file_name,
     const std::string&         src_pop_name,
     const std::string&         dst_pop_name,
     const NODE_IDX_T&          dst_start,
     const NODE_IDX_T&          src_start,
     uint64_t&                  nedges,
     DST_BLK_PTR_T&             block_base,
     DST_PTR_T&                 edge_base,
     vector<DST_BLK_PTR_T>&     dst_blk_ptr,
     vector<NODE_IDX_T>&        dst_idx,
     vector<DST_PTR_T>&         dst_ptr,
     vector<NODE_IDX_T>&        src_idx,
     size_t                     offset,
     size_t                     numitems
     )
    {
        herr_t ierr = 0;

        hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
        assert(fapl >= 0);

        hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, fapl);
        assert(file >= 0);

        // determine number of blocks in projection
        uint64_t num_blocks = hdf5::dataset_num_elements
          (file, hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_BLK_PTR)) - 1;

        // determine number of edges in projection
        nedges = hdf5::dataset_num_elements
          (file, hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::SRC_IDX));

        hsize_t read_blocks = 0;

        if (numitems > 0) 
          {
            if (offset < num_blocks)
              {
                read_blocks = num_blocks-offset;
              }
          }
        else
          {
            read_blocks = num_blocks;
          }

      
        DST_BLK_PTR_T block_rebase = 0;
        hsize_t block = read_blocks+1;

        // read destination block pointers

        if (read_blocks > 0)
          {
            // allocate buffer and memory dataspace
            dst_blk_ptr.resize(block, 0);

            ierr = hdf5::read_serial<DST_BLK_PTR_T>
              (
               file,
               hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_BLK_PTR),
               block,
               DST_BLK_PTR_H5_NATIVE_T,
               dst_blk_ptr,
               H5P_DEFAULT
               );
            assert(ierr >= 0);

            // rebase the block_ptr array to local offsets
            // REBASE is going to be the start offset for the hyperslab

            block_rebase = dst_blk_ptr[0];

            for (size_t i = 0; i < dst_blk_ptr.size(); ++i)
              {
                dst_blk_ptr[i] -= block_rebase;
              }
          }

        // read destination block indices

        if (block > 0)
          {
            hsize_t dst_idx_block = block-1;
            dst_idx.resize(dst_idx_block, 0);

            assert(dst_idx.size() > 0);

            ierr = hdf5::read_serial<NODE_IDX_T>
              (
               file,
               hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_BLK_IDX),
               dst_idx_block,
               NODE_IDX_H5_NATIVE_T,
               dst_idx,
               H5P_DEFAULT
               );
            assert(ierr >= 0);
          }

        DST_PTR_T dst_rebase = 0;

        // read destination pointers

        if (block > 0)
          {
            hsize_t dst_ptr_block = (hsize_t)(dst_blk_ptr.back() - dst_blk_ptr.front());
            dst_ptr.resize(dst_ptr_block, 0);
            assert(dst_ptr.size() > 0);

            ierr = hdf5::read_serial<DST_PTR_T>
              (
               file,
               hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_PTR),
               dst_ptr_block,
               DST_PTR_H5_NATIVE_T,
               dst_ptr,
               H5P_DEFAULT
               );
            assert(ierr >= 0);

            dst_rebase = dst_ptr[0];
            edge_base = dst_rebase;
            for (size_t i = 0; i < dst_ptr.size(); ++i)
              {
                dst_ptr[i] -= dst_rebase;
              }
          }

        if (block > 0)
          {
            hsize_t src_idx_block = (hsize_t)(dst_ptr.back() - dst_ptr.front());

            if (src_idx_block > 0)
              {
                // allocate buffer and memory dataspace
                src_idx.resize(src_idx_block, 0);
                assert(src_idx.size() > 0);

                ierr = hdf5::read_serial<NODE_IDX_T>
                  (
                   file,
                   hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::SRC_IDX),
                   src_idx_block,
                   NODE_IDX_H5_NATIVE_T,
                   src_idx,
                   H5P_DEFAULT
                   );
                assert(ierr >= 0);
              }

          }

        assert(H5Fclose(file) >= 0);
        assert(H5Pclose(fapl) >= 0);
        assert(ierr == 0);

        return ierr;
      }

    }
  }

