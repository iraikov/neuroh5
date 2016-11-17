// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file dbs_edge_reader.cc
///
///  Functions for reading edge information in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016 Project Neurograph.
//==============================================================================

#include "debug.hh"

#include "dbs_edge_reader.hh"
#include "dbs_read_template.hh"
#include "hdf5_path_names.hh"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <cstring>

#undef NDEBUG
#include <cassert>

#define MAX_PRJ_NAME 1024
#define MAX_EDGE_ATTR_NAME 1024

using namespace std;
using namespace ngh5::io::hdf5;

namespace ngh5
{
  namespace io
  {
    namespace hdf5
    {
      // Calculate the starting and end block for each rank
      void compute_bins
      (
       size_t                           num_blocks,
       size_t                           size,
       vector< pair<hsize_t,hsize_t> > &bins
       )
      {
        hsize_t remainder=0, offset=0, buckets=0;

        for (size_t i=0; i<size; i++)
          {
            remainder = num_blocks - offset;
            buckets   = (size - i);
            bins[i]   = make_pair(offset, remainder / buckets);
            offset    += bins[i].second;
          }
      }

      /**************************************************************************
       * Read the basic DBS graph structure
       *************************************************************************/

      herr_t read_dbs_projection
      (
       MPI_Comm                   comm,
       const std::string&         file_name,
       const std::string&         proj_name,
       const vector<pop_range_t>& pop_vector,
       NODE_IDX_T&                dst_start,
       NODE_IDX_T&                src_start,
       uint64_t&                  nedges,
       DST_BLK_PTR_T&             block_base,
       DST_PTR_T&                 edge_base,
       vector<DST_BLK_PTR_T>&     dst_blk_ptr,
       vector<NODE_IDX_T>&        dst_idx,
       vector<DST_PTR_T>&         dst_ptr,
       vector<NODE_IDX_T>&        src_idx
       )
      {
        hid_t fapl, file;
        herr_t ierr = 0;
        unsigned int rank, size;

        assert(MPI_Comm_size(comm, (int*)&size) >= 0);
        assert(MPI_Comm_rank(comm, (int*)&rank) >= 0);

        /************************************************************************
         * MPI rank 0 reads and broadcasts the number of nodes
         ***********************************************************************/

        uint64_t num_blocks;

        // process 0 reads the size of dst_blk_ptr and the source and target
        // populations
        if (rank == 0)
          {
            uint32_t dst_pop, src_pop;
            hid_t file, fspace, mspace, dset;
            hsize_t one = 1;

            // determine number of blocks in projection
            file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            assert(file >= 0);
            dset = H5Dopen2(file, projection_path_join(proj_name,
                                                       DST_BLK_PTR).c_str(),
                            H5P_DEFAULT);
            assert(dset >= 0);
            fspace = H5Dget_space(dset);
            assert(fspace >= 0);
            num_blocks = (uint64_t) H5Sget_simple_extent_npoints(fspace) - 1;
            assert(num_blocks > 0);
            assert(H5Sclose(fspace) >= 0);
            assert(H5Dclose(dset) >= 0);

            // determine number of edges in projection
            dset = H5Dopen2(file,
                            projection_path_join(proj_name, SRC_IDX).c_str(),
                            H5P_DEFAULT);
            assert(dset >= 0);
            fspace = H5Dget_space(dset);
            assert(fspace >= 0);
            nedges = (uint64_t) H5Sget_simple_extent_npoints(fspace);
            assert(nedges > 0);
            assert(H5Sclose(fspace) >= 0);
            assert(H5Dclose(dset) >= 0);

            // determine source and destination populations
            mspace = H5Screate_simple(1, &one, NULL);
            assert(mspace >= 0);
            ierr = H5Sselect_all(mspace);
            assert(ierr >= 0);

            dset = H5Dopen2(file,
                            projection_path_join(proj_name, DST_POP).c_str(),
                            H5P_DEFAULT);
            assert(dset >= 0);
            fspace = H5Dget_space(dset);
            assert(fspace >= 0);

            ierr = H5Dread(dset, NODE_IDX_H5_NATIVE_T, mspace, fspace,
                           H5P_DEFAULT, &dst_pop);
            assert(ierr >= 0);

            assert(H5Sclose(fspace) >= 0);
            assert(H5Dclose(dset) >= 0);
            assert(H5Sclose(mspace) >= 0);

            mspace = H5Screate_simple(1, &one, NULL);
            assert(mspace >= 0);
            ierr = H5Sselect_all(mspace);
            assert(ierr >= 0);

            dset = H5Dopen2(file,
                            projection_path_join(proj_name, SRC_POP).c_str(),
                            H5P_DEFAULT);
            assert(dset >= 0);
            fspace = H5Dget_space(dset);
            assert(fspace >= 0);

            ierr = H5Dread(dset, NODE_IDX_H5_NATIVE_T, mspace, fspace,
                           H5P_DEFAULT, &src_pop);
            assert(ierr >= 0);

            assert(H5Sclose(fspace) >= 0);
            assert(H5Dclose(dset) >= 0);
            assert(H5Sclose(mspace) >= 0);
            assert(H5Fclose(file) >= 0);

            dst_start = pop_vector[dst_pop].start;
            src_start = pop_vector[src_pop].start;

            DEBUG("num_blocks = ", num_blocks,
                  " dst_start = ", dst_start,
                  " src_start = ", src_start,
                  "\n");
          }

        assert(MPI_Bcast(&nedges, 1, MPI_UINT64_T, 0, comm) >= 0);
        assert(MPI_Bcast(&num_blocks, 1, MPI_UINT64_T, 0, comm) >= 0);
        assert(MPI_Bcast(&dst_start, 1, MPI_UINT32_T, 0, comm) >= 0);
        assert(MPI_Bcast(&src_start, 1, MPI_UINT32_T, 0, comm) >= 0);

        /************************************************************************
         * read the connectivity in DBS format
         ***********************************************************************/

        hsize_t start, stop, block;
        vector< pair<hsize_t,hsize_t> > bins;

        // determine which blocks of block_ptr are read by which rank
        bins.resize(size);
        compute_bins(num_blocks, size, bins);

        // determine start and stop block for the current rank
        start = bins[rank].first;
        stop  = bins[rank].first + bins[rank].second + 1;
        block_base = start;

        block = stop - start;

        DEBUG("Task ",rank,": ","num_blocks = ", num_blocks, "\n");
        DEBUG("Task ",rank,": ","start = ", start, " stop = ", stop, "\n");
        DEBUG("Task ",rank,": ","block = ", block, "\n");

        fapl = H5Pcreate(H5P_FILE_ACCESS);
        assert(fapl >= 0);
        assert(H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL) >= 0);

        file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, fapl);
        assert(file >= 0);

        DST_BLK_PTR_T block_rebase = 0;

        // read destination block pointers

        if (block > 0)
          {
            // allocate buffer and memory dataspace
            dst_blk_ptr.resize(block);

            ierr = dbs_read<DST_BLK_PTR_T>
              (
               file,
               projection_path_join(proj_name, DST_BLK_PTR),
               start,
               block,
               DST_BLK_PTR_H5_NATIVE_T,
               dst_blk_ptr
               );
            assert(ierr >= 0);

            // rebase the block_ptr array to local offsets
            // REBASE is going to be the start offset for the hyperslab

            block_rebase = dst_blk_ptr[0];
            DEBUG("Task ",rank,": ","block_rebase = ", block_rebase, "\n");

            for (size_t i = 0; i < dst_blk_ptr.size(); ++i)
              {
                dst_blk_ptr[i] -= block_rebase;
              }
          }

        // read destination block indices

        if (block > 0)
          {
            if (rank == size-1)
              {
                block = block-1;
              }

            dst_idx.resize(block);
            assert(dst_idx.size() > 0);

            DEBUG("Task ",rank,": ", "dst_idx: block = ", block, "\n");
            DEBUG("Task ",rank,": ", "dst_idx: start = ", start, "\n");

            ierr = dbs_read<NODE_IDX_T>
              (
               file,
               projection_path_join(proj_name, DST_BLK_IDX),
               start,
               block,
               NODE_IDX_H5_NATIVE_T,
               dst_idx
               );
            assert(ierr >= 0);
          }

        DST_PTR_T dst_rebase = 0;

        // read destination pointers

        if (block > 0)
          {
            DEBUG("Task ",rank,": ", "dst_ptr: dst_blk_ptr.front() = ",
                  dst_blk_ptr.front(), "\n");
            DEBUG("Task ",rank,": ", "dst_ptr: dst_blk_ptr.back() = ",
                  dst_blk_ptr.back(), "\n");

            if (rank < size-1)
              {
                block = (hsize_t)(dst_blk_ptr.back() - dst_blk_ptr.front() + 1);
              }
            else
              {
                block = (hsize_t)(dst_blk_ptr.back() - dst_blk_ptr.front());
              }

            hsize_t start = (hsize_t)block_rebase;
            dst_ptr.resize(block);
            assert(dst_ptr.size() > 0);

            DEBUG("Task ",rank,": ", "dst_ptr: start = ", start, "\n");
            DEBUG("Task ",rank,": ", "dst_ptr: block = ", block, "\n");

            ierr = dbs_read<DST_PTR_T>
              (
               file,
               projection_path_join(proj_name, DST_PTR),
               start,
               block,
               DST_PTR_H5_NATIVE_T,
               dst_ptr
               );
            assert(ierr >= 0);

            dst_rebase = dst_ptr[0];
            edge_base = dst_rebase;
            DEBUG("Task ",rank,": ", "dst_ptr: dst_rebase = ", dst_rebase, "\n");
            for (size_t i = 0; i < dst_ptr.size(); ++i)
              {
                dst_ptr[i] -= dst_rebase;
              }
          }

        // read source indices

        if (block > 0)
          {
            DEBUG("Task ",rank,": ", "src_idx: dst_ptr.front() = ",
                  dst_ptr.front(), "\n");
            DEBUG("Task ",rank,": ", "src_idx: dst_ptr.back() = ",
                  dst_ptr.back(), "\n");

            hsize_t block = (hsize_t)(dst_ptr.back() - dst_ptr.front());
            hsize_t start = (hsize_t)dst_rebase;

            DEBUG("Task ",rank,": ", "src_idx: block = ", block, "\n");
            DEBUG("Task ",rank,": ", "src_idx: start = ", start, "\n");

            if (block > 0)
              {
                // allocate buffer and memory dataspace
                src_idx.resize(block);
                assert(src_idx.size() > 0);

                ierr = dbs_read<NODE_IDX_T>
                  (
                   file,
                   projection_path_join(proj_name, SRC_IDX),
                   start,
                   block,
                   NODE_IDX_H5_NATIVE_T,
                   src_idx
                   );
                assert(ierr >= 0);
              }

            DEBUG("Task ",rank,": ", "src_idx: done\n");
          }

        assert(H5Fclose(file) >= 0);

        DEBUG("Task ",rank,": ", "read_dbs_projection done\n");
        return ierr;
      }
    }
  }
}
