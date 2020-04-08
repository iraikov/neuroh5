// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_syn_projection.cc
///
///  Functions for reading edge information in cell attribute format
///
///  Copyright (C) 2016-2018 Project NeuroH5.
//==============================================================================

#include "debug.hh"
#include <string>

#include "neuroh5_types.hh"
#include "dataset_num_elements.hh"
#include "read_template.hh"
#include "path_names.hh"
#include "read_syn_projection.hh"
#include "throw_assert.hh"

using namespace std;

namespace neuroh5
{
  namespace io
  {
    // Calculate the starting and end block for each rank
    void compute_bins
    (
     const size_t&                    num_blocks,
     const size_t&                    size,
     vector< pair<hsize_t,hsize_t> >& bins
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
     * Read the basic synapse connectivity structure
     *************************************************************************/

      
    // .../source_gid 
    // .../source_gid/gid 
    // .../source_gid/ptr 
    // .../source_gid/value 
    // .../syn_id
    // .../syn_id/gid 
    // .../syn_id/ptr 
    // .../syn_id/value 

      
    herr_t read_syn_projection
    (
     MPI_Comm              comm,
     const string&         file_name,
     const string&         prefix,
     vector<NODE_IDX_T>&   dst_gid,
     vector<DST_PTR_T>&    src_gid_ptr,
     vector<NODE_IDX_T>&   src_gid,
     vector<DST_PTR_T>&    syn_id_ptr,
     vector<NODE_IDX_T>&   syn_id
     )
    {
      herr_t ierr = 0;
      unsigned int rank, size;
      throw_assert_nomsg(MPI_Comm_size(comm, (int*)&size) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Comm_rank(comm, (int*)&rank) == MPI_SUCCESS);

      hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
      throw_assert_nomsg(fapl >= 0);
      throw_assert_nomsg(H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL) >= 0);

      hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, fapl);
      throw_assert_nomsg(file >= 0);

      // determine number of edges in projection
      uint64_t num_blocks = hdf5::dataset_num_elements
        (file, prefix+"/source_gid/gid");

      vector< pair<hsize_t,hsize_t> > bins;

      // determine which blocks of block_ptr are read by which rank
      bins.resize(size);
      compute_bins(num_blocks, size, bins);

      // determine start and stop block for the current rank
      hsize_t start = bins[rank].first;
      hsize_t stop  = bins[rank].first + bins[rank].second + 1;

      hsize_t block = stop - start;
        
      DEBUG("Task ",rank,": ","num_blocks = ", num_blocks, "\n");
      DEBUG("Task ",rank,": ","start = ", start, " stop = ", stop, "\n");
      DEBUG("Task ",rank,": ","block = ", block, "\n");

      /* Create property list for collective dataset operations. */
      hid_t rapl = H5Pcreate (H5P_DATASET_XFER);
      ierr = H5Pset_dxpl_mpio (rapl, H5FD_MPIO_COLLECTIVE);

      // read source gid pointers

      // allocate buffer and memory dataspace
      src_gid_ptr.resize(block);
        
      ierr = hdf5::read<DST_PTR_T>
        (
         file,
         prefix+"/source_gid/ptr",
         start,
         block,
         DST_PTR_H5_NATIVE_T,
         src_gid_ptr,
         rapl
         );
      throw_assert_nomsg(ierr >= 0);
        
      // rebase the src_gid_ptr array to local offsets
      // REBASE is going to be the start offset for the hyperslab

      DST_PTR_T src_gid_rebase = src_gid_ptr[0];
        
      for (size_t i = 0; i < src_gid_ptr.size(); ++i)
        {
          src_gid_ptr[i] -= src_gid_rebase;
        }

      hsize_t dst_gid_block;
      if (rank == size-1)
        {
          dst_gid_block = block-1;
        } else
        {
          dst_gid_block = block;
        }

      // read target gids
      // allocate buffer and memory dataspace
      dst_gid.resize(dst_gid_block);
        
      ierr = hdf5::read<NODE_IDX_T>
        (
         file,
         prefix+"/source_gid/gid",
         start,
         dst_gid_block,
         NODE_IDX_H5_NATIVE_T,
         dst_gid,
         rapl
         );
        
      throw_assert_nomsg(ierr >= 0);

      // read source indices
      hsize_t src_gid_block = (hsize_t)(src_gid_ptr.back() - src_gid_ptr.front());
      hsize_t src_gid_start = (hsize_t)src_gid_rebase;

      // allocate buffer and memory dataspace
      src_gid.resize(src_gid_block);
      throw_assert_nomsg(src_gid.size() > 0);

      ierr = hdf5::read<NODE_IDX_T>
        (
         file,
         prefix+"/source_gid/value",
         src_gid_start,
         src_gid_block,
         NODE_IDX_H5_NATIVE_T,
         src_gid,
         rapl
         );
      throw_assert_nomsg(ierr >= 0);

      // Read syn_id_pointers
        
      // allocate buffer and memory dataspace
      syn_id_ptr.resize(block);
        
      ierr = hdf5::read<DST_PTR_T>
        (
         file,
         prefix+"/syn_id/ptr",
         start,
         block,
         DST_PTR_H5_NATIVE_T,
         syn_id_ptr,
         rapl
         );
      throw_assert_nomsg(ierr >= 0);

      DST_PTR_T syn_id_rebase = syn_id_ptr[0];

      for (size_t i = 0; i < syn_id_ptr.size(); ++i)
        {
          syn_id_ptr[i] -= syn_id_rebase;
        }

      // read synapse indices
      hsize_t syn_id_block = (hsize_t)(syn_id_ptr.back() - syn_id_ptr.front());
      hsize_t syn_id_start = (hsize_t)syn_id_rebase;

      // allocate buffer and memory dataspace
      syn_id.resize(syn_id_block);
      throw_assert_nomsg(syn_id.size() > 0);

      ierr = hdf5::read<NODE_IDX_T>
        (
         file,
         prefix+"/syn_id/value",
         syn_id_start,
         syn_id_block,
         NODE_IDX_H5_NATIVE_T,
         syn_id,
         rapl
         );
      throw_assert_nomsg(ierr >= 0);

        
      throw_assert_nomsg(H5Fclose(file) >= 0);

      ierr = H5Pclose(rapl);
      throw_assert_nomsg(ierr == 0);

      DEBUG("Task ",rank,": ", "read_syn_projection done\n");
      return ierr;
    }
  }
}
