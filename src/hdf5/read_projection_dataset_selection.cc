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

#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>

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
     vector< pair<hsize_t,hsize_t> >& src_idx_ranges,
     vector<NODE_IDX_T>&        src_idx,
     size_t&                    total_num_edges,
     bool collective
     )
    {
      herr_t ierr = 0;
      unsigned int rank, size;
      throw_assert_nomsg(MPI_Comm_size(comm, (int*)&size) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Comm_rank(comm, (int*)&rank) == MPI_SUCCESS);

      size_t num_blocks=0;
      
      if (rank == 0)
        {
          hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
          throw_assert_nomsg(file >= 0);

          // determine number of blocks in projection
          num_blocks = hdf5::dataset_num_elements
            (file, hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_BLK_PTR));
          if (num_blocks > 0)
            num_blocks--;
          
          // determine number of edges in projection
          total_num_edges = hdf5::dataset_num_elements
            (file, hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::SRC_IDX));

          throw_assert_nomsg(H5Fclose(file) >= 0);
        }
      
      throw_assert_nomsg(MPI_Bcast(&num_blocks, 1, MPI_SIZE_T, 0, comm) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Bcast(&total_num_edges, 1, MPI_SIZE_T, 0, comm) == MPI_SUCCESS);

      hsize_t read_blocks = num_blocks;
      
      if (read_blocks > 0)
        {
          DST_BLK_PTR_T block_rebase = 0;
          vector<DST_BLK_PTR_T> dst_blk_ptr(read_blocks+1, 0);

          mpi::MPI_DEBUG(comm, "read_projection_dataset_selection: reading destination block pointer for: ", 
                         src_pop_name, " -> ", dst_pop_name, ": ", read_blocks, " blocks");

          // read destination block pointers
          if (rank == 0)
            {
              hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
              throw_assert_nomsg(file >= 0);
          
              ierr = hdf5::read<DST_BLK_PTR_T>
                (
                 file,
                 hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_BLK_PTR),
                 0,
                 read_blocks+1,
                 DST_BLK_PTR_H5_NATIVE_T,
                 dst_blk_ptr,
                 H5P_DEFAULT
                 );
              throw_assert_nomsg(ierr >= 0);
          
              // rebase the block_ptr array to local offsets
              // REBASE is going to be the start offset for the hyperslab
              block_rebase = dst_blk_ptr[0];
          
              for (size_t i = 0; i < dst_blk_ptr.size(); ++i)
                {
                  dst_blk_ptr[i] -= block_rebase;
                }

              throw_assert_nomsg(H5Fclose(file) >= 0);
            }

          throw_assert_nomsg(MPI_Bcast(&dst_blk_ptr[0], dst_blk_ptr.size(), MPI_ATTR_PTR_T, 0, comm) == MPI_SUCCESS);


          // read destination block indices
          hsize_t dst_blk_idx_block=0;
          vector<NODE_IDX_T> dst_blk_idx;
          
          if (dst_blk_ptr.size() > 0)
            dst_blk_idx_block = read_blocks;
          else
            dst_blk_idx_block = 0;
          if (dst_blk_idx_block > 0)
            {
              dst_blk_idx.resize(dst_blk_idx_block, 0);
            }

          mpi::MPI_DEBUG(comm, "read_projection_dataset_selection: reading destination block index for: ", 
                         src_pop_name, " -> ", dst_pop_name, ": ", dst_blk_idx_block, " blocks");
          
          if (rank == 0)
            {
              hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
              throw_assert_nomsg(file >= 0);
              
              ierr = hdf5::read<NODE_IDX_T>
                (
                 file,
                 hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_BLK_IDX),
                 0,
                 dst_blk_idx_block,
                 NODE_IDX_H5_NATIVE_T,
                 dst_blk_idx,
                 H5P_DEFAULT
                 );
              throw_assert_nomsg(ierr >= 0);
              throw_assert_nomsg(H5Fclose(file) >= 0);
            }

          throw_assert_nomsg(MPI_Bcast(&dst_blk_idx[0], dst_blk_idx_block, NODE_IDX_MPI_T, 0, comm) == MPI_SUCCESS);
          
          // read destination pointers
          hsize_t dst_ptr_block=0, dst_ptr_start=0;
          if (dst_blk_ptr.size() > 0)
            {
              dst_ptr_start = (hsize_t)block_rebase;
              dst_ptr_block = (hsize_t)(dst_blk_ptr.back() - dst_blk_ptr.front());
              if (dst_ptr_block < dst_blk_ptr.back())
                {
                  dst_ptr_block ++;
                }
            }
          
          vector<DST_PTR_T> dst_ptr;
          if (dst_ptr_block > 0)
            {
              dst_ptr.resize(dst_ptr_block, 0);
            }

          mpi::MPI_DEBUG(comm, "read_projection_dataset_selection: reading destination pointer for: ", 
                         src_pop_name, " -> ", dst_pop_name, ": ", dst_ptr_block, " elements");
          if (rank == 0)
            {
              hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
              throw_assert_nomsg(file >= 0);

              ierr = hdf5::read<DST_PTR_T>
                (
                 file,
                 hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_PTR),
                 dst_ptr_start,
                 dst_ptr_block,
                 DST_PTR_H5_NATIVE_T,
                 dst_ptr,
                 H5P_DEFAULT
                 );
              throw_assert_nomsg(ierr >= 0);

              throw_assert_nomsg(H5Fclose(file) >= 0);
            }

          throw_assert_nomsg(MPI_Bcast(&dst_ptr[0], dst_ptr_block, MPI_ATTR_PTR_T, 0, comm) == MPI_SUCCESS);
          
          DST_PTR_T dst_rebase = 0;
          // Create source index ranges based on selection_dst_ptr
          src_idx_ranges.clear();
          ATTR_PTR_T selection_dst_ptr_pos = 0;
          if (dst_ptr_block > 0)
            {
              dst_rebase = dst_ptr[0];
              edge_base = dst_rebase;
              for (size_t i = 0; i < dst_ptr.size(); ++i)
                {
                  dst_ptr[i] -= dst_rebase;
                }

              vector<NODE_IDX_T> dst_idx;
              // Create node index in order to determine which edges to read
              for (size_t i=0; i < dst_blk_idx.size(); i++)
                {
                  NODE_IDX_T dst_base = dst_blk_idx[i];
                  DST_BLK_PTR_T sz = dst_blk_ptr[i+1] - dst_blk_ptr[i];
                  if (i == (dst_blk_idx.size()-1))
                    {
                      sz--;
                    }
                  for (size_t j=0; j<sz; j++)
                    {
                      dst_idx.push_back(dst_base + j);
                    }
                }

              mpi::MPI_DEBUG(comm, "read_projection_dataset_selection: creating selection ranges for: ", 
                             selection.size(), " selection indices");
              
              for (const NODE_IDX_T& s : selection) 
                {
                  if (s >= dst_start)
                    {
                      NODE_IDX_T n = s-dst_start; 
                      auto it = std::find(dst_idx.begin(), dst_idx.end(), n);
                      
                      if (it != dst_idx.end())
                        {
                          selection_dst_idx.push_back(s);
                          
                          ptrdiff_t pos = it - dst_idx.begin();

                          hsize_t src_idx_start=dst_ptr[pos];
                          hsize_t src_idx_block=dst_ptr[pos+1]-src_idx_start;

                          src_idx_ranges.push_back(make_pair(src_idx_start+dst_rebase, src_idx_block));
                          selection_dst_ptr.push_back(selection_dst_ptr_pos);
                          selection_dst_ptr_pos += src_idx_block;
                        }
                      else
                        {
                          throw runtime_error(string("read_projection_dataset_selection: destination index ")+
                                              std::to_string(s)+
                                              string(" not found in destination index dataset ")+
                                              hdf5::edge_attribute_path(src_pop_name, dst_pop_name, 
                                                                        hdf5::EDGES, hdf5::DST_PTR));
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
        
          mpi::MPI_DEBUG(comm, "read_projection_dataset_selection: reading source indices for: ", 
                         src_pop_name, " -> ", dst_pop_name, ": ", src_idx.size(), " elements");
          
          {
            
            /* Create property list for parallel file access. */
            hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
            throw_assert_nomsg(fapl >= 0);
            throw_assert_nomsg(H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL) >= 0);
            
            /* Create property list for collective dataset operations. */
            hid_t rapl = H5Pcreate (H5P_DATASET_XFER);
            if (collective)
              {
                ierr = H5Pset_dxpl_mpio (rapl, H5FD_MPIO_COLLECTIVE);
              }
            
            hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, fapl);
            throw_assert_nomsg(file >= 0);
            
            ierr = hdf5::read_selection<NODE_IDX_T>
              (
               file,
               hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::SRC_IDX),
               NODE_IDX_H5_NATIVE_T,
               src_idx_ranges,
               src_idx,
               rapl
               );
            throw_assert_nomsg(ierr >= 0);
            
            throw_assert_nomsg(H5Fclose(file) >= 0);
            throw_assert_nomsg(H5Pclose(fapl) >= 0);
            throw_assert_nomsg(H5Pclose(rapl) >= 0);

          }
          
        }

      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "error in MPI_Barrier");
      

      return ierr;
    }

    
  }
}

