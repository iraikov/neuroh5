// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_projection_datasets.cc
///
///  Functions for reading edge information in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2024 Project NeuroH5.
//==============================================================================

#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>  // For setw

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
    // Structure to hold assignments for each rank
    // Structure of arrays to hold assignments for all ranks
    
    struct RankAssignments
    {
      // Range of destination blocks assigned to each rank
      vector<hsize_t> dst_block_start;
      vector<hsize_t> dst_block_count;
      // Range of source indices each rank will read
      vector<hsize_t> src_idx_start;
      vector<hsize_t> src_idx_count;
      // Destination pointers relevant to each rank
      vector<hsize_t> dst_ptr_start;
      vector<hsize_t> dst_ptr_count;
      
      // For each rank, the relevant destination indices
      vector<vector<NODE_IDX_T>> local_dst_indices;

      // The last rank that has non-zero assignment
      rank_t last_rank;
      
      // Constructor to initialize with proper size
      RankAssignments(int size) : 
        dst_block_start(size, 0),
        dst_block_count(size, 0),
        src_idx_start(size, 0),
        src_idx_count(size, 0),
        dst_ptr_start(size, 0),
        dst_ptr_count(size, 0),
        last_rank(0),
        local_dst_indices(size) {}
    };


    // Pretty print method for RankAssignments
    void print_rank_assignments(
                                MPI_Comm comm,
                                const RankAssignments& assignments,
                                std::ostream& os = std::cout
                                )
    {
      unsigned int rank, size;
      MPI_Comm_size(comm, (int*)&size);
      MPI_Comm_rank(comm, (int*)&rank);
      
      // Only rank 0 prints
      if (rank == 0) 
        {
          os << "\n";
          os << "==========================================================\n";
          os << "                   RANK ASSIGNMENT TABLE                  \n";
          os << "==========================================================\n";
          
          // Table header
          os << std::left << std::setw(6) << "Rank" 
             << std::setw(14) << "Dst Blk Start" 
             << std::setw(14) << "Dst Blk Count" 
             << std::setw(14) << "Src Idx Start" 
             << std::setw(14) << "Src Idx Count" 
             << std::setw(14) << "Dst Ptr Start" 
             << std::setw(14) << "Dst Ptr Count" 
             << "Dst Indices\n";
          os << std::string(100, '-') << "\n";
          
          // Print information for each rank
          for (unsigned int r = 0; r < size; r++)
            {
              os << std::left << std::setw(6) << r
                 << std::setw(14) << assignments.dst_block_start[r]
                 << std::setw(14) << assignments.dst_block_count[r]
                 << std::setw(14) << assignments.src_idx_start[r]
                 << std::setw(14) << assignments.src_idx_count[r]
                 << std::setw(14) << assignments.dst_ptr_start[r]
                 << std::setw(14) << assignments.dst_ptr_count[r];
              
              // Print destination indices (limit to first 5 if there are many)
              os << "[";
              size_t num_indices = assignments.local_dst_indices[r].size();
              size_t display_count = std::min(num_indices, static_cast<size_t>(5));
              
              for (size_t i = 0; i < display_count; i++)
                {
                  if (i > 0) os << ", ";
                  os << assignments.local_dst_indices[r][i];
                }
              
              if (num_indices > display_count)
                {
                  os << ", ... (" << (num_indices - display_count) << " more)";
                }
              os << "]\n";
            }
          
          // Print summary statistics
          hsize_t total_blocks = 0;
          hsize_t total_edges = 0;
          
          for (unsigned int r = 0; r < size; r++)
            {
              total_blocks += assignments.dst_block_count[r];
              total_edges += assignments.src_idx_count[r];
            }
          
          os << std::string(100, '-') << "\n";
          os << "Total Blocks: " << total_blocks << "\n";
          os << "Total Edges: " << total_edges << "\n";
          os << "==========================================================\n\n";
        }
    
      // Make sure all ranks wait until printing is done
      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "error in MPI_Barrier");
    }
    
    herr_t read_projection_ptr
    (
     const std::string&         file_name,
     const std::string&         src_pop_name,
     const std::string&         dst_pop_name,
     vector<DST_BLK_PTR_T>&     dst_blk_ptr,
     vector<NODE_IDX_T>&        dst_idx,
     vector<DST_PTR_T>&         dst_ptr,
     size_t&                    total_num_edges,
     hsize_t&                   total_read_blocks
     )
    {
      
      herr_t ierr = 0;
    
      // Open file
      hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
      hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, fapl);
      
      // Get dataset sizes
      total_read_blocks = hdf5::dataset_num_elements
        (file, hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_BLK_PTR));
    
      total_num_edges = hdf5::dataset_num_elements
        (file, hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::SRC_IDX));
      
      // Read dst_blk_ptr
      dst_blk_ptr.resize(total_read_blocks, 0);
      ierr = hdf5::read<DST_BLK_PTR_T>
        (
         file,
         hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_BLK_PTR),
         0,
         total_read_blocks,
         DST_BLK_PTR_H5_NATIVE_T,
         dst_blk_ptr,
         H5P_DEFAULT
         );
    
      // Read dst_idx
      dst_idx.resize(total_read_blocks-1, 0);
      ierr = hdf5::read<NODE_IDX_T>
        (
         file,
         hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_BLK_IDX),
         0,
         total_read_blocks-1,
         NODE_IDX_H5_NATIVE_T,
         dst_idx,
         H5P_DEFAULT
         );
    
      // Read dst_ptr
      size_t total_ptr = dst_blk_ptr[total_read_blocks-1] - dst_blk_ptr[0];
      
      dst_ptr.resize(total_ptr, 0);
      ierr = hdf5::read<DST_PTR_T>
        (
         file,
         hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_PTR),
         0,
         total_ptr,
         DST_PTR_H5_NATIVE_T,
         dst_ptr,
         H5P_DEFAULT
         );
      
      H5Fclose(file);
      H5Pclose(fapl);
      
      return ierr;
    }

    // Function to distribute assignments to all ranks
    void distribute_assignments
    (
     MPI_Comm         comm,
     RankAssignments &rank_assignments
     )
    {
      unsigned int rank, size;
      MPI_Comm_size(comm, (int*)&size);
      MPI_Comm_rank(comm, (int*)&rank);

      MPI_Request bcast_req[6];
      int bcast_req_count = 0;

      // Broadcast simple members
      throw_assert(MPI_Ibcast(rank_assignments.dst_block_start.data(), size, MPI_UNSIGNED_LONG_LONG, 0, comm, &bcast_req[bcast_req_count++]) == MPI_SUCCESS,
                   "error in MPI_Ibcast");
      throw_assert(MPI_Ibcast(rank_assignments.dst_block_count.data(), size, MPI_UNSIGNED_LONG_LONG, 0, comm, &bcast_req[bcast_req_count++]) == MPI_SUCCESS,
                   "error in MPI_Ibcast");
      throw_assert(MPI_Ibcast(rank_assignments.src_idx_start.data(), size, MPI_UNSIGNED_LONG_LONG, 0, comm, &bcast_req[bcast_req_count++]) == MPI_SUCCESS,
                   "error in MPI_Ibcast");
      throw_assert(MPI_Ibcast(rank_assignments.src_idx_count.data(), size, MPI_UNSIGNED_LONG_LONG, 0, comm, &bcast_req[bcast_req_count++]) == MPI_SUCCESS,
                   "error in MPI_Ibcast");
      throw_assert(MPI_Ibcast(rank_assignments.dst_ptr_start.data(), size, MPI_UNSIGNED_LONG_LONG, 0, comm, &bcast_req[bcast_req_count++]) == MPI_SUCCESS,
                   "error in MPI_Ibcast");
      throw_assert(MPI_Ibcast(rank_assignments.dst_ptr_count.data(), size, MPI_UNSIGNED_LONG_LONG, 0, comm, &bcast_req[bcast_req_count++]) == MPI_SUCCESS,
                   "error in MPI_Ibcast");

      // Wait for broadcasts to complete
      throw_assert(MPI_Waitall(bcast_req_count, bcast_req, MPI_STATUSES_IGNORE) == MPI_SUCCESS,
                   "error in MPI_Waitall");

      // For the destination indices, we'll create a flattened array and index structure
      vector<int> indices_counts(size, 0);
      vector<int> indices_displs(size, 0);
      vector<NODE_IDX_T> all_indices;
    
      if (rank == 0)
        {
          // Calculate counts and displacements
          for (unsigned int i = 0; i < size; i++)
            {
              indices_counts[i] = rank_assignments.local_dst_indices[i].size();
              if (i > 0)
                {
                  indices_displs[i] = indices_displs[i-1] + indices_counts[i-1];
                }
            }
          
          // Create flattened array
          size_t total_indices = indices_displs[size-1] + indices_counts[size-1];
          all_indices.resize(total_indices);
          
          // Copy data into the flattened array
          for (unsigned int i = 0; i < size; i++)
            {
              if (indices_counts[i] > 0)
                {
                  std::copy(
                            rank_assignments.local_dst_indices[i].begin(),
                            rank_assignments.local_dst_indices[i].end(),
                            all_indices.begin() + indices_displs[i]
                            );
                }
            }
        }
    
      // Broadcast the counts and displacements
      bcast_req_count = 0;
      throw_assert(MPI_Ibcast(indices_counts.data(), size, MPI_INT, 0, comm,
                              &bcast_req[bcast_req_count++]) == MPI_SUCCESS,
                   "error in MPI_Ibcast");
      throw_assert(MPI_Ibcast(indices_displs.data(), size, MPI_INT, 0, comm,
                              &bcast_req[bcast_req_count++]) == MPI_SUCCESS,
                   "error in MPI_Ibcast");

      // Wait for count and displacement broadcasts to complete
      throw_assert(MPI_Waitall(bcast_req_count, bcast_req, MPI_STATUSES_IGNORE) == MPI_SUCCESS,
                   "error in MPI_Waitall");
    
      // Resize the local destination indices for this rank
      if (rank != 0)
        {
          rank_assignments.local_dst_indices[rank].resize(indices_counts[rank]);
        }
    
      // Scatter the destination indices to each rank
      // All ranks must participate in MPI_Scatterv, even with zero counts
      NODE_IDX_T* recv_buffer = indices_counts[rank] > 0 ? 
        rank_assignments.local_dst_indices[rank].data() : 
        nullptr;
      
      MPI_Request scatter_req;

      throw_assert(MPI_Iscatterv(rank == 0 ? all_indices.data() : nullptr,
                                indices_counts.data(), indices_displs.data(), MPI_NODE_IDX_T,
                                recv_buffer, indices_counts[rank], MPI_NODE_IDX_T,
                                0, comm, &scatter_req
                                ) == MPI_SUCCESS,
                   "error in MPI_Scatterv");
      throw_assert(MPI_Wait(&scatter_req, MPI_STATUS_IGNORE) == MPI_SUCCESS,
                   "error in MPI_Wait");

    }

    // Function to assign destination blocks to ranks
    void assign_blocks_to_ranks
    (
     const vector<DST_BLK_PTR_T>& dst_blk_ptr,
     const vector<NODE_IDX_T>&    dst_idx,
     const vector<DST_PTR_T>&     dst_ptr,
     RankAssignments              &rank_assignments,
     unsigned int                 size,
     size_t                       offset = 0,
     size_t                       numitems = 0
     )
    {

      // Calculate the range to process
      hsize_t total_blocks = dst_blk_ptr.size() - 1;
      hsize_t read_blocks = 0;
    
      if (numitems > 0)
        {
          // Read a specific number of blocks starting from offset
          if (offset < total_blocks)
            {
              read_blocks = std::min((hsize_t)numitems, total_blocks - offset);
            } else
            {
              // Offset is beyond available blocks
              read_blocks = 0;
            }
        } else
        {
          // Read all blocks
          read_blocks = total_blocks;
          offset = 0;
        }
        
    // If nothing to read, return with empty assignments
    if (read_blocks == 0)
      {
        for (unsigned int i = 0; i < size; i++)
          {
            rank_assignments.dst_block_start[i] = 0;
            rank_assignments.dst_block_count[i] = 0;
            rank_assignments.src_idx_start[i] = 0;
            rank_assignments.src_idx_count[i] = 0;
            rank_assignments.dst_ptr_start[i] = 0;
            rank_assignments.dst_ptr_count[i] = 0;
            rank_assignments.local_dst_indices[i].clear();
          }
        return;
    }

      // Simple approach: distribute blocks evenly
      hsize_t blocks_per_rank = total_blocks / size;
      hsize_t remainder = total_blocks % size;

      // Calculate the end of the destination block pointer range
      hsize_t last_block = offset + read_blocks - 1;
      hsize_t block_ptr_end = (last_block < total_blocks) ? 
        dst_blk_ptr[last_block + 1] : dst_blk_ptr[total_blocks];

      hsize_t current_block = offset;
      hsize_t dst_ptr_end = dst_ptr.size()-1;
      if (last_block < total_blocks)
        {
          dst_ptr_end = std::min(dst_ptr_end, block_ptr_end);
        }
    
      for (unsigned int i = 0; i < size; i++)
        {
          // Calculate how many blocks this rank gets
          hsize_t rank_block_count = blocks_per_rank + (i < remainder ? 1 : 0);
        
          // Assign blocks to ranks
          rank_assignments.dst_block_start[i] = current_block;
          rank_assignments.dst_block_count[i] = rank_block_count;
        
          if (rank_block_count > 0)
            {
              // Calculate the end block for this rank
              hsize_t block_end = current_block + rank_block_count - 1;
              hsize_t dst_ptr_start=0, dst_ptr_count=0;
              
              // Calculate destination pointer range
              dst_ptr_start = dst_blk_ptr[current_block];
              dst_ptr_count = dst_blk_ptr[block_end + 1] - dst_blk_ptr[current_block];
              rank_assignments.dst_ptr_start[i] = dst_ptr_start;
              rank_assignments.dst_ptr_count[i] = dst_ptr_count;

              // Calculate source index range
              rank_assignments.src_idx_start[i] = dst_ptr[dst_ptr_start];
              rank_assignments.src_idx_count[i] = dst_ptr[min(dst_ptr_start + dst_ptr_count, dst_ptr_end)] - dst_ptr[dst_ptr_start];
              
              // Store relevant destination indices
              rank_assignments.local_dst_indices[i].clear();
              for (hsize_t j = current_block; j <= block_end; j++)
                {
                  rank_assignments.local_dst_indices[i].push_back(dst_idx[j]);
                }
              rank_assignments.last_rank = i;
            } else
            {
              // This rank gets no blocks
              rank_assignments.dst_ptr_start[i] = 0;
              rank_assignments.dst_ptr_count[i] = 0;
              rank_assignments.src_idx_start[i] = 0;
              rank_assignments.src_idx_count[i] = 0;
              rank_assignments.local_dst_indices[i].clear();
            }
          
          current_block += rank_block_count;
        }
    }

    // Function for each rank to read its portion of src_idx
    herr_t read_projection_src_idx
    (
     MPI_Comm                   comm,
     const std::string&         file_name,
     const std::string&         src_pop_name,
     const std::string&         dst_pop_name,
     const RankAssignments&      rank_assignments,
     vector<NODE_IDX_T>&        src_idx
     )
    {
      herr_t ierr = 0;
      unsigned int rank, size;
      MPI_Comm_size(comm, (int*)&size);
      MPI_Comm_rank(comm, (int*)&rank);
    
      //      if (rank_assignments.src_idx_count[rank] > 0)
        {
          // Open the file with parallel access
          hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
#ifdef HDF5_IS_PARALLEL
          H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL);
#endif
          hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, fapl);
        
          // Create property list for collective dataset operations
          hid_t rapl = H5Pcreate(H5P_DATASET_XFER);
#ifdef HDF5_IS_PARALLEL
          H5Pset_dxpl_mpio(rapl, H5FD_MPIO_COLLECTIVE);
#endif
        
          // Resize to accommodate the data
          src_idx.resize(rank_assignments.src_idx_count[rank], 0);
        
          // Read the data
          ierr = hdf5::read<NODE_IDX_T>
            (
             file,
             hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::SRC_IDX),
             rank_assignments.src_idx_start[rank],
             rank_assignments.src_idx_count[rank],
             NODE_IDX_H5_NATIVE_T,
             src_idx,
             rapl
             );
          
          H5Pclose(rapl);
          H5Fclose(file);
          H5Pclose(fapl);
        }
    
      return ierr;
    }


    // Function to distribute pointer arrays
    void distribute_ptr_arrays
    (
     MPI_Comm                   comm,
     unsigned int               rank,
     const RankAssignments&     assignments,
     vector<DST_BLK_PTR_T>&     dst_blk_ptr,
     vector<NODE_IDX_T>&        dst_idx,
     vector<DST_PTR_T>&         dst_ptr,
     const vector<DST_BLK_PTR_T>& full_dst_blk_ptr, // Original arrays on rank 0
     const vector<NODE_IDX_T>&  full_dst_idx,
     const vector<DST_PTR_T>&   full_dst_ptr
     )
    {
      unsigned int size;
      MPI_Comm_size(comm, (int*)&size);
    
      // Arrays to gather the count and displacement data for the collective operations
      vector<int> blk_ptr_counts(size);
      vector<int> blk_ptr_displs(size, 0);
      vector<int> dst_idx_counts(size);
      vector<int> dst_idx_displs(size, 0);
      vector<int> dst_ptr_counts(size);
      vector<int> dst_ptr_displs(size, 0);
      
      // To store rebased pointers on rank 0 before distribution
      vector<DST_BLK_PTR_T> all_rebased_blk_ptr;
      vector<NODE_IDX_T> all_dst_idx;
      vector<DST_PTR_T> all_rebased_dst_ptr;
      
      // If on rank 0, prepare data for all ranks
      if (rank == 0)
        {
          // Gather information about block counts and displacements for all ranks
          for (unsigned int r = 0; r < size; r++)
            {
              // Get values from the assignments structure
              hsize_t r_dst_block_count = assignments.dst_block_count[r];
              hsize_t r_dst_ptr_count = assignments.dst_ptr_count[r];
              
              // Set counts for the scatter operations and add 1 for the sentinel value in dst_blk_ptr
              blk_ptr_counts[r] = r_dst_block_count > 0 ? r_dst_block_count + 1 : 0;
              dst_idx_counts[r] = r_dst_block_count;     // One dst_idx per block
              dst_ptr_counts[r] = ((r_dst_ptr_count > 0) && (r < assignments.last_rank)) ?
                r_dst_ptr_count + 1 : r_dst_ptr_count;
              
              // Calculate displacements for the scatter operations
              if (r > 0)
                {
                  blk_ptr_displs[r] = blk_ptr_displs[r-1] + blk_ptr_counts[r-1];
                  dst_idx_displs[r] = dst_idx_displs[r-1] + dst_idx_counts[r-1];
                  dst_ptr_displs[r] = dst_ptr_displs[r-1] + dst_ptr_counts[r-1];
                }
            }
        
          // Allocate space for all rebased pointers
          size_t total_blk_ptr_count = blk_ptr_displs[size-1] + blk_ptr_counts[size-1];
          size_t total_dst_idx_count = dst_idx_displs[size-1] + dst_idx_counts[size-1];
          size_t total_dst_ptr_count = dst_ptr_displs[size-1] + dst_ptr_counts[size-1];
          all_rebased_blk_ptr.resize(total_blk_ptr_count);
          all_dst_idx.resize(total_dst_idx_count);
          all_rebased_dst_ptr.resize(total_dst_ptr_count);
          
          // Prepare rebased pointers for each rank
          for (unsigned int r = 0; r < size; r++)
            {
              // Skip ranks with no data
              if (blk_ptr_counts[r] <= 0) continue;
              
              hsize_t r_dst_block_start = assignments.dst_block_start[r];
              //hsize_t r_dst_block_count = assignments.dst_block_count[r];
              hsize_t r_dst_ptr_start = assignments.dst_ptr_start[r];
              
              // Rebase dst_blk_ptr
              for (int i = 0; i < blk_ptr_counts[r]; i++)
                {
                  all_rebased_blk_ptr[blk_ptr_displs[r] + i] = 
                    full_dst_blk_ptr[r_dst_block_start + i] - full_dst_blk_ptr[r_dst_block_start];
                }

              // Copy dst_idx
              for (int i = 0; i < dst_idx_counts[r]; i++)
                {
                  all_dst_idx[dst_idx_displs[r] + i] = full_dst_idx[r_dst_block_start + i];
                }
              
              // Rebase dst_ptr
              if (dst_ptr_counts[r] > 0)
                {
                  DST_PTR_T base = full_dst_ptr[r_dst_ptr_start];
                  for (int i = 0; i < dst_ptr_counts[r]; i++) {
                    all_rebased_dst_ptr[dst_ptr_displs[r] + i] = 
                      full_dst_ptr[r_dst_ptr_start + i] - base;
                  }
                }
            }
        }

      // Broadcast the counts and displacements to all ranks
      // Requests for non-blocking operations
      MPI_Request bcast_req[6];
      int bcast_req_count = 0;
      
      throw_assert(MPI_Ibcast(blk_ptr_counts.data(), size, MPI_INT, 0, comm, &bcast_req[bcast_req_count++]) == MPI_SUCCESS,
                   "error in MPI_Ibcast");
      throw_assert(MPI_Ibcast(blk_ptr_displs.data(), size, MPI_INT, 0, comm, &bcast_req[bcast_req_count++]) == MPI_SUCCESS,
                   "error in MPI_Ibcast");
      throw_assert(MPI_Ibcast(dst_idx_counts.data(), size, MPI_INT, 0, comm, &bcast_req[bcast_req_count++]) == MPI_SUCCESS,
                   "error in MPI_Ibcast");
      throw_assert(MPI_Ibcast(dst_idx_displs.data(), size, MPI_INT, 0, comm, &bcast_req[bcast_req_count++]) == MPI_SUCCESS,
                   "error in MPI_Ibcast");
      throw_assert(MPI_Ibcast(dst_ptr_counts.data(), size, MPI_INT, 0, comm, &bcast_req[bcast_req_count++]) == MPI_SUCCESS,
                   "error in MPI_Ibcast");
      throw_assert(MPI_Ibcast(dst_ptr_displs.data(), size, MPI_INT, 0, comm, &bcast_req[bcast_req_count++]) == MPI_SUCCESS,
                   "error in MPI_Ibcast");
    
      // Wait for count and displacement broadcasts to complete
      throw_assert(MPI_Waitall(bcast_req_count, bcast_req, MPI_STATUSES_IGNORE) == MPI_SUCCESS,
                   "error in MPI_Waitall");
    
      // Resize the destination arrays based on this rank's counts
      if (blk_ptr_counts[rank] > 0)
        {
          dst_blk_ptr.resize(blk_ptr_counts[rank]);
        } else
        {
          dst_blk_ptr.clear();
        }
      
      if (dst_idx_counts[rank] > 0)
        {
          dst_idx.resize(dst_idx_counts[rank]);
        } else
        {
          dst_idx.clear();
        }
      
      if (dst_ptr_counts[rank] > 0)
        {
          dst_ptr.resize(dst_ptr_counts[rank]);
        } else {
        dst_ptr.clear();
      }
    
      // Now perform the non-blocking scatter operations
      // Requests for non-blocking operations
      MPI_Request scatter_req[3];
      int scatter_req_count = 0;

      // All ranks must participate in MPI_Scatterv, even with zero counts
      DST_BLK_PTR_T* dst_blk_ptr_recv_buffer = !dst_blk_ptr.empty() ? 
        dst_blk_ptr.data() :
        nullptr;

      throw_assert(MPI_Iscatterv(
                                 rank == 0 ? all_rebased_blk_ptr.data() : nullptr,
                                 blk_ptr_counts.data(), blk_ptr_displs.data(), MPI_ATTR_PTR_T,
                                 dst_blk_ptr_recv_buffer, dst_blk_ptr.size(), MPI_ATTR_PTR_T,
                                 0, comm, &scatter_req[scatter_req_count++]
                                 ) == MPI_SUCCESS,
                   "error in MPI_Iscatterv");

      // Scatter dst_idx
      NODE_IDX_T* dst_idx_recv_buffer = !dst_idx.empty() ? 
        dst_idx.data() :
        nullptr;
      
      throw_assert(MPI_Iscatterv(
                                 rank == 0 ? all_dst_idx.data() : nullptr,
                                 dst_idx_counts.data(), dst_idx_displs.data(), MPI_NODE_IDX_T,
                                 dst_idx_recv_buffer, dst_idx.size(), MPI_NODE_IDX_T,
                                 0, comm, &scatter_req[scatter_req_count++]
                                 ) == MPI_SUCCESS,
                   "error in MPI_Iscatterv");
      
      DST_PTR_T* dst_ptr_recv_buffer = !dst_ptr.empty() ? 
        dst_ptr.data() :
        nullptr;

      throw_assert(MPI_Iscatterv(
                                 rank == 0 ? all_rebased_dst_ptr.data() : nullptr,
                                 dst_ptr_counts.data(), dst_ptr_displs.data(), MPI_ATTR_PTR_T,
                                 dst_ptr_recv_buffer, dst_ptr.size(), MPI_ATTR_PTR_T,
                                 0, comm, &scatter_req[scatter_req_count++]
                                 ) == MPI_SUCCESS,
                   "error in MPI_Iscatterv");

      // Wait for all operations to complete
      throw_assert(MPI_Waitall(scatter_req_count, scatter_req, MPI_STATUSES_IGNORE) == MPI_SUCCESS,
                   "error in MPI_Waitall");

      
    }
    
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

      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "error in MPI_Barrier");

      // Structure to hold per-rank assignments
      RankAssignments rank_assignments(size);
    
      // Only rank 0 reads the pointer data and partitions the data
      if (rank == 0)
        {
          // Step 1: Read pointer datasets
          vector<DST_BLK_PTR_T> full_dst_blk_ptr;
          vector<DST_PTR_T> full_dst_ptr;
          vector<NODE_IDX_T> full_dst_idx;
    
          ierr = read_projection_ptr(file_name, src_pop_name, dst_pop_name, 
                                     full_dst_blk_ptr, full_dst_idx, full_dst_ptr, 
                                     total_num_edges, total_read_blocks);


          // Step 2: Assign destination blocks to ranks
          assign_blocks_to_ranks(full_dst_blk_ptr, full_dst_idx, full_dst_ptr, rank_assignments, size);

          // Step 3: Distribute appropriate parts of pointer arrays to all ranks
          distribute_ptr_arrays(comm, rank, rank_assignments, dst_blk_ptr, dst_idx, dst_ptr, 
                                full_dst_blk_ptr, full_dst_idx, full_dst_ptr);
        }
      else
        {
          
          // Other ranks just participate in the distribution
          distribute_ptr_arrays(comm, rank, rank_assignments, dst_blk_ptr, dst_idx, dst_ptr, 
                                vector<DST_BLK_PTR_T>(), vector<NODE_IDX_T>(), vector<DST_PTR_T>());
        }

      if (debug_enabled)
        {
          print_rank_assignments(comm, rank_assignments);
        }
      
      // Step 4: Broadcast total edges and blocks to all ranks
      MPI_Request bcast_req[2];
      int bcast_req_count = 0;

      throw_assert(MPI_Ibcast(&total_num_edges, 1, MPI_SIZE_T, 0, comm, &bcast_req[bcast_req_count++]) == MPI_SUCCESS,
                   "error in MPI_Ibcast");
      throw_assert(MPI_Ibcast(&total_read_blocks, 1, MPI_UNSIGNED_LONG_LONG, 0, comm, &bcast_req[bcast_req_count++]) == MPI_SUCCESS,
                   "error in MPI_Ibcast");

      // Wait for broadcasts to complete
      throw_assert(MPI_Waitall(bcast_req_count, bcast_req, MPI_STATUSES_IGNORE) == MPI_SUCCESS,
                   "error in MPI_Waitall");

      // Step 5: Distribute assignments to all ranks
      distribute_assignments(comm, rank_assignments);
      block_base = rank_assignments.dst_block_start[rank];
      edge_base = rank_assignments.src_idx_start[rank];

      
      // Step 6: Each rank reads its portion of src_idx based on its assignment
      ierr = read_projection_src_idx(comm, file_name, src_pop_name, dst_pop_name, 
                                     rank_assignments, src_idx);

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

