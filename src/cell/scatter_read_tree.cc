// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file scatter_read_tree.cc
///
///  Read and scatter tree structures.
///
///  Copyright (C) 2016-2019 Project NeuroH5.
//==============================================================================

#include <mpi.h>
#include <hdf5.h>

#include <vector>
#include <map>

#include "neuroh5_types.hh"
#include "attr_map.hh"
#include "cell_attributes.hh"
#include "rank_range.hh"
#include "append_rank_tree_map.hh"
#include "alltoallv_template.hh"
#include "serialize_tree.hh"
#include "dataset_num_elements.hh"
#include "path_names.hh"
#include "read_template.hh"
#include "throw_assert.hh"

using namespace std;

namespace neuroh5
{

  namespace cell
  {
  
    /*****************************************************************************
     * Load tree data structures from HDF5 and scatter to all ranks
     *****************************************************************************/
    int scatter_read_trees
    (
     MPI_Comm                        all_comm,
     const string                   &file_name,
     const int                       io_size,
     const vector<string>           &attr_name_spaces,
     // A vector that maps nodes to compute ranks
     const map<CELL_IDX_T, rank_t>   &node_rank_map,
     const string                    &pop_name,
     const CELL_IDX_T                 pop_start,
     map<CELL_IDX_T, neurotree_t>    &tree_map,
     map<string, data::NamedAttrMap> &attr_maps,
     size_t offset = 0,
     size_t numitems = 0
     )
    {
      herr_t status; hid_t fapl=-1, rapl=-1, file=-1;
      vector< pair<hsize_t,hsize_t> > ranges;
      size_t dset_size, read_size; hsize_t start=0, end=0, block=0;
    
      vector<char> sendbuf; 
      vector<int> sendcounts, sdispls;
    
      // MPI Communicator for I/O ranks
      MPI_Comm io_comm;
      // MPI group color value used for I/O ranks
      int io_color = 1;

      throw_assert_nomsg(io_size > 0);
    
      int srank, ssize; size_t rank, size;
      throw_assert_nomsg(MPI_Comm_size(all_comm, &ssize) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Comm_rank(all_comm, &srank) == MPI_SUCCESS);
      throw_assert_nomsg(srank >= 0);
      throw_assert_nomsg(ssize > 0);
      rank = srank;
      size = ssize;

      // Am I an I/O rank?
      if (srank < io_size)
        {
          MPI_Comm_split(all_comm,io_color,rank,&io_comm);
          MPI_Comm_set_errhandler(io_comm, MPI_ERRORS_RETURN);
        
          fapl = H5Pcreate(H5P_FILE_ACCESS);
          throw_assert_nomsg(fapl >= 0);
          throw_assert_nomsg(H5Pset_fapl_mpio(fapl, io_comm, MPI_INFO_NULL) >= 0);
        
          /* Create property list for collective dataset operations. */
          rapl = H5Pcreate (H5P_DATASET_XFER);
          status = H5Pset_dxpl_mpio (rapl, H5FD_MPIO_COLLECTIVE);

          file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, fapl);
          throw_assert_nomsg(file >= 0);
          dset_size = hdf5::dataset_num_elements(file, hdf5::cell_attribute_path(hdf5::TREES, string(pop_name), hdf5::CELL_INDEX));

          status = H5Fclose(file);
          throw_assert_nomsg(status == 0);

          status = H5Pclose(fapl);
          throw_assert_nomsg(status == 0);
          
          if (numitems > 0)
            {
              if (offset < dset_size)
                {
                  read_size = min(numitems*size, dset_size-offset);
                }
              else
                {
                  read_size = 0;
                }
            }
          else
            {
              read_size = dset_size;
            }

          if (read_size > 0)
            {
              // determine which blocks of block_ptr are read by which I/O rank
              mpi::rank_ranges(read_size, io_size, ranges);
              
              start = ranges[rank].first + offset;
              end   = start + ranges[rank].second;

              block = end - start + 1;
            }
          else
            {
              block = 0;
            }

        }
      else
        {
          MPI_Comm_split(all_comm,0,rank,&io_comm);
        }
      MPI_Barrier(all_comm);
    
      sendcounts.resize(size,0);
      sdispls.resize(size,0);

      if (block > 0)
        {
          map <rank_t, map<CELL_IDX_T, neurotree_t> > rank_tree_map;

          {
            data::NamedAttrMap attr_values;

            read_cell_attributes (io_comm, file_name, hdf5::TREES,
                                  pop_name, pop_start, attr_values,
                                  offset, numitems * size);

            
            data::append_rank_tree_map(attr_values, node_rank_map, rank_tree_map);
          }

          data::serialize_rank_tree_map (size, rank, rank_tree_map, sendcounts, sendbuf, sdispls);
        }

      throw_assert_nomsg(MPI_Comm_free(&io_comm) == MPI_SUCCESS);

      vector<int> recvcounts, rdispls;
      vector<char> recvbuf;


      throw_assert_nomsg(mpi::alltoallv_vector<char>(all_comm, MPI_CHAR, sendcounts, sdispls, sendbuf,
                                                     recvcounts, rdispls, recvbuf) >= 0);
      sendbuf.clear();

      if (recvbuf.size() > 0)
        {
          data::deserialize_rank_tree_map (size, recvbuf, recvcounts, rdispls, tree_map);
        }
      recvbuf.clear();


      for (string attr_name_space : attr_name_spaces)
        {
          data::NamedAttrMap attr_map;
          scatter_read_cell_attributes(all_comm, file_name, io_size,
                                       attr_name_space, node_rank_map,
                                       pop_name, pop_start, attr_map,
                                       offset, numitems);
          attr_maps.insert(make_pair(attr_name_space, attr_map));
        }

    
      return 0;
    }
  }
}
