// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file scatter_read_tree.cc
///
///  Read and scatter tree structures.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================

#include <mpi.h>
#include <hdf5.h>

#include <cassert>
#include <vector>
#include <map>

#include "neuroh5_types.hh"
#include "attr_map.hh"
#include "cell_attributes.hh"
#include "rank_range.hh"
#include "append_rank_tree_map.hh"
#include "alltoallv_packed.hh"
#include "pack_tree.hh"
#include "dataset_num_elements.hh"
#include "bcast_string_vector.hh"
#include "enum_type.hh"
#include "path_names.hh"
#include "read_template.hh"

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
      herr_t status; hid_t fapl=-1, rapl=-1, file=-1, hdf5_swc_type=-1;
      vector< pair<hsize_t,hsize_t> > ranges;
      size_t dset_size, read_size; hsize_t start=0, end=0, block=0;
    
      vector<uint8_t> sendbuf; int sendpos = 0;
      vector<int> sendcounts, sdispls;
    
      // MPI Communicator for I/O ranks
      MPI_Comm io_comm;
      // MPI group color value used for I/O ranks
      int io_color = 1;

      assert(io_size > 0);
    
      int srank, ssize; size_t rank, size;
      assert(MPI_Comm_size(all_comm, &ssize) >= 0);
      assert(MPI_Comm_rank(all_comm, &srank) >= 0);
      assert(srank >= 0);
      assert(ssize > 0);
      rank = srank;
      size = ssize;

      /* Create HDF5 enumerated type for reading SWC type information */
      hdf5_swc_type = hdf5::create_H5Tenum<SWC_TYPE_T> (swc_type_enumeration);

      // Am I an I/O rank?
      if (srank < io_size)
        {
          MPI_Comm_split(all_comm,io_color,rank,&io_comm);
          MPI_Comm_set_errhandler(io_comm, MPI_ERRORS_RETURN);
        
          fapl = H5Pcreate(H5P_FILE_ACCESS);
          assert(fapl >= 0);
          assert(H5Pset_fapl_mpio(fapl, io_comm, MPI_INFO_NULL) >= 0);
        
          /* Create property list for collective dataset operations. */
          rapl = H5Pcreate (H5P_DATASET_XFER);
          status = H5Pset_dxpl_mpio (rapl, H5FD_MPIO_COLLECTIVE);

          file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, fapl);
          assert(file >= 0);
          dset_size = hdf5::dataset_num_elements(io_comm, file, hdf5::cell_attribute_path(hdf5::TREES, string(pop_name), hdf5::CELL_INDEX));

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
            vector<SEC_PTR_T>  sec_ptr;
            vector<TOPO_PTR_T> topo_ptr;
            vector<ATTR_PTR_T> attr_ptr;
          
            // allocate buffer and memory dataspace
            attr_ptr.resize(block);
            status = hdf5::read<ATTR_PTR_T> (file,
                                             hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::ATTR_PTR),
                                             start, block,
                                             ATTR_PTR_H5_NATIVE_T,
                                             attr_ptr, rapl);
            assert(status >= 0);
          
            sec_ptr.resize(block);
            status = hdf5::read<SEC_PTR_T> (file,
                                            hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::SEC_PTR),
                                            start, block,
                                            SEC_PTR_H5_NATIVE_T,
                                            sec_ptr, rapl);
            assert(status >= 0);
          
            topo_ptr.resize(block);
            status = hdf5::read<TOPO_PTR_T> (file,
                                             hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::TOPO_PTR),
                                             start, block,
                                             TOPO_PTR_H5_NATIVE_T,
                                             topo_ptr, rapl);
            assert(status >= 0);
            
            vector<CELL_IDX_T> index_vector;
            vector<SECTION_IDX_T> src_vector, dst_vector;
            vector<SECTION_IDX_T> sections;
            vector<COORD_T> xcoords;
            vector<COORD_T> ycoords;
            vector<COORD_T> zcoords;
            vector<REALVAL_T> radiuses;
            vector<LAYER_IDX_T> layers;
            vector<PARENT_NODE_IDX_T> parents;
            vector<SWC_TYPE_T> swc_types;

            index_vector.resize(block-1);
            status = hdf5::read<CELL_IDX_T> (file,
                                             hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::CELL_INDEX),
                                             start, block-1,
                                             CELL_IDX_H5_NATIVE_T,
                                             index_vector, rapl);

          
            hsize_t topo_start = topo_ptr[0];
            size_t topo_block = topo_ptr.back()-topo_start;
          
            src_vector.resize(topo_block);
            status = hdf5::read<SECTION_IDX_T> (file,
                                                hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::SRCSEC),
                                                topo_start, topo_block,
                                                SECTION_IDX_H5_NATIVE_T,
                                                src_vector, rapl);
            assert(status == 0);
            dst_vector.resize(topo_block);
            status = hdf5::read<SECTION_IDX_T> (file,
                                                hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::DSTSEC),
                                                topo_start, topo_block,
                                                SECTION_IDX_H5_NATIVE_T,
                                                dst_vector, rapl);
            assert(status == 0);
          
            hsize_t sec_start = sec_ptr[0];
            size_t sec_block = sec_ptr.back()-sec_start;
          
            sections.resize(sec_block);
            status = hdf5::read<SECTION_IDX_T> (file,
                                                hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::SECTION),
                                                sec_start, sec_block,
                                                SECTION_IDX_H5_NATIVE_T,
                                                sections, rapl);
            assert(status == 0);
          
          
          
            hsize_t attr_start = attr_ptr[0];
            size_t attr_block = attr_ptr.back()-attr_start;
          
            xcoords.resize(attr_block);
            status = hdf5::read<COORD_T> (file,
                                          hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::X_COORD),
                                          attr_start, attr_block,
                                          COORD_H5_NATIVE_T,
                                          xcoords, rapl);
            assert(status == 0);
            ycoords.resize(attr_block);
            status = hdf5::read<COORD_T> (file,
                                          hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::Y_COORD),
                                          attr_start, attr_block,
                                          COORD_H5_NATIVE_T,
                                          ycoords, rapl);
            assert(status == 0);
            zcoords.resize(attr_block);
            status = hdf5::read<COORD_T> (file,
                                          hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::Z_COORD),
                                          attr_start, attr_block,
                                          COORD_H5_NATIVE_T,
                                          zcoords, rapl);
            assert(status == 0);
            radiuses.resize(attr_block);
            status = hdf5::read<REALVAL_T> (file,
                                            hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::RADIUS),
                                            attr_start, attr_block,
                                            REAL_H5_NATIVE_T,
                                            radiuses, rapl);
            assert(status == 0);
            layers.resize(attr_block);
            status = hdf5::read<LAYER_IDX_T> (file,
                                              hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::LAYER),
                                              attr_start, attr_block,
                                              LAYER_IDX_H5_NATIVE_T,
                                              layers, rapl);
            assert(status == 0);
            parents.resize(attr_block);
            status = hdf5::read<PARENT_NODE_IDX_T> (file,
                                                    hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::PARENT),
                                                    attr_start, attr_block,
                                                    PARENT_NODE_IDX_H5_NATIVE_T,
                                                    parents, rapl);
            assert(status == 0);
            swc_types.resize(attr_block);
            status = hdf5::read<SWC_TYPE_T> (file,
                                             hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::SWCTYPE),
                                             attr_start, attr_block,
                                             hdf5_swc_type,
                                             swc_types, rapl);
            assert(status == 0);
          
          
            status = H5Fclose (file);
            assert(status == 0);
            status = H5Pclose (fapl);
            assert(status == 0);
            status = H5Tclose(hdf5_swc_type);
            assert(status == 0);
            status = H5Pclose(rapl);
            assert(status == 0);

            data::append_rank_tree_map(start, block-1, 
                                       node_rank_map, pop_start,
                                       sec_ptr, topo_ptr, attr_ptr,
                                       index_vector, src_vector, dst_vector, sections,
                                       xcoords, ycoords, zcoords,
                                       radiuses, layers, parents,
                                       swc_types, rank_tree_map);
          }

          assert(mpi::pack_rank_tree_map (all_comm, rank_tree_map, sendcounts, sdispls, sendpos, sendbuf) >= 0);
        }

      vector<int> recvcounts, rdispls;
      vector<uint8_t> recvbuf;
      assert(mpi::alltoallv_packed(all_comm, sendcounts, sdispls, sendbuf,
                                   recvcounts, rdispls, recvbuf) >= 0);
      sendbuf.clear();

      int recvpos = 0;
      assert(mpi::unpack_tree_map (all_comm, recvbuf, recvpos, tree_map) >= 0);
      recvbuf.clear();

      assert(MPI_Comm_free(&io_comm) == MPI_SUCCESS);

      for (string attr_name_space : attr_name_spaces)
        {
          data::NamedAttrMap attr_map;
          scatter_read_cell_attributes(all_comm, file_name, io_size,
                                       attr_name_space, node_rank_map,
                                       pop_name, pop_start, attr_map);
          attr_maps.insert(make_pair(attr_name_space, attr_map));
        }

    
      return 0;
    }
  }
}
