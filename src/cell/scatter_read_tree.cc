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
    
    void append_rank_tree_map
    (
     const size_t start,
     const size_t num_trees,
     const map<CELL_IDX_T, size_t>& node_rank_map,
     const CELL_IDX_T pop_start,
     vector<SEC_PTR_T>& sec_ptr,
     vector<TOPO_PTR_T>& topo_ptr,
     vector<ATTR_PTR_T>& attr_ptr,
     vector<CELL_IDX_T>& all_gid_vector,
     vector<SECTION_IDX_T>& all_src_vector,
     vector<SECTION_IDX_T>& all_dst_vector,
     vector<SECTION_IDX_T>& all_sections,
     vector<COORD_T>& all_xcoords,
     vector<COORD_T>& all_ycoords,
     vector<COORD_T>& all_zcoords,
     vector<REALVAL_T>& all_radiuses,
     vector<LAYER_IDX_T>& all_layers,
     vector<PARENT_NODE_IDX_T>& all_parents,
     vector<SWC_TYPE_T>& all_swc_types,
     map <size_t, map<CELL_IDX_T, neurotree_t> > &rank_tree_map)
    {
      for (size_t i=0; i<num_trees; i++)
        {
          hsize_t topo_start = topo_ptr[i]-topo_ptr[0]; size_t topo_block = topo_ptr[i+1]-topo_ptr[0]-topo_start;

          vector<SECTION_IDX_T>::const_iterator src_first = all_src_vector.begin() + topo_start;
          vector<SECTION_IDX_T>::const_iterator src_last  = all_src_vector.begin() + topo_start + topo_block;
          vector<SECTION_IDX_T> tree_src_vector;
          tree_src_vector.insert(tree_src_vector.begin(),src_first,src_last);
        
          vector<SECTION_IDX_T>::const_iterator dst_first = all_dst_vector.begin() + topo_start;
          vector<SECTION_IDX_T>::const_iterator dst_last  = all_dst_vector.begin() + topo_start + topo_block;
          vector<SECTION_IDX_T> tree_dst_vector;
          tree_dst_vector.insert(tree_dst_vector.begin(),dst_first,dst_last);
        
          hsize_t sec_start = sec_ptr[i]-sec_ptr[0]; size_t sec_block = sec_ptr[i+1]-sec_ptr[0]-sec_start;
        
          vector<SECTION_IDX_T>::const_iterator sec_first = all_sections.begin() + sec_start;
          vector<SECTION_IDX_T>::const_iterator sec_last  = all_sections.begin() + sec_start + sec_block;
          vector<SECTION_IDX_T> tree_sections;
          tree_sections.insert(tree_sections.begin(),sec_first,sec_last);
        
          hsize_t attr_start = attr_ptr[i]-attr_ptr[0]; size_t attr_block = attr_ptr[i+1]-attr_ptr[0]-attr_start;

          vector<COORD_T>::const_iterator xcoords_first = all_xcoords.begin() + attr_start;
          vector<COORD_T>::const_iterator xcoords_last  = all_xcoords.begin() + attr_start + attr_block;
          vector<COORD_T> tree_xcoords;
          tree_xcoords.insert(tree_xcoords.begin(),xcoords_first,xcoords_last);

          vector<COORD_T>::const_iterator ycoords_first = all_ycoords.begin() + attr_start;
          vector<COORD_T>::const_iterator ycoords_last  = all_ycoords.begin() + attr_start + attr_block;
          vector<COORD_T> tree_ycoords;
          tree_ycoords.insert(tree_ycoords.begin(),ycoords_first,ycoords_last);

          vector<COORD_T>::const_iterator zcoords_first = all_zcoords.begin() + attr_start;
          vector<COORD_T>::const_iterator zcoords_last  = all_zcoords.begin() + attr_start + attr_block;
          vector<COORD_T> tree_zcoords;
          tree_zcoords.insert(tree_zcoords.begin(),zcoords_first,zcoords_last);

          vector<REALVAL_T>::const_iterator radiuses_first = all_radiuses.begin() + attr_start;
          vector<REALVAL_T>::const_iterator radiuses_last  = all_radiuses.begin() + attr_start + attr_block;
          vector<REALVAL_T> tree_radiuses;
          tree_radiuses.insert(tree_radiuses.begin(),radiuses_first,radiuses_last);

          vector<LAYER_IDX_T>::const_iterator layers_first = all_layers.begin() + attr_start;
          vector<LAYER_IDX_T>::const_iterator layers_last  = all_layers.begin() + attr_start + attr_block;
          vector<LAYER_IDX_T> tree_layers;
          tree_layers.insert(tree_layers.begin(),layers_first,layers_last);

          vector<PARENT_NODE_IDX_T>::const_iterator parents_first = all_parents.begin() + attr_start;
          vector<PARENT_NODE_IDX_T>::const_iterator parents_last  = all_parents.begin() + attr_start + attr_block;
          vector<PARENT_NODE_IDX_T> tree_parents;
          tree_parents.insert(tree_parents.begin(),parents_first,parents_last);
        
          vector<SWC_TYPE_T>::const_iterator swc_types_first = all_swc_types.begin() + attr_start;
          vector<SWC_TYPE_T>::const_iterator swc_types_last  = all_swc_types.begin() + attr_start + attr_block;
          vector<SWC_TYPE_T> tree_swc_types;
          tree_swc_types.insert(tree_swc_types.begin(),swc_types_first,swc_types_last);

          CELL_IDX_T gid = pop_start+all_gid_vector[i];
          size_t dst_rank;
          auto it = node_rank_map.find(gid);
          if (it == node_rank_map.end())
            {
              printf("gid %d not found in node rank map\n", gid);
            }
          assert(it != node_rank_map.end());
          dst_rank = it->second;
          neurotree_t tree = make_tuple(gid,tree_src_vector,tree_dst_vector,tree_sections,
                                        tree_xcoords,tree_ycoords,tree_zcoords,
                                        tree_radiuses,tree_layers,tree_parents,
                                        tree_swc_types);
          map<CELL_IDX_T, neurotree_t> &tree_map = rank_tree_map[dst_rank];
          tree_map.insert(make_pair(gid, tree));
                                   
        }
    }


  
    /*****************************************************************************
     * Load tree data structures from HDF5 and scatter to all ranks
     *****************************************************************************/
    int scatter_read_trees
    (
     MPI_Comm                      all_comm,
     const string                 &file_name,
     const int                     io_size,
     const bool                    opt_attrs,
     const string                 &attr_name_space,
     // A vector that maps nodes to compute ranks
     const map<CELL_IDX_T,size_t> &node_rank_map,
     const string                 &pop_name,
     const CELL_IDX_T              pop_start,
     map<CELL_IDX_T, neurotree_t> &tree_map,
     data::NamedAttrMap           &attr_map,
     size_t offset = 0,
     size_t numitems = 0
     )
    {
      herr_t status; hid_t fapl, rapl, file, hdf5_swc_type;
      vector< pair<hsize_t,hsize_t> > ranges;
      size_t dset_size, read_size; hsize_t start=0, end=0, block=0;
    
      vector<uint8_t> sendbuf; int sendpos = 0;
      vector<int> sendcounts, sdispls, recvcounts, rdispls;
    
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
    
      // Am I an I/O rank?
      if (srank < io_size)
        {
          MPI_Comm_split(all_comm,io_color,rank,&io_comm);
          MPI_Comm_set_errhandler(io_comm, MPI_ERRORS_RETURN);


          /* Create HDF5 enumerated type for reading SWC type information */
          hdf5_swc_type = hdf5::create_H5Tenum<SWC_TYPE_T> (swc_type_enumeration);
        
          fapl = H5Pcreate(H5P_FILE_ACCESS);
          assert(fapl >= 0);
          assert(H5Pset_fapl_mpio(fapl, io_comm, MPI_INFO_NULL) >= 0);
        
          /* Create property list for collective dataset operations. */
          rapl = H5Pcreate (H5P_DATASET_XFER);
          status = H5Pset_dxpl_mpio (rapl, H5FD_MPIO_COLLECTIVE);

          file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, fapl);
          assert(file >= 0);
          dset_size = hdf5::dataset_num_elements(io_comm, file, hdf5::cell_attribute_path(hdf5::TREES, string(pop_name), hdf5::ATTR_PTR))-1;

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
      recvcounts.resize(size,0);
      rdispls.resize(size,0);

      if (block > 0)
        {
          map <size_t, map<CELL_IDX_T, neurotree_t> > rank_tree_map;

          {
            vector<SEC_PTR_T>  sec_ptr;
            vector<TOPO_PTR_T> topo_ptr;
            vector<ATTR_PTR_T> attr_ptr;

            vector<CELL_IDX_T> gid_vector;
            vector<SECTION_IDX_T> src_vector, dst_vector;
            vector<SECTION_IDX_T> sections;
            vector<COORD_T> xcoords;
            vector<COORD_T> ycoords;
            vector<COORD_T> zcoords;
            vector<REALVAL_T> radiuses;
            vector<LAYER_IDX_T> layers;
            vector<PARENT_NODE_IDX_T> parents;
            vector<SWC_TYPE_T> swc_types;
          
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

            gid_vector.resize(block-1);
            status = hdf5::read<CELL_IDX_T> (file,
                                             hdf5::cell_attribute_path(hdf5::TREES, pop_name, hdf5::CELL_INDEX),
                                             start, block-1,
                                             CELL_IDX_H5_NATIVE_T,
                                             gid_vector, rapl);

          
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

            append_rank_tree_map(start, block-1, 
                                 node_rank_map, pop_start,
                                 sec_ptr, topo_ptr, attr_ptr,
                                 gid_vector, src_vector, dst_vector, sections,
                                 xcoords, ycoords, zcoords,
                                 radiuses, layers, parents,
                                 swc_types, rank_tree_map);
          }

          vector<int> rank_sequence;
          // Recommended all-to-all communication pattern: start at the current rank, then wrap around;
          // (as opposed to starting at rank 0)
          for (size_t dst_rank = rank; dst_rank < size; dst_rank++)
            {
              rank_sequence.push_back(dst_rank);
            }
          for (size_t dst_rank = 0; dst_rank < rank; dst_rank++)
            {
              rank_sequence.push_back(dst_rank);
            }

          for (const size_t& dst_rank : rank_sequence)
            {
              auto it1 = rank_tree_map.find(dst_rank); 
              sdispls[dst_rank] = sendpos;

              if (it1 != rank_tree_map.end())
                {
                  for (auto it2 = it1->second.cbegin(); it2 != it1->second.cend(); ++it2)
                    {
                      CELL_IDX_T  gid   = it2->first;
                      const neurotree_t &tree = it2->second;
                    
                      mpi::pack_tree(all_comm, gid, tree, sendpos, sendbuf);
                    }
                }
              sendcounts[dst_rank] = sendpos - sdispls[dst_rank];
            }

        }

      // 1. Each ALL_COMM rank sends a tree size to
      //    every other ALL_COMM rank (non IO_COMM ranks pass zero),
      //    and creates sendcounts and sdispls arrays

      assert(MPI_Alltoall(&sendcounts[0], 1, MPI_INT,
                          &recvcounts[0], 1, MPI_INT, all_comm) >= 0);
    
      // 2. Each ALL_COMM rank accumulates the vector sizes and allocates
      //    a receive buffer, recvcounts, and rdispls
    
      size_t recvbuf_size = recvcounts[0];
      for (int p = 1; p < ssize; ++p)
        {
          rdispls[p] = rdispls[p-1] + recvcounts[p-1];
          recvbuf_size += recvcounts[p];
        }
      //assert(recvbuf_size > 0);
      vector<uint8_t> recvbuf(recvbuf_size);
    
      // 3. Each ALL_COMM rank participates in the MPI_Alltoallv
      assert(MPI_Alltoallv(&sendbuf[0], &sendcounts[0], &sdispls[0], MPI_PACKED,
                           &recvbuf[0], &recvcounts[0], &rdispls[0], MPI_PACKED,
                           all_comm) >= 0);
      sendbuf.clear();

      int recvpos = 0; 
      while ((size_t)recvpos < recvbuf_size)
        {
          mpi::unpack_tree(all_comm, recvbuf, recvpos, tree_map);
          assert((size_t)recvpos <= recvbuf_size);
        }
      recvbuf.clear();

      assert(MPI_Comm_free(&io_comm) == MPI_SUCCESS);

      if (opt_attrs)
        {
          scatter_read_cell_attributes(all_comm, file_name, io_size,
                                       attr_name_space, node_rank_map,
                                       pop_name, pop_start, attr_map);
        }

    
      return 0;
    }
  }
}
