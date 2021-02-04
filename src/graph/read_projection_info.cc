// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_projection_info.cc
///
///  Functions for reading edge information.
///
///  Copyright (C) 2016-2021 Project NeuroH5.
//==============================================================================


#include "neuroh5_types.hh"
#include "read_projection_info.hh"
#include "edge_attributes.hh"
#include "read_projection_datasets.hh"
#include "validate_edge_list.hh"
#include "append_edge_map.hh"
#include "mpi_debug.hh"
#include "debug.hh"

#include <iostream>
#include <sstream>
#include <string>

#include "throw_assert.hh"

using namespace std;

namespace neuroh5
{
  namespace graph
  {
    
    /**************************************************************************
     * Reads information about the given projection.
     *************************************************************************/

    herr_t read_projection_info
    (
     MPI_Comm                   comm,
     const std::string&         file_name,
     const vector<string>& edge_attr_name_spaces,
     const bool                 read_node_index,
     const pop_range_map_t&     pop_ranges,
     const set < pair<pop_t, pop_t> >& pop_pairs,
     const std::string&         src_pop_name,
     const std::string&         dst_pop_name,
     const NODE_IDX_T           src_start,
     const NODE_IDX_T           dst_start, 
     vector< pair<string, string> >& prj_names,
     vector < map <string, vector < vector<string> > > > & edge_attr_names_vector,
     std::vector<std::vector<NODE_IDX_T>>& prj_node_index
     
     )
    {
      herr_t ierr = 0;
      unsigned int rank, size;
      throw_assert(MPI_Comm_size(comm, (int*)&size) == MPI_SUCCESS,
                   "read_projection: invalid MPI communicator");
      throw_assert(MPI_Comm_rank(comm, (int*)&rank) == MPI_SUCCESS,
                   "read_projection: invalid MPI communicator");

      bool has_projection_flag = false;
      has_projection(comm, file_name, src_pop_name, dst_pop_name, has_projection_flag);
      if (!has_projection_flag)
        return ierr;
      
      DST_BLK_PTR_T block_base;
      DST_PTR_T edge_base;
      vector<DST_BLK_PTR_T> dst_blk_ptr;
      vector<NODE_IDX_T> dst_idx;
      vector<DST_PTR_T> dst_ptr;

      std::vector<NODE_IDX_T> node_index;
      if (read_node_index)
        {
          int io_color = 1;
          MPI_Comm io_comm;

          if (rank == 0)
            {
              MPI_Comm_split(comm,io_color,rank,&io_comm);
              MPI_Comm_set_errhandler(io_comm, MPI_ERRORS_RETURN);
            }
          else
            {
              MPI_Comm_split(comm,0,rank,&io_comm);
            }

          if (rank == 0)
            {
              throw_assert(hdf5::read_projection_node_datasets(io_comm, file_name, src_pop_name, dst_pop_name,
                                                               block_base, edge_base,
                                                               dst_blk_ptr, dst_idx, dst_ptr) >= 0,
                           "read_projection_info: read_projection_node_datasets error");
            }
          throw_assert_nomsg(MPI_Barrier(io_comm) == MPI_SUCCESS);
          MPI_Comm_free(&io_comm);
          
          if (dst_blk_ptr.size() > 0)
            {
              size_t dst_ptr_size = dst_ptr.size();
              for (size_t b = 0; b < dst_blk_ptr.size()-1; ++b)
                {
                  size_t low_dst_ptr = dst_blk_ptr[b],
                    high_dst_ptr = dst_blk_ptr[b+1];
                  
                  NODE_IDX_T dst_base = dst_idx[b];
                  for (size_t i = low_dst_ptr, ii = 0; i < high_dst_ptr; ++i, ++ii)
                    {
                      if (i < dst_ptr_size-1)
                        {
                          NODE_IDX_T node = dst_base + ii + dst_start;
                          node_index.push_back(node);
                        }
                    }
                }
            }

          prj_node_index.push_back(node_index);
        }
      

      map <string, vector < vector<string> > > edge_attr_names;
      for (string attr_namespace : edge_attr_name_spaces) 
        {
          bool has_namespace = false;
          throw_assert(has_edge_attribute_namespace(comm, file_name, src_pop_name, dst_pop_name, attr_namespace, has_namespace) >= 0,
                       "read_projection: has_edge_attribute_namespace error");
          if (has_namespace)
            {
              edge_attr_names[attr_namespace].resize(data::AttrVal::num_attr_types);
              vector< pair<string,AttrKind> > edge_attr_info;
              throw_assert(graph::get_edge_attributes(comm, file_name, src_pop_name, dst_pop_name,
                                                      attr_namespace, edge_attr_info) >= 0,
                           "read_projection: get_edge_attributes error");

              for (auto &attr_it : edge_attr_info)
                {
                  const AttrKind attr_kind = attr_it.second;
                  size_t attr_size = attr_kind.size;

                  size_t attr_index;
                  switch (attr_kind.type)
                    {
                    case UIntVal:
                      if (attr_size == 4)
                        {
                          attr_index = data::AttrVal::attr_index_uint32;
                        }
                      else if (attr_size == 2)
                        {
                          attr_index = data::AttrVal::attr_index_uint16;
                        }
                      else if (attr_size == 1)
                        {
                          attr_index = data::AttrVal::attr_index_uint8;
                        }
                      else
                        {
                          throw runtime_error("Unsupported integer attribute size");
                        };
                      break;
                    case SIntVal:
                      if (attr_size == 4)
                        {
                          attr_index = data::AttrVal::attr_index_int32;
                        }
                      else if (attr_size == 2)
                        {
                          attr_index = data::AttrVal::attr_index_int16;
                        }
                      else if (attr_size == 1)
                        {
                          attr_index = data::AttrVal::attr_index_int8;
                        }
                      else
                        {
                          throw runtime_error("Unsupported integer attribute size");
                        };
                      break;
                    case FloatVal:
                      attr_index = data::AttrVal::attr_index_float;
                      break;
                    case EnumVal:
                      if (attr_size == 1)
                        {
                          attr_index = data::AttrVal::attr_index_uint8;
                        }
                      else
                        {
                          throw runtime_error("Unsupported enumerated attribute size");
                        };
                      break;
                    default:
                      throw runtime_error("Unsupported attribute type");
                      break;
                    }
                  
                  edge_attr_names[attr_namespace][attr_index].push_back(attr_it.first);
                }
            }
        }
      
      size_t local_prj_num_edges=0;

      prj_names.push_back(make_pair(src_pop_name, dst_pop_name));
      edge_attr_names_vector.push_back (edge_attr_names);
#ifdef NEUROH5_DEBUG          
      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                   "read_projection_info: error in MPI_Barrier");
#endif
      
      return ierr;
    }


    /////////////////////////////////////////////////////////////////////////
    herr_t has_projection
    (
     MPI_Comm                      comm,
     const string&                 file_name,
     const string&                 src_pop_name,
     const string&                 dst_pop_name,
     bool &has_projection
     )
    {
      herr_t ierr=0;
      int root=0;
      int rank, size;
      throw_assert_nomsg(MPI_Comm_size(comm, &size) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS);
      uint8_t has_projection_flag = 0;

      if (rank == root)
        {
          hid_t in_file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
          throw_assert_nomsg(in_file >= 0);
          
          string path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_BLK_PTR);
          
          ierr = hdf5::exists_dataset (in_file, path);
          if (ierr > 0)
            {
              has_projection_flag = 1;
            }
          else
            {
              has_projection_flag = 0;
            }
          ierr = H5Fclose(in_file);
        }

      throw_assert_nomsg(MPI_Bcast(&has_projection_flag, 1, MPI_UINT8_T, root, comm) == MPI_SUCCESS);
      
      if (has_projection_flag > 0)
        {
          has_projection = true;
        }
      else
        {
          has_projection = false;
        }
      return ierr;
    }

    
  }
}

