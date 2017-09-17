

#include "debug.hh"
#include "neuroh5_types.hh"
#include "path_names.hh"
#include "append_projection.hh"
#include "write_template.hh"
#include "edge_attributes.hh"

#include <algorithm>
#include <cassert>
#include <map>

using namespace std;

namespace neuroh5
{
  namespace graph
  {

    template <class T>
    void append_edge_attribute_map (hid_t file,
                                    const string &src_pop_name,
                                    const string &dst_pop_name,
                                    const map <string, data::NamedAttrVal>& edge_attr_map,
                                    const map <string, vector < vector <string> > >& edge_attr_names)
    {
      for (auto iter : edge_attr_map)
        {
          const string& attr_namespace = iter.first;
          const data::NamedAttrVal& edge_attr_values = iter.second;
          
          for (size_t i=0; i<edge_attr_values.size_attr_vec<T>(); i++)
            {
              const string& attr_name = edge_attr_names.at(attr_namespace)[data::AttrVal::attr_type_index<T>()][i];
              string path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name, attr_namespace, attr_name);
              graph::write_edge_attribute<T>(file, path, edge_attr_values.attr_vec<T>(i));
            }
        }
    }


    void append_projection
    (
     hid_t                     file,
     const string&             src_pop_name,
     const string&             dst_pop_name,
     const NODE_IDX_T&         src_start,
     const NODE_IDX_T&         src_end,
     const NODE_IDX_T&         dst_start,
     const NODE_IDX_T&         dst_end,
     const size_t&             num_edges,
     const edge_map_t&         prj_edge_map,
     const map<string, vector < vector<string> > >& edge_attr_names,
     const hsize_t             cdim,
     const hsize_t             block_size
     )
    {
      // do a sanity check on the input
      assert(src_start < src_end);
      assert(dst_start < dst_end);

      // get the I/O communicator
      MPI_Comm comm;
      MPI_Info info;
      hid_t fapl = H5Fget_access_plist(file);
      assert(H5Pget_fapl_mpio(fapl, &comm, &info) >= 0);

      int ssize, srank;
      assert(MPI_Comm_size(comm, &ssize) == MPI_SUCCESS);
      assert(MPI_Comm_rank(comm, &srank) == MPI_SUCCESS);
      size_t size, rank;
      size = (size_t)ssize;
      rank = (size_t)srank;
        
      assert(H5Pclose(fapl) >= 0);

      size_t num_dest = prj_edge_map.size();
      size_t num_blocks = num_dest > 0 ? 1 : 0;
      if (rank == size-1)
        {
          num_blocks++;
        }
        
      // create relative destination pointers and source index
      vector<DST_BLK_PTR_T> dst_blk_ptr; 
      vector<DST_PTR_T> dst_ptr;
      vector<NODE_IDX_T> dst_blk_idx, src_idx;
      NODE_IDX_T first_idx = 0, last_idx = 0;
      hsize_t num_block_edges = 0, num_prj_edges = 0;
      if (!prj_edge_map.empty())
        {
          first_idx = (prj_edge_map.begin())->first;
          last_idx  = first_idx;
          dst_blk_idx.push_back(first_idx - dst_start);
          dst_blk_ptr.push_back(0);
          for (auto iter = prj_edge_map.begin(); iter != prj_edge_map.end(); ++iter)
            {
              NODE_IDX_T dst = iter->first;
              edge_tuple_t et = iter->second;
              vector<NODE_IDX_T> &v = get<0>(et);
              
              // creates new block if non-contiguous dst indices
              if (((dst-1) > last_idx) || (num_block_edges > block_size))
                {
                  dst_blk_idx.push_back(dst - dst_start);
                  dst_blk_ptr.push_back(dst_ptr.size());
                  num_blocks++;
                  num_block_edges = 0;
                }
              last_idx = dst;
              
              copy(v.begin(), v.end(), back_inserter(src_idx));
              dst_ptr.push_back(num_prj_edges);
              num_prj_edges += v.size();
              num_block_edges += v.size();
            }
        }
      dst_ptr.push_back(num_prj_edges);
      assert(num_edges == src_idx.size());


      // exchange allocation data

      vector<size_t> sendbuf_num_blocks(size, num_blocks);
      vector<size_t> recvbuf_num_blocks(size);
      assert(MPI_Allgather(&sendbuf_num_blocks[0], 1, MPI_SIZE_T,
                           &recvbuf_num_blocks[0], 1, MPI_SIZE_T, comm)
             == MPI_SUCCESS);

      vector<size_t> sendbuf_num_dest(size, num_dest);
      vector<size_t> recvbuf_num_dest(size);
      assert(MPI_Allgather(&sendbuf_num_dest[0], 1, MPI_SIZE_T,
                           &recvbuf_num_dest[0], 1, MPI_SIZE_T, comm)
             == MPI_SUCCESS);

      vector<size_t> sendbuf_num_edge(size, num_edges);
      vector<size_t> recvbuf_num_edge(size);
      assert(MPI_Allgather(&sendbuf_num_edge[0], 1, MPI_SIZE_T,
                           &recvbuf_num_edge[0], 1, MPI_SIZE_T, comm)
             == MPI_SUCCESS);

      // determine last rank that has data
      size_t last_rank = size-1;
      if (size > 1)
        {
          for (size_t p=1; p<size; p++)
            {
              if (recvbuf_num_blocks[p] == 0)
                {
                  last_rank = p-1;
                  break;
                }
            }
        }

      hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
      assert(lcpl >= 0);
      assert(H5Pset_create_intermediate_group(lcpl, 1) >= 0);

      size_t total_num_blocks=0;
      for (size_t p=0; p<size; p++)
        {
          total_num_blocks = total_num_blocks + recvbuf_num_blocks[p];
        }

      size_t total_num_dests=0;
      for (size_t p=0; p<size; p++)
        {
          total_num_dests = total_num_dests + recvbuf_num_dest[p];
        }

      size_t total_num_edges=0;
      for (size_t p=0; p<size; p++)
        {
          total_num_edges = total_num_edges + recvbuf_num_edge[p];
        }
        
      string path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_BLK_IDX);
      hsize_t dst_blk_idx_dims    = total_num_blocks, one=1;

      hid_t dset = H5Dopen2 (file, path.c_str(), H5P_DEFAULT);
      assert(dset >= 0);

      hid_t fspace = H5Dget_space(dset);
      assert(fspace >= 0);

      hsize_t dst_blk_start       = (hsize_t) H5Sget_simple_extent_npoints(fspace);
      hsize_t dst_blk_idx_newsize = dst_blk_start + dst_blk_idx_dims;
      if (dst_blk_idx_newsize > 0)
        {
          herr_t ierr = H5Dset_extent (dset, &dst_blk_idx_newsize);
          assert(ierr >= 0);
        }
      assert(H5Sclose(fspace) >= 0);

      hsize_t block = num_blocks;
      hid_t mspace  = H5Screate_simple(1, &block, &block);
      assert(mspace >= 0);
      assert(H5Sselect_all(mspace) >= 0);
      hsize_t start = dst_blk_start;
      for (size_t p = 0; p < rank; ++p)
        {
          start += recvbuf_num_blocks[p];
        }
        
      fspace = H5Dget_space(dset);
      assert(fspace >= 0);
      assert(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                 &one, &block) >= 0);
      assert(H5Dwrite(dset, NODE_IDX_H5_NATIVE_T, mspace, fspace, H5P_DEFAULT,
                      &dst_blk_idx[0]) >= 0);
      assert(H5Dclose(dset) >= 0);
      assert(H5Sclose(mspace) >= 0);
      assert(H5Sclose(fspace) >= 0);

      /*
        vector<NODE_IDX_T> v_dst_start(1, dst_start);         
        write(file, path, NODE_IDX_H5_FILE_T, v_dst_start);
      */

      if (block > 0)
        {
          for (size_t p = 0; p < rank; ++p)
            {
              dst_blk_ptr[0] += recvbuf_num_dest[p];
            }
          for (size_t i = 1; i < dst_blk_ptr.size(); i++)
            {
              dst_blk_ptr[i] += dst_blk_ptr[0];
            }
          if (rank == last_rank) // last rank writes the total destination count
            {
              dst_blk_ptr.push_back(dst_blk_ptr[0] + recvbuf_num_dest[rank] + 1);
            }
        }

      path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_BLK_PTR);
      hsize_t dst_blk_ptr_dims = (hsize_t)total_num_blocks+1;

      dset = H5Dopen2 (file, path.c_str(), H5P_DEFAULT);
      assert(dset >= 0);

      hsize_t dst_blk_ptr_newsize = dst_blk_start + dst_blk_ptr_dims;
      if (dst_blk_ptr_newsize > 0)
        {
          herr_t ierr = H5Dset_extent (dset, &dst_blk_ptr_newsize);
          assert(ierr >= 0);
        }

      if (rank == last_rank)
        {
          block = num_blocks+1;
        }
      else
        {
          block = num_blocks;
        }

      mspace  = H5Screate_simple(1, &block, &block);
      assert(mspace >= 0);
      assert(H5Sselect_all(mspace) >= 0);

      start = dst_blk_start;
      for (size_t p = 0; p < rank; ++p)
        {
          start += recvbuf_num_blocks[p];
        }

      if (block > 0)
        {
          assert(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                     &one, &block) >= 0);
        }
      else
        {
          assert(H5Sselect_none(fspace) >= 0);
        }
      fspace = H5Dget_space(dset);
      assert(fspace >= 0);
      assert(H5Dwrite(dset, DST_BLK_PTR_H5_NATIVE_T, mspace, fspace,
                      H5P_DEFAULT, &dst_blk_ptr[0]) >= 0);

      assert(H5Dclose(dset) >= 0);
      assert(H5Sclose(mspace) >= 0);
      assert(H5Sclose(fspace) >= 0);

      /*
	if (rank == 0)
        {
        DEBUG("writing dbp\n");
        }

        write(file, path, DST_BLK_PTR_H5_FILE_T, dbp);
      */
        

      // write destination pointers
      // # dest. pointers = number of destinations + 1
      size_t s = 0;
      for (size_t p = 0; p < rank; ++p)
        {
          s += recvbuf_num_edge[p];
        }

      for (size_t idst = 0; idst < dst_ptr.size(); ++idst)
        {
          dst_ptr[idst] += s;
        }

      if (rank != last_rank) // only the last rank writes an additional element
        {
          dst_ptr.resize(num_dest);
        }
      path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::DST_PTR);
      hsize_t dst_ptr_dims = total_num_dests+1;

      dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
      assert(dset >= 0);

      fspace = H5Dget_space(dset);
      assert(fspace >= 0);
      hsize_t dst_ptr_start = (hsize_t) H5Sget_simple_extent_npoints(fspace)-1;
      assert(H5Sclose(fspace) >= 0);

      hsize_t dst_ptr_newsize = dst_ptr_start + dst_ptr_dims;
      if (dst_ptr_newsize > 0)
        {
          herr_t ierr = H5Dset_extent (dset, &dst_ptr_newsize);
          assert(ierr >= 0);
        }

      block = (hsize_t) dst_ptr.size();
      mspace = H5Screate_simple(1, &block, &block);
      assert(mspace >= 0);
      assert(H5Sselect_all(mspace) >= 0);
      
      fspace = H5Dget_space(dset);
      assert(fspace >= 0);
      if (block > 0)
        {
          start = (hsize_t)dst_blk_ptr[0] + dst_ptr_start;
          assert(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                     &one, &block) >= 0);
        }
      else
        {
          assert(H5Sselect_none(fspace) >= 0);
        }

      assert(H5Dwrite(dset, DST_PTR_H5_NATIVE_T, mspace, fspace,
                      H5P_DEFAULT, &dst_ptr[0]) >= 0);

      assert(H5Dclose(dset) >= 0);
      assert(H5Sclose(mspace) >= 0);
      assert(H5Sclose(fspace) >= 0);

      /*
	if (rank == 0)
        {
        DEBUG("writing dst_ptr\n");
        }
        write(file, path, DST_PTR_H5_FILE_T, dst_ptr);
      */

      // write source index
      // # source indexes = number of edges

      path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name, hdf5::EDGES, hdf5::SRC_IDX);
      hsize_t src_idx_dims = total_num_edges;

      dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
      assert(dset >= 0);

      fspace = H5Dget_space(dset);
      hsize_t src_idx_start = (hsize_t) H5Sget_simple_extent_npoints(fspace);
      assert(H5Sclose(fspace) >= 0);

      hsize_t src_idx_newsize = src_idx_start + src_idx_dims;
      if (src_idx_newsize > 0)
        {
          herr_t ierr = H5Dset_extent (dset, &src_idx_newsize);
          assert(ierr >= 0);
        }

      block = (hsize_t) src_idx.size();
      mspace = H5Screate_simple(1, &block, &block);
      assert(mspace >= 0);
      assert(H5Sselect_all(mspace) >= 0);
      
      fspace = H5Dget_space(dset);
      assert(fspace >= 0);
      if (block > 0)
        {
          start = (hsize_t)dst_ptr[0] + src_idx_start;
          assert(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                     &one, &block) >= 0);
        }
      else
        {
          assert(H5Sselect_none(fspace) >= 0);
        }
      assert(H5Dwrite(dset, NODE_IDX_H5_NATIVE_T, mspace, fspace,
                      H5P_DEFAULT, &src_idx[0]) >= 0);

      assert(H5Dclose(dset) >= 0);
      assert(H5Sclose(mspace) >= 0);
      assert(H5Sclose(fspace) >= 0);

      /*
        write(file, path, NODE_IDX_H5_FILE_T, src_idx);
      */

      vector <string> edge_attr_name_spaces;
      map <string, data::NamedAttrVal> edge_attr_map;

      for (auto const& iter : edge_attr_names)
        {
          const string & attr_namespace = iter.first;
          const vector <vector <string> >& attr_names = iter.second;

          data::NamedAttrVal& edge_attr_values = edge_attr_map[attr_namespace];
          
          edge_attr_values.float_values.resize(attr_names[data::AttrVal::attr_index_float].size());
          edge_attr_values.uint8_values.resize(attr_names[data::AttrVal::attr_index_uint8].size());
          edge_attr_values.uint16_values.resize(attr_names[data::AttrVal::attr_index_uint16].size());
          edge_attr_values.uint32_values.resize(attr_names[data::AttrVal::attr_index_uint32].size());
          edge_attr_values.int8_values.resize(attr_names[data::AttrVal::attr_index_int8].size());
          edge_attr_values.int16_values.resize(attr_names[data::AttrVal::attr_index_int16].size());
          edge_attr_values.int32_values.resize(attr_names[data::AttrVal::attr_index_int32].size());

          edge_attr_name_spaces.push_back(attr_namespace);
        }
        
      for (auto iter = prj_edge_map.cbegin(); iter != prj_edge_map.cend(); ++iter)
        {
          const edge_tuple_t& et = iter->second;
          const vector<NODE_IDX_T>& v = get<0>(et);
          const vector <data::AttrVal>& a = get<1>(et);
          if (v.size() > 0)
            {
              size_t ni=0;
              for (auto const& attr_values : a)
                {
                  const string & attr_namespace = edge_attr_name_spaces[ni];
                  edge_attr_map[attr_namespace].append(attr_values);
                  ni++;
                }
            }
        }

      append_edge_attribute_map<float>(file, src_pop_name, dst_pop_name,
                                       edge_attr_map, edge_attr_names);
      append_edge_attribute_map<uint8_t>(file, src_pop_name, dst_pop_name,
                                         edge_attr_map, edge_attr_names);
      append_edge_attribute_map<uint16_t>(file, src_pop_name, dst_pop_name,
                                          edge_attr_map, edge_attr_names);
      append_edge_attribute_map<uint32_t>(file, src_pop_name, dst_pop_name,
                                          edge_attr_map, edge_attr_names);
      append_edge_attribute_map<int8_t>(file, src_pop_name, dst_pop_name,
                                        edge_attr_map, edge_attr_names);
      append_edge_attribute_map<int16_t>(file, src_pop_name, dst_pop_name,
                                         edge_attr_map, edge_attr_names);
      append_edge_attribute_map<int32_t>(file, src_pop_name, dst_pop_name,
                                         edge_attr_map, edge_attr_names);
        
        
      // clean-up
      assert(H5Pclose(lcpl) >= 0);
    }
  }
}
