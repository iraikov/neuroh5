
#include <mpi.h>

#include <cassert>
#include <vector>
#include <map>

#include "neurotrees_types.hh"
#include "attrmap.hh"
#include "pack_tree.hh"

using namespace std;

#define MAX_ATTR_NAME_LEN 128

namespace neuroio
{

  namespace mpi
  {
    
    /***************************************************************************
     * Prepare MPI packed data structures with attributes for a given tree.
     **************************************************************************/

    void pack_size_gid
    (
     MPI_Comm comm,
     int &sendsize
     )
    {
      int packsize=0;
      assert(MPI_Pack_size(1, MPI_CELL_IDX_T, comm, &packsize) == MPI_SUCCESS);

      sendsize += packsize;
    }

    void pack_size_attr_values
    (
     MPI_Comm comm,
     const MPI_Datatype mpi_type,
     const size_t num_elems,
     int &sendsize
     )
    {
      int packsize=0;
      if (num_elems > 0)
        {
          assert(MPI_Pack_size(num_elems, mpi_type, comm, &packsize) == MPI_SUCCESS);
          sendsize += packsize;
        }
    }
  
    void pack_gid
    (
     MPI_Comm comm,
     const CELL_IDX_T gid,
     const int &sendbuf_size,
     vector<uint8_t> &sendbuf,
     int &sendpos
     )
    {
      assert(MPI_Pack(&gid, 1, MPI_CELL_IDX_T, &sendbuf[0], sendbuf_size, &sendpos, comm)
             == MPI_SUCCESS);
    }
  
    int pack_tree
    (
     MPI_Comm comm,
     const CELL_IDX_T &gid,
     const neurotree_t &tree,
     int &sendpos,
     vector<uint8_t> &sendbuf
     )
    {
      int ierr = 0;
      int packsize=0, sendsize = 0;

      int rank, size;
      assert(MPI_Comm_size(comm, &size) >= 0);
      assert(MPI_Comm_rank(comm, &rank) >= 0);

      const vector<SECTION_IDX_T> &src_vector = get<1>(tree);
      const vector<SECTION_IDX_T> &dst_vector = get<2>(tree);
      const vector<SECTION_IDX_T> &sections = get<3>(tree);
      const vector<COORD_T> &xcoords = get<4>(tree);
      const vector<COORD_T> &ycoords = get<5>(tree);
      const vector<COORD_T> &zcoords = get<6>(tree);
      const vector<REALVAL_T> &radiuses = get<7>(tree);
      const vector<LAYER_IDX_T> &layers = get<8>(tree);
      const vector<PARENT_NODE_IDX_T> &parents = get<9>(tree);
      const vector<SWC_TYPE_T> &swc_types = get<10>(tree);

      // gid size
      pack_size_gid(comm, sendsize);
      // topology, section, attr size
      assert(MPI_Pack_size(3, MPI_UINT32_T, comm, &packsize) == MPI_SUCCESS);
      sendsize += packsize;
      // topology vector sizes
      pack_size_attr_values(comm, MPI_SECTION_IDX_T, src_vector.size(), sendsize);
      pack_size_attr_values(comm, MPI_SECTION_IDX_T, dst_vector.size(), sendsize);
      pack_size_attr_values(comm, MPI_SECTION_IDX_T, sections.size(), sendsize);
      // coordinate sizes 
      pack_size_attr_values(comm, MPI_COORD_T, xcoords.size(), sendsize);
      pack_size_attr_values(comm, MPI_COORD_T, ycoords.size(), sendsize);
      pack_size_attr_values(comm, MPI_COORD_T, zcoords.size(), sendsize);
      // radius sizes
      pack_size_attr_values(comm, MPI_REALVAL_T, radiuses.size(), sendsize);
      // layer size
      pack_size_attr_values(comm, MPI_LAYER_IDX_T, layers.size(), sendsize);
      // parent node size
      pack_size_attr_values(comm, MPI_PARENT_NODE_IDX_T, parents.size(), sendsize);
      // SWC type size
      pack_size_attr_values(comm, MPI_SWC_TYPE_T, swc_types.size(), sendsize);
      
      sendbuf.resize(sendbuf.size() + sendsize);

      int sendbuf_size = sendbuf.size();

      // Create MPI_PACKED object with all the tree data
      pack_gid(comm, gid, sendbuf_size, sendbuf, sendpos);
      vector<uint32_t> data_sizes;
      data_sizes.push_back(src_vector.size());
      data_sizes.push_back(sections.size());
      data_sizes.push_back(xcoords.size());
      assert(MPI_Pack(&data_sizes[0], data_sizes.size(), MPI_UINT32_T,
                      &sendbuf[0], sendbuf_size, &sendpos, comm) == MPI_SUCCESS);
      pack_attr_values<SECTION_IDX_T>(comm, MPI_SECTION_IDX_T, src_vector, sendbuf_size, sendbuf, sendpos);
      pack_attr_values<SECTION_IDX_T>(comm, MPI_SECTION_IDX_T, dst_vector, sendbuf_size, sendbuf, sendpos);
      pack_attr_values<SECTION_IDX_T>(comm, MPI_SECTION_IDX_T, sections, sendbuf_size, sendbuf, sendpos);
      // coordinate 
      pack_attr_values<COORD_T>(comm, MPI_COORD_T, xcoords, sendbuf_size, sendbuf, sendpos);
      pack_attr_values<COORD_T>(comm, MPI_COORD_T, ycoords, sendbuf_size, sendbuf, sendpos);
      pack_attr_values<COORD_T>(comm, MPI_COORD_T, zcoords, sendbuf_size, sendbuf, sendpos);
      // radius
      pack_attr_values<REALVAL_T>(comm, MPI_REALVAL_T, radiuses, sendbuf_size, sendbuf, sendpos);
      // layer
      pack_attr_values<LAYER_IDX_T>(comm, MPI_LAYER_IDX_T, layers, sendbuf_size, sendbuf, sendpos);
      // parent node
      pack_attr_values<PARENT_NODE_IDX_T>(comm, MPI_PARENT_NODE_IDX_T, parents, sendbuf_size, sendbuf, sendpos);
      // SWC type
      pack_attr_values<SWC_TYPE_T>(comm, MPI_SWC_TYPE_T, swc_types, sendbuf_size, sendbuf, sendpos);

      
      return ierr;
    }

    void unpack_gid
    (
     MPI_Comm comm,
     CELL_IDX_T &gid,
     const size_t &recvbuf_size,
     const vector<uint8_t> &recvbuf,
     int &recvpos
     )
    {
      assert(MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos,
                        &gid, 1, MPI_CELL_IDX_T, comm) == MPI_SUCCESS);
    }

    int unpack_tree
    (
     MPI_Comm comm,
     const vector<uint8_t> &recvbuf,
     int &recvpos,
     map<CELL_IDX_T, neurotree_t> &tree_map
     )
    {
      int ierr = 0;
      
      CELL_IDX_T gid;
      vector<SECTION_IDX_T> src_vector;
      vector<SECTION_IDX_T> dst_vector;
      vector<SECTION_IDX_T> sections;
      vector<COORD_T> xcoords;
      vector<COORD_T> ycoords;
      vector<COORD_T> zcoords;
      vector<REALVAL_T> radiuses;
      vector<LAYER_IDX_T> layers;
      vector<PARENT_NODE_IDX_T> parents;
      vector<SWC_TYPE_T> swc_types;

      size_t recvbuf_size = recvbuf.size();
      
      unpack_gid(comm, gid, recvbuf_size, recvbuf, recvpos);
      vector<uint32_t> data_sizes(3);
      assert(MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos,
                        &data_sizes[0], 3, MPI_UINT32_T, comm) == MPI_SUCCESS);

      unpack_attr_values<SECTION_IDX_T>(comm, MPI_SECTION_IDX_T, data_sizes[0], src_vector, recvbuf_size, recvbuf, recvpos);
      unpack_attr_values<SECTION_IDX_T>(comm, MPI_SECTION_IDX_T, data_sizes[0], dst_vector, recvbuf_size, recvbuf, recvpos);
      unpack_attr_values<SECTION_IDX_T>(comm, MPI_SECTION_IDX_T, data_sizes[1], sections, recvbuf_size, recvbuf, recvpos);
      // coordinate 
      unpack_attr_values<COORD_T>(comm, MPI_COORD_T, data_sizes[2], xcoords, recvbuf_size, recvbuf, recvpos);
      unpack_attr_values<COORD_T>(comm, MPI_COORD_T, data_sizes[2], ycoords, recvbuf_size, recvbuf, recvpos);
      unpack_attr_values<COORD_T>(comm, MPI_COORD_T, data_sizes[2], zcoords, recvbuf_size, recvbuf, recvpos);
      // radius
      unpack_attr_values<REALVAL_T>(comm, MPI_REALVAL_T, data_sizes[2], radiuses, recvbuf_size, recvbuf, recvpos);
      // layer
      unpack_attr_values<LAYER_IDX_T>(comm, MPI_LAYER_IDX_T, data_sizes[2], layers, recvbuf_size, recvbuf, recvpos);
      // parent node
      unpack_attr_values<PARENT_NODE_IDX_T>(comm, MPI_PARENT_NODE_IDX_T, data_sizes[2], parents, recvbuf_size, recvbuf, recvpos);
      // SWC type
      unpack_attr_values<SWC_TYPE_T>(comm, MPI_SWC_TYPE_T, data_sizes[2], swc_types, recvbuf_size, recvbuf, recvpos);

      neurotree_t tree = make_tuple(gid,src_vector,dst_vector,sections,
                                    xcoords,ycoords,zcoords,
                                    radiuses,layers,parents,
                                    swc_types);

      tree_map.insert(make_pair(gid, tree));
      
      return ierr;
    }
  }
}
