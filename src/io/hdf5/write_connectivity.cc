
#include "write_connectivity.hh"

#include <algorithm>
#include <cassert>
#include <map>

using namespace std;

namespace ngh5
{
  namespace io
  {
    namespace hdf5
    {
      void write_connectivity
      (
       hid_t                     file,
       const NODE_IDX_T&         first_node,
       const NODE_IDX_T&         last_node,
       const vector<NODE_IDX_T>& edges
       )
      {
        // do a sanity check on the input
        assert(first_node < last_node);
        assert(edges.size()%2 == 0);

        uint64_t num_edges = edges.size()/2;
        assert(num_edges > 0);

        // build destination->source(s) map as a side-effect
        map<NODE_IDX_T, vector<NODE_IDX_T> > dst_src_map;
        for (NODE_IDX_T inode = first_node; inode < last_node; ++inode)
          {
            dst_src_map.insert(make_pair(inode, vector<NODE_IDX_T>()));
          }

        map<NODE_IDX_T, vector<NODE_IDX_T> >::iterator iter;

        for (size_t i = 1; i < edges.size(); i += 2)
          {
            // all destination node IDs must be in range
            assert(first_node <= edges[i] && edges[i] <= last_node);

            iter = dst_src_map.find(edges[i]);
            assert (iter != dst_src_map.end());
            iter->second.push_back(edges[i-1]);
          }

        uint64_t num_dest = dst_src_map.size();
        assert(num_dest > 0 && num_dest < (last_node - first_node + 1));

        // sort the source arrays and create relative destination pointers
        // and source index
        vector<uint64_t> dst_ptr(1, 0);
        vector<uint32_t> src_idx;
        size_t pos = 0;
        for (iter = dst_src_map.begin(); iter != dst_src_map.end(); ++iter)
          {
            sort(iter->second.begin(), iter->second.end());
            dst_ptr.push_back(dst_ptr[pos++] + iter->second.size());
            copy(iter->second.begin(), iter->second.end(),
                 back_inserter(src_idx));
            // save memory
            iter->second.clear();
          }

        // get the I/O communicator
        MPI_Comm comm;
        MPI_Info info;
        hid_t fapl = H5Fget_access_plist(file);
        assert(H5Pget_fapl_mpio(fapl, &comm, &info) >= 0);

        int size, rank;
        assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS);
        assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS);

        assert(H5Pclose(fapl) >= 0);

        // exchange allocation data

        vector<uint64_t> sendbuf_num_dest(size, num_dest);
        vector<uint64_t> recvbuf_num_dest(size);
        assert(MPI_Allgather(&sendbuf_num_dest[0], 1, MPI_UINT64_T,
                             &recvbuf_num_dest[0], 1, MPI_UINT64_T, comm)
               == MPI_SUCCESS);

        vector<uint64_t> sendbuf_num_edge(size, num_edges);
        vector<uint64_t> recvbuf_num_edge(size);
        assert(MPI_Allgather(&sendbuf_num_edge[0], 1, MPI_UINT64_T,
                             &recvbuf_num_edge[0], 1, MPI_UINT64_T, comm)
               == MPI_SUCCESS);

        hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
        assert(lcpl >= 0);
        assert(H5Pset_create_intermediate_group(lcpl, 1) >= 0);

        // write destination block index (= first_node)

        string path = "/Connectivity/Destination Block Index";

        // write destination block pointer
        // TODO: correct the destination block pointers by their global offset

        path = "/Connectivity/Destination Block Pointer";

        // write destination pointers
        // # dest. pointers = number of destinations
        // TODO: correct the destination pointers by their global offset

        path = "/Connectivity/Destination Pointer";

        // write source index
        // # source indexes = number of edges

        path = "/Connectivity/Source Index";


        // clean-up
        assert(H5Pclose(lcpl) >= 0);

      }
    }
  }
}
