#ifndef APPEND_PROJECTION_HH
#define APPEND_PROJECTION_HH

#include <string>
#include <vector>
#include <map>
#include <hdf5.h>

#include "neuroh5_types.hh"

namespace neuroh5
{
  namespace graph
  {
    void append_projection
    (
     MPI_Comm                  comm,
     hid_t                     file,
     const std::string&        src_pop_name,
     const std::string&        dst_pop_name,
     const NODE_IDX_T&         src_start,
     const NODE_IDX_T&         src_end,
     const NODE_IDX_T&         dst_start,
     const NODE_IDX_T&         dst_end,
     const size_t&             num_edges,
     const edge_map_t&         prj_edge_map,
     const std::map <std::string, std::pair <size_t, data::AttrIndex > >& edge_attr_index,
     const hsize_t            cdim = 4096,
     const hsize_t            block_size = 1000000,
     const bool collective = true
     );


  }
}

#endif
