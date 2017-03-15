#ifndef CONTEXTMNGR_HH
#define CONTEXTMNGR_HH

#include "hdf5.h"

#include <vector>

namespace ngh5
{
  namespace model
  {
    class ContextMngr
    {
    public:

      static void initialize();

      template<typename T>
      static void node_scatter
      (
       const std::vector<T>& io_buf,
       std::vector<T>&       all_buf,
       size_t item_count = 1
       );

      template<typename T>
      static void node_gather
      (
       const std::vector<T>& all_buf,
       std::vector<T>&       io_buf,
       size_t item_count = 1
       );

      template<typename T>
      static void edge_scatter
      (
       const std::vector<T>& io_buf,
       std::vector<T>&       all_buf,
       size_t item_count = 1
       );

      template<typename T>
      static void edge_gather
      (
       const std::vector<T>& all_buf,
       std::vector<T>&       io_buf,
       size_t item_count = 1
       );

      static hid_t get_hdf5_file_handle();

      static void finalize();

    private:

      static bool m_initialized;

      // MPI stuff
      static MPI_Comm m_io_comm;
      static MPI_Comm m_all_comm;

      // HDF5 stuff
      static hid_t m_ngh5_file;

      static hid_t m_lcpl_interm_grps;

      // I don't think we need to store the partitionings/ownership info
      // explicitely. (?)

      // L1 -> L2 scatter
      // Isn't the reverse scatter just send <-> recv?

      std::vector<int> m_node_sendcounts;
      std::vector<int> m_node_sdispls;
      std::vector<int> m_node_recvcounts;
      std::vector<int> m_node_displs;

      std::vector<int> m_edge_sendcounts;
      std::vector<int> m_edge_sdispls;
      std::vector<int> m_edge_recvcounts;
      std::vector<int> m_edge_displs;

      ContextMngr() {};
      ~ContextMngr() {};

    };
  }
}




#endif
