#ifndef PACK_TREE_HH
#define PACK_TREE_HH

#include <mpi.h>

#include <vector>
#include <map>

#include "neuroh5_types.hh"
#include "attr_map.hh"

namespace neuroh5
{

  namespace mpi
  {
    void pack_size_gid
    (
     MPI_Comm comm,
     int &sendsize
     );

    void pack_size_attr_values
    (
     MPI_Comm comm,
     const MPI_Datatype mpi_type,
     const size_t num_elems,
     int &sendsize
     );
  
    void pack_gid
    (
     MPI_Comm comm,
     const CELL_IDX_T gid,
     const int &sendbuf_size,
     vector<uint8_t> &sendbuf,
     int &sendpos
     );
  
    int pack_tree
    (
     MPI_Comm comm,
     const CELL_IDX_T &gid,
     const neurotree_t &tree,
     int &sendpos,
     vector<uint8_t> &sendbuf
     );

  

    void unpack_gid
    (
     MPI_Comm comm,
     CELL_IDX_T &gid,
     const size_t &recvbuf_size,
     const vector<uint8_t> &recvbuf,
     int &recvpos
     );

    int unpack_tree
    (
     MPI_Comm comm,
     const vector<uint8_t> &recvbuf,
     int &recvpos,
     map<CELL_IDX_T, neurotree_t> &tree_map
     );


    template <class T>
    void pack_size_attr
    (
     MPI_Comm comm,
     const MPI_Datatype mpi_type,
     const CELL_IDX_T gid,
     const vector< vector<T> > &values,
     int &sendsize
     )
    {
      int packsize=0;
      assert(MPI_Pack_size(1, MPI_UINT8_T, comm, &packsize) == MPI_SUCCESS);
      sendsize+=packsize;
      for (size_t k = 0; k < values.size(); k++)
        {
          size_t num_elems = values[k].size();
          assert(MPI_Pack_size(1, MPI_UINT32_T, comm, &packsize) == MPI_SUCCESS);
          sendsize+=packsize;
          if (num_elems > 0)
            {
              pack_size_attr_values(comm, mpi_type, num_elems, sendsize);
            }
        }
    }
  
    template <class T>
    void pack_attr_values
    (
     MPI_Comm comm,
     const MPI_Datatype mpi_type,
     const vector<T> &values,
     const int &sendbuf_size,
     vector<uint8_t> &sendbuf,
     int &sendpos
     )
    {
      assert(MPI_Pack(&values[0], values.size(), mpi_type, &sendbuf[0], sendbuf_size, &sendpos, comm)
             == MPI_SUCCESS);
    }

    template <class T>
    void pack_attr
    (
     MPI_Comm comm,
     const MPI_Datatype mpi_type,
     const CELL_IDX_T gid,
     const vector< vector<T> > &values,
     const int &sendbuf_size,
     int &sendpos,
     vector<uint8_t> &sendbuf
     )
    {
      int rank;
      assert(MPI_Comm_rank(comm, &rank) >= 0);
      uint8_t num_attrs = values.size();
      assert(MPI_Pack(&num_attrs, 1, MPI_UINT8_T, &sendbuf[0],
                      sendbuf_size, &sendpos, comm) == MPI_SUCCESS);
      if (num_attrs > 0)
        {
          vector<uint32_t> num_elems;
          for (size_t k = 0; k < num_attrs; k++)
            {
              num_elems.push_back(values[k].size());
            }
          assert(MPI_Pack(&num_elems[0], num_elems.size(), MPI_UINT32_T, &sendbuf[0],
                          sendbuf_size, &sendpos, comm) == MPI_SUCCESS);
          for (size_t k = 0; k < num_attrs; k++)
            {
              if (num_elems[k] > 0)
                {
                  pack_attr_values(comm, mpi_type, values[k], sendbuf_size, sendbuf, sendpos);
                }
            }
        }
    }

    


    template <class T>
    void unpack_attr_values
    (
     MPI_Comm comm,
     const MPI_Datatype mpi_type,
     const uint32_t num_elems,
     vector<T> &values,
     const size_t &recvbuf_size,
     const vector<uint8_t> &recvbuf,
     int &recvpos
     )
    {
      int ierr;
      values.resize(num_elems);
      assert(recvpos < (int)recvbuf_size);
      ierr = MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos,
                        &values[0], num_elems, mpi_type, comm);
      assert(ierr == MPI_SUCCESS);
    }

    template <class T>
    void unpack_attr
    (
     MPI_Comm comm,
     const MPI_Datatype mpi_type,
     const CELL_IDX_T gid,
     data::NamedAttrMap& m,
     const size_t &recvbuf_size,
     const vector<uint8_t> &recvbuf,
     int &recvpos
     )
    {
      int rank;
      assert(MPI_Comm_rank(comm, &rank) >= 0);
      uint8_t num_attrs=0;
      assert(MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos,
                        &num_attrs, 1, MPI_UINT8_T, comm) == MPI_SUCCESS);
      if (num_attrs > 0)
        {
          vector<uint32_t> num_elems(num_attrs);
          assert(MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos,
                            &num_elems[0], num_attrs, MPI_UINT32_T, comm) == MPI_SUCCESS);
          for (size_t k = 0; k < num_attrs; k++)
            {
              assert(recvpos < (int)recvbuf_size);
              if (num_elems[k] > 0)
                {
                  vector<T> v(num_elems[k]);
                  unpack_attr_values<T>(comm, mpi_type, num_elems[k], v, recvbuf_size, recvbuf, recvpos);
                  m.insert<T>(k, gid, v);
                }
            }
        }
    }
  }
}

#endif
