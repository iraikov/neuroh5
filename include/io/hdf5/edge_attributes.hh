#ifndef EDGE_ATTRIBUTES_HH
#define EDGE_ATTRIBUTES_HH

#include "infer_datatype.hh"
#include "ngh5_types.hh"
#include "edge_attr.hh"

#include "hdf5.h"
#include "mpi.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

namespace ngh5
{
  namespace io
  {
    namespace hdf5
    {
      /// @brief Discovers the list of edge attributes.
      ///
      /// @param file_name      Input file name
      ///
      /// @param proj_name      The (abbreviated) name of the projection data
      ///                       set.
      ///
      /// @param out_attributes    A vector of pairs, one for each edge attribute
      ///                          discovered. The pairs contain the attribute
      ///                          name and the attributes HDF5 file datatype.
      ///                          NOTE: The datatype handles MUST be closed by
      ///                          the caller (via H5Tclose).
      ///
      /// @return                  HDF5 error code.
      herr_t get_edge_attributes
      (
       const std::string&                           file_name,
       const std::string&                           proj_name,
       std::vector< std::pair<std::string,hid_t> >& out_attributes
       );

      /// @brief Determines the number of edge attributes for each supported
      //         type.
      ///
      ///
      /// @param attributes    A vector of pairs, one for each edge attribute
      ///                      discovered. The pairs contain the attribute name
      ///                      and the attributes HDF5 file datatype.
      ///
      /// @param num_attrs     A vector which indicates the number of attributes
      ///                      of each type.
      ///                      - Index 0 float type
      ///                      - Index 1: uint8/enum type
      ///                      - Index 1: uint16 type
      ///                      - Index 1: uint32 type
      ///
      /// @return                  HDF5 error code.
      herr_t num_edge_attributes
      (
       const std::vector< std::pair<std::string,hid_t> >& attributes,
       std:: vector <uint32_t> &num_attrs
       );

      /// @brief Reads the values of edge attributes.
      ///
      /// @param file_name      Input file name
      ///
      /// @param proj_name      The (abbreviated) name of the projection.
      ///
      /// @param attr_name      The name of the attribute.
      ///
      /// @param edge_base      Edge offset (returned by read_dbs_projection).
      ///
      /// @param edge_count     Edge count.
      ///
      /// @param attr_h5type    The HDF5 type of the attribute.
      ///
      /// @param attr_values    An EdgeNamedAttr object that holds attribute
      //                        values.
      herr_t read_edge_attributes
      (
       MPI_Comm              comm,
       const std::string&    file_name,
       const std::string&    proj_name,
       const std::string&    attr_name,
       const DST_PTR_T       edge_base,
       const DST_PTR_T       edge_count,
       const hid_t           attr_h5type,
       model::EdgeNamedAttr& attr_values
       );

      extern int read_all_edge_attributes
      (
       MPI_Comm                                           comm,
       const std::string&                                 file_name,
       const std::string&                                 prj_name,
       const DST_PTR_T                                    edge_base,
       const DST_PTR_T                                    edge_count,
       const std::vector< std::pair<std::string,hid_t> >& edge_attr_info,
       model::EdgeNamedAttr&                              edge_attr_values
       );


      template <typename T>
      void write_edge_attribute
      (
       hid_t                    loc,
       const std::string&       path,
       std::vector<NODE_IDX_T>& edge_id,
       std::vector<T>&          value
       )
      {
        assert(edge_id.size() == 2*value.size());

        // get a file handle and retrieve the MPI info
        hid_t file = H5Iget_file_id(loc);
        assert(file >= 0);

        MPI_Comm comm;
        MPI_Info info;
        assert(H5Pget_fapl_mpio(file, &comm, &info) >= 0);

        int size, rank;
        assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS);
        assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS);

        uint32_t my_count = (uint32_t)edge_id.size();
        std::vector<uint32_t> all_counts(size);
        assert(MPI_Allgather(&my_count, 1, MPI_UINT32_T, &all_counts[0], 1,
                             MPI_UINT32_T, comm) == MPI_SUCCESS);

        // calculate the total dataset size and the offset of my piece
        hsize_t start = 0, total = 0, count = 1, block = my_count;
        for (size_t p = 0; p < size; ++p)
          {
            if (p < rank)
              {
                start += (hsize_t) all_counts[p];
              }
            total += (hsize_t) all_counts[p];
          }

        // create dataspaces and selections
        hid_t mspace = H5Screate_simple(1, &block, &block);
        assert(mspace >= 0);
        assert(H5Sselect_all(mspace) >= 0);
        hid_t fspace = H5Screate_simple(1, &total, &total);
        assert(fspace >= 0);
        assert(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                   &count, &block) >= 0);

        // figure the type

        T dummy;
        hid_t ftype = infer_datatype(dummy);
        assert(ftype >= 0);
        hid_t mtype = H5Tget_native_type(ftype, H5T_DIR_ASCEND);
        assert(mtype >= 0);

        // Ready to roll!

        // we write the values first
        // everything needs to be scaled by 2 for the edge IDs

        hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
        assert(lcpl >= 0);
        assert(H5Pset_create_intermediate_group(lcpl, 1) >= 0);

        hid_t dset = H5Dcreate(loc, (path + "/value").c_str(), ftype, fspace,
                               lcpl, H5P_DEFAULT, H5P_DEFAULT);
        assert(dset >= 0);
        assert(H5Dwrite(dset, mtype, mspace, fspace, H5P_DEFAULT, &value[0])
               >= 0);

        assert(H5Dclose(dset) >= 0);
        assert(H5Tclose(mtype) >= 0);
        assert(H5Tclose(ftype) >= 0);
        assert(H5Sclose(fspace) >= 0);
        assert(H5Sclose(mspace) >= 0);

        // scale by factor 2
        block *= 2;
        mspace = H5Screate_simple(1, &block, &block);
        assert(mspace >= 0);
        assert(H5Sselect_all(mspace) >= 0);
        total *= 2;
        fspace = H5Screate_simple(1, &total, &total);
        assert(fspace >= 0);
        start *= 2;
        assert(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                   &count, &block) >= 0);

        dset = H5Dcreate(loc, (path + "/edge_id").c_str(), H5T_STD_U32LE,
                         fspace, lcpl, H5P_DEFAULT, H5P_DEFAULT);
        assert(dset >= 0);
        assert(H5Dwrite(dset, H5T_NATIVE_UINT32, mspace, fspace, H5P_DEFAULT,
                        &edge_id[0]) >= 0);

        assert(H5Dclose(dset) >= 0);
        assert(H5Sclose(fspace) >= 0);
        assert(H5Sclose(mspace) >= 0);
        assert(H5Pclose(lcpl) >= 0);
      }

      template <typename T>
      void read_edge_attribute
      (
       hid_t                    loc,
       const std::string&       path,
       std::vector<NODE_IDX_T>& edge_id,
       std::vector<T>&          value
       )
      {
        // read node IDs

        hid_t dset = H5Dopen(loc, (path + "/edge_id").c_str(), H5P_DEFAULT);
        assert(dset >= 0);
        hid_t fspace = H5Dget_space(dset);
        assert(fspace >= 0);
        hssize_t size = H5Sget_simple_extent_npoints(fspace);
        assert(size > 0);
        edge_id.resize(size);
        assert(H5Dread(dset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                       &edge_id[0]) >= 0);
        assert(H5Sclose(fspace) >= 0);
        assert(H5Dclose(dset) >= 0);

        // read values

        dset = H5Dopen(loc, (path + "/value").c_str(), H5P_DEFAULT);
        assert(dset >= 0);
        fspace = H5Dget_space(dset);
        assert(fspace >= 0);
        size = H5Sget_simple_extent_npoints(fspace);
        assert(size > 0 && 2*size == edge_id.size());
        value.resize(size);

        hid_t ftype = H5Dget_type(dset);
        assert(ftype >= 0);
        hid_t ntype = H5Tget_native_type(ftype, H5T_DIR_ASCEND);
        assert(H5Dread(dset, ntype, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                       &value[0]) >= 0);
        assert(H5Tclose(ntype) >= 0);
        assert(H5Tclose(ftype) >= 0);
        assert(H5Sclose(fspace) >= 0);
        assert(H5Dclose(dset) >= 0);
      }
    }
  }
}

#endif
