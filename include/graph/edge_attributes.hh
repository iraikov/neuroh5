#ifndef EDGE_ATTRIBUTES_HH
#define EDGE_ATTRIBUTES_HH

#include "neuroh5_types.hh"

#include "infer_datatype.hh"
#include "attr_val.hh"
#include "hdf5_edge_attributes.hh"

#include "hdf5.h"
#include "mpi.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

namespace neuroh5
{
  namespace graph
  {
    /// @brief Discovers the list of edge attributes.
    ///
    /// @param file_name      Input file name
    ///
    /// @param src_pop_name   The name of the source population
    ///
    /// @param dst_pop_name   The name of the destination population
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
     const std::string&                           src_pop_name,
     const std::string&                           dst_pop_name,
     const string&                                name_space,
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
     std:: vector <size_t> &num_attrs
     );

    /// @brief Reads the values of edge attributes.
    ///
    /// @param file_name      Input file name
    ///
    /// @param src_pop_name   The name of the source population.
    ///
    /// @param dst_pop_name   The name of the destination population.
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
     const std::string&    src_pop_name,
     const std::string&    dst_pop_name,
     const std::string&    name_space,
     const std::string&    attr_name,
     const DST_PTR_T       edge_base,
     const DST_PTR_T       edge_count,
     const hid_t           attr_h5type,
     data::NamedAttrVal&   attr_values
     );

    extern int read_all_edge_attributes
    (
     MPI_Comm                                           comm,
     const std::string&                                 file_name,
     const std::string&                                 src_pop_name,
     const std::string&                                 dst_pop_name,
     const std::string&                                 name_space,
     const DST_PTR_T                                    edge_base,
     const DST_PTR_T                                    edge_count,
     const std::vector< std::pair<std::string,hid_t> >& edge_attr_info,
     data::NamedAttrVal&                              edge_attr_values
     );

    herr_t read_edge_attributes_serial
    (
     const std::string&    file_name,
     const std::string&    src_pop_name,
     const std::string&    dst_pop_name,
     const std::string&    name_space,
     const std::string&    attr_name,
     const DST_PTR_T       edge_base,
     const DST_PTR_T       edge_count,
     const hid_t           attr_h5type,
     data::NamedAttrVal&   attr_values
     );

    extern int read_all_edge_attributes_serial
    (
     const std::string&                                 file_name,
     const std::string&                                 src_pop_name,
     const std::string&                                 dst_pop_name,
     const std::string&                                 name_space,
     const DST_PTR_T                                    edge_base,
     const DST_PTR_T                                    edge_count,
     const std::vector< std::pair<std::string,hid_t> >& edge_attr_info,
     data::NamedAttrVal&                              edge_attr_values
     );

    
    template <typename T>
    herr_t write_edge_attribute
    (
     hid_t                    loc,
     const std::string&       path,
     const std::vector<T>&    value
     )
    {
      // get a file handle and retrieve the MPI info
      hid_t file = H5Iget_file_id(loc);
      assert(file >= 0);

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

      uint32_t my_count = (uint32_t)value.size();
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


      hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
      assert(lcpl >= 0);
      assert(H5Pset_create_intermediate_group(lcpl, 1) >= 0);

      hid_t dset = H5Dcreate(loc, path.c_str(), ftype, fspace,
                             lcpl, H5P_DEFAULT, H5P_DEFAULT);
      assert(dset >= 0);
      assert(H5Dwrite(dset, mtype, mspace, fspace, H5P_DEFAULT, &value[0])
             >= 0);

      assert(H5Dclose(dset) >= 0);
      assert(H5Tclose(mtype) >= 0);
      assert(H5Sclose(fspace) >= 0);
      assert(H5Sclose(mspace) >= 0);
      assert(H5Pclose(lcpl) >= 0);

      assert(MPI_Comm_free(&comm) == MPI_SUCCESS);

      return 0;
    }
    
    template <typename T>
    herr_t append_edge_attribute
    (
     hid_t                    loc,
     const string&            src_pop_name,
     const string&            dst_pop_name,
     const std::string&       attr_namespace,
     const std::string&       attr_name,
     const std::vector<T>&    value,
     const size_t chunk_size = 4000
     )
    {
      // get a file handle and retrieve the MPI info
      hid_t file = H5Iget_file_id(loc);
      assert(file >= 0);

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

      T dummy;
      hid_t ftype = infer_datatype(dummy);
      assert(ftype >= 0);
      hid_t mtype = H5Tget_native_type(ftype, H5T_DIR_ASCEND);
      assert(mtype >= 0);

      string attr_prefix = hdf5::edge_attribute_prefix(src_pop_name,
                                                       dst_pop_name,
                                                       attr_namespace);
      string attr_path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name,
                                                   attr_namespace, attr_name);
      
      if (!(H5Lexists (file, attr_prefix.c_str(), H5P_DEFAULT) > 0) ||
          !(H5Lexists (file, attr_path.c_str(), H5P_DEFAULT) > 0))
        {
          hdf5::create_edge_attribute_datasets(file, src_pop_name, dst_pop_name,
                                               attr_namespace, attr_name,
                                               ftype, chunk_size);
        }

      hdf5::append_edge_attribute<T>(file, src_pop_name, dst_pop_name,
                                     attr_namespace, attr_name,
                                     value);
      assert(MPI_Comm_free(&comm) >= 0);
      return 0;
    }

    template <class T>
    void append_edge_attribute_map (hid_t file,
                                    const string &src_pop_name,
                                    const string &dst_pop_name,
                                    const map <string, data::NamedAttrVal>& edge_attr_map,
                                    const map <string, vector < vector <string> > >& edge_attr_names)
    {
      for (auto const& iter : edge_attr_map)
        {
          const string& attr_namespace = iter.first;
          const data::NamedAttrVal& edge_attr_values = iter.second;
          
          for (size_t i=0; i<edge_attr_values.size_attr_vec<T>(); i++)
            {
              const string& attr_name = edge_attr_names.at(attr_namespace)[data::AttrVal::attr_type_index<T>()][i];
              graph::append_edge_attribute<T>(file, src_pop_name, dst_pop_name, attr_namespace, attr_name, edge_attr_values.attr_vec<T>(i));
            }
        }
    }

  }
}

#endif
