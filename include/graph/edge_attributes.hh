#ifndef EDGE_ATTRIBUTES_HH
#define EDGE_ATTRIBUTES_HH

#include "neuroh5_types.hh"
#include "attr_val.hh"
#include "infer_datatype.hh"
#include "attr_kind_datatype.hh"
#include "hdf5_edge_attributes.hh"
#include "exists_dataset.hh"

#include <hdf5.h>
#include <mpi.h>

#include <cstdint>
#include <string>
#include <vector>

namespace neuroh5
{
  namespace graph
  {
    /// @brief Checks if a particular edge attribute namespace exists in the given file.
    ///
    /// @param file_name      Input file name
    ///
    /// @param src_pop_name   The name of the source population
    ///
    /// @param dst_pop_name   The name of the destination population
    ///
    /// @param name_space    Name space.
    ///
    /// @param has_name_space    Boolean flag, set to true if the name space exists.
    ///
    /// @return                  HDF5 error code.
    herr_t has_edge_attribute_namespace
    (
     MPI_Comm                      comm,
     const string&                 file_name,
     const string&                 src_pop_name,
     const string&                 dst_pop_name,
     const string&                 name_space,
     bool &has_namespace
     );

    
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
     MPI_Comm                                     comm,
     const std::string&                           file_name,
     const std::string&                           src_pop_name,
     const std::string&                           dst_pop_name,
     const string&                                name_space,
     std::vector< std::pair<std::string,AttrKind> >& out_attributes
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
    ///
    /// @return                  HDF5 error code.
    herr_t num_edge_attributes
    (
     const std::vector< std::pair<std::string,AttrKind> >& attributes,
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
    extern herr_t read_edge_attributes
    (
     MPI_Comm              comm,
     const std::string&    file_name,
     const std::string&    src_pop_name,
     const std::string&    dst_pop_name,
     const std::string&    name_space,
     const std::string&    attr_name,
     const DST_PTR_T       edge_base,
     const DST_PTR_T       edge_count,
     const AttrKind        attr_kind,
     data::NamedAttrVal&   attr_values,
     bool collective = true
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
     const std::vector< std::pair<std::string,AttrKind> >& edge_attr_info,
     data::NamedAttrVal&                              edge_attr_values
     );
    
    extern herr_t read_edge_attribute_selection
    (
     MPI_Comm              comm,
     const std::string&    file_name,
     const std::string&    src_pop_name,
     const std::string&    dst_pop_name,
     const std::string&    name_space,
     const std::string&    attr_name,
     const DST_PTR_T&      edge_base,
     const DST_PTR_T&      edge_count,
     const vector<NODE_IDX_T>&   selection_dst_idx,
     const vector<DST_PTR_T>&    selection_dst_ptr,
     const vector< pair<hsize_t,hsize_t> >& src_idx_ranges,
     const AttrKind        attr_kind,
     data::NamedAttrVal&   attr_values,
     bool collective = true
     );

    extern int read_all_edge_attribute_selection
    (
     MPI_Comm                    comm,
     const std::string&          file_name,
     const std::string&          src_pop_name,
     const std::string&          dst_pop_name,
     const std::string&          name_space,
     const DST_PTR_T&            edge_base,
     const DST_PTR_T&            edge_count,
     const vector<NODE_IDX_T>&   selection_dst_idx,
     const vector<DST_PTR_T>&    selection_dst_ptr,
     const vector< pair<hsize_t,hsize_t> >& src_idx_ranges,
     const std::vector< std::pair<std::string,AttrKind> >& edge_attr_info,
     data::NamedAttrVal&         edge_attr_values
     );


    
    template <typename T>
    herr_t write_edge_attribute
    (
     MPI_Comm                 comm,
     hid_t                    loc,
     const std::string&       path,
     const std::vector<T>&    value,
     const bool collective = true
     )
    {
      // get a file handle and retrieve the MPI info
      hid_t file = H5Iget_file_id(loc);
      throw_assert(file >= 0, "error in H5Iget_file_id");

      hid_t wapl = H5P_DEFAULT;
      if (collective)
	{
	  wapl = H5Pcreate(H5P_DATASET_XFER);
	  throw_assert(wapl >= 0, "error in H5Pcreate");
	  throw_assert(H5Pset_dxpl_mpio(wapl, H5FD_MPIO_COLLECTIVE) >= 0, 
                       "error in H5Pset_dxpl_mpio");
	}

      int ssize, srank;
      throw_assert(MPI_Comm_size(comm, &ssize) == MPI_SUCCESS, "error in MPI_Comm_size");
      throw_assert(MPI_Comm_rank(comm, &srank) == MPI_SUCCESS, "error in MPI_Comm_rank");
      size_t size, rank;
      size = (size_t)ssize;
      rank = (size_t)srank;

      uint32_t my_count = (uint32_t)value.size();
      std::vector<uint32_t> all_counts(size);
      throw_assert(MPI_Allgather(&my_count, 1, MPI_UINT32_T, &all_counts[0], 1,
                                 MPI_UINT32_T, comm) == MPI_SUCCESS,
                   "error in MPI_Allgather");

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
      throw_assert(mspace >= 0, "error in H5Screate_simple");
      throw_assert(H5Sselect_all(mspace) >= 0, "error in H5Sselect_all");
      hid_t fspace = H5Screate_simple(1, &total, &total);
      throw_assert(fspace >= 0, "error in H5Screate_simple");
      throw_assert(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL,
                                       &count, &block) >= 0,
                   "error in H5Sselect_hyperslab");

      // figure the type

      T dummy;
      hid_t ftype = infer_datatype(dummy);
      throw_assert(ftype >= 0, "error in infer_datatype");
      hid_t mtype = H5Tget_native_type(ftype, H5T_DIR_ASCEND);
      throw_assert(mtype >= 0, "H5Tget_native_type");


      hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
      throw_assert(lcpl >= 0, "error in H5Pcreate");
      throw_assert(H5Pset_create_intermediate_group(lcpl, 1) >= 0, 
                   "error in H5Pset_create_intermediate_group");

      hid_t dset = H5Dcreate(loc, path.c_str(), ftype, fspace,
                             lcpl, H5P_DEFAULT, H5P_DEFAULT);
      throw_assert(dset >= 0, "error in H5Dcreate");
      throw_assert(H5Dwrite(dset, mtype, mspace, fspace, wapl, &value[0])
                   >= 0, "error in H5Dwrite");

      throw_assert(H5Dclose(dset) >= 0, "error in H5Dclose");
      throw_assert(H5Tclose(mtype) >= 0, "error in H5Tclose");
      throw_assert(H5Sclose(fspace) >= 0, "error in H5Sclose");
      throw_assert(H5Sclose(mspace) >= 0, "error in H5Sclose");
      throw_assert(H5Pclose(lcpl) >= 0, "error in H5Pclose");
      throw_assert(H5Pclose(wapl) >= 0, "error in H5Pclose");

      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS, "error in MPI_Barrier");
      throw_assert(H5Fclose (file) >= 0, "error in H5Fclose");

      return 0;
    }

    template <class T>
    void write_edge_attribute_map (MPI_Comm comm,
                                   hid_t file,
                                   const string &src_pop_name,
                                   const string &dst_pop_name,
                                   const map <string, data::NamedAttrVal>& edge_attr_map,
                                   const std::map <std::string, std::pair <size_t, data::AttrIndex > >& edge_attr_index)


    {
      for (auto const& iter : edge_attr_map)
        {
          const string& attr_namespace = iter.first;
          const data::NamedAttrVal& edge_attr_values = iter.second;

          auto ns_it = edge_attr_index.find(attr_namespace);
          if (ns_it != edge_attr_index.end())
            {
              const data::AttrIndex& attr_index = ns_it->second.second;
              const std::vector<std::string>& attr_names = attr_index.attr_names<T>();
              
              for (const std::string& attr_name: attr_names)
                {
                  size_t i = attr_index.attr_index<T>(attr_name);
                  string path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name, attr_namespace, attr_name);
                  graph::write_edge_attribute<T>(comm, file, path, edge_attr_values.attr_vec<T>(i));
                }
            }
          else
            {
              throw std::runtime_error("write_edge_attribute_map: namespace mismatch");
            }
        }
    }
    
    template <typename T>
    herr_t append_edge_attribute
    (
     MPI_Comm                 comm,
     hid_t                    loc,
     const string&            src_pop_name,
     const string&            dst_pop_name,
     const std::string&       attr_namespace,
     const std::string&       attr_name,
     const std::vector<T>&    value,
     const size_t chunk_size = 4000,
     const bool collective = true
     )
    {
      // get a file handle and retrieve the MPI info
      hid_t file = H5Iget_file_id(loc);
      throw_assert(file >= 0, "error in H5Iget_file_id");

      T dummy;
      hid_t ftype = infer_datatype(dummy);
      throw_assert(ftype >= 0, "error in infer_datatype");
      hid_t mtype = H5Tget_native_type(ftype, H5T_DIR_ASCEND);
      throw_assert(mtype >= 0, "error in H5Tget_native_type");

      string attr_prefix = hdf5::edge_attribute_prefix(src_pop_name,
                                                       dst_pop_name,
                                                       attr_namespace);
      string attr_path = hdf5::edge_attribute_path(src_pop_name, dst_pop_name,
                                                   attr_namespace, attr_name);
      
      if (!(hdf5::exists_dataset (file, attr_path) > 0))
        {
          hdf5::create_edge_attribute_datasets(file, src_pop_name, dst_pop_name,
                                               attr_namespace, attr_name,
                                               ftype, chunk_size);
	  throw_assert(MPI_Barrier(comm) == MPI_SUCCESS, "error in MPI_Barrier");
        }

      hdf5::append_edge_attribute<T>(comm, file, src_pop_name, dst_pop_name,
                                     attr_namespace, attr_name,
                                     value);

      throw_assert(MPI_Barrier(comm) == MPI_SUCCESS, "error in MPI_Barrier");
      
      throw_assert(H5Fclose (file) >= 0, "error in H5Fclose");

      return 0;
    }

    template <class T>
    void append_edge_attribute_map (MPI_Comm                 comm,
                                    hid_t file,
                                    const string &src_pop_name,
                                    const string &dst_pop_name,
                                    const map <string, data::NamedAttrVal>& edge_attr_map,
                                    const std::map <std::string, std::pair <size_t, data::AttrIndex > >& edge_attr_index,
				    const size_t chunk_size = 4000)

    {
      for (auto const& iter : edge_attr_map)
        {
          const string& attr_namespace = iter.first;
          const data::NamedAttrVal& edge_attr_values = iter.second;

          auto ns_it = edge_attr_index.find(attr_namespace);
          if (ns_it != edge_attr_index.end())
            {
              const data::AttrIndex& attr_index = ns_it->second.second;
              const std::vector<std::string>& attr_names = attr_index.attr_names<T>();
              
              for (const std::string& attr_name: attr_names)
                {
                  size_t i = attr_index.attr_index<T>(attr_name);
                  graph::append_edge_attribute<T>(comm, file, src_pop_name, dst_pop_name,
                                                  attr_namespace, attr_name,
                                                  edge_attr_values.attr_vec<T>(i),
						  chunk_size);
                }
            }
          else
            {
              throw std::runtime_error("append_edge_attribute_map: namespace mismatch");
            }

        }
    }

  }
}

#endif
