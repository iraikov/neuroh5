
#include <hdf5.h>
#include <cassert>
#include <string>

#include "neuroio_types.hh"
#include "hdf5_types.hh"
#include "hdf5_enum_type.hh"
#include "hdf5_path_names.hh"
#include "create_tree_dataset.hh"

namespace neuroio
{

  namespace io
  {

    namespace hdf5
    {
  
      void throw_err(char const* err_message)
      {
        fprintf(stderr, "Error: %s\n", err_message);
        abort();
      }



      /*****************************************************************************
       * Create extensible dataset for tree structures
       *****************************************************************************/
      int create_tree_dataset
      (
       MPI_Comm comm,
       hid_t  file,
       const std::string& pop_name
       )
      {
        herr_t status;  
        hid_t  dataspace, group, dataset;  
        hid_t  plist;                     

        hsize_t dims[1]    = {0}; /* dataset dimensions at creation time */		
        hsize_t maxdims[1] = {H5S_UNLIMITED};
        hsize_t cdims[1]   = {1000}; /* chunking dimensions */		
    
        int rank, size;
        assert(MPI_Comm_size(comm, &size) >= 0);
        assert(MPI_Comm_rank(comm, &rank) >= 0);
        
        /* Create HDF5 enumerated type for reading SWC type information */
        hid_t hdf5_swc_type = create_H5Tenum<SWC_TYPE_T> (swc_type_enumeration);

        hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
        assert(lcpl >= 0);
        assert(H5Pset_create_intermediate_group(lcpl, 1) >= 0);
        
        /* Create a population group, if one does not already exist.  */
        if ((!H5Lexists (file, POPS.c_str(), H5P_DEFAULT)) ||
            (!H5Lexists (file, population_path(pop_name).c_str(), H5P_DEFAULT)));
        {
          group = H5Gcreate2(file, population_path(pop_name).c_str(), lcpl, H5P_DEFAULT, H5P_DEFAULT);
          assert(group >= 0);
          status = H5Gclose(group);
          assert(status == 0);
        }
    
        /* Create a trees group in the population group (it is an error if one already exists).  */  
        if (H5Lexists (file, population_trees_path(pop_name).c_str(), H5P_DEFAULT))
          {
            throw_err("Population trees group already exists");
          }
        else
          {
            group = H5Gcreate2(file, population_trees_path(pop_name).c_str(), lcpl, H5P_DEFAULT, H5P_DEFAULT);
            assert(group >= 0);
            status = H5Gclose(group);
            assert(status == 0);
          }

        /* Create dataset creation properties, i.e. to enable chunking  */
        plist  = H5Pcreate (H5P_DATASET_CREATE);
        status = H5Pset_chunk(plist, 1, cdims);
        //status = H5Pset_deflate (plist, 6); 

        /* Create the data space with unlimited dimensions. */
        dataspace = H5Screate_simple (1, dims, maxdims); 

    
        /* Create an Attribute Pointer dataset within the specified file creation properties.  */
        dataset = H5Dcreate2 (file, cell_attribute_path(TREES, pop_name, ATTR_PTR).c_str(),
                              ATTR_PTR_H5_NATIVE_T, dataspace,
                              lcpl, plist, H5P_DEFAULT);

        assert(dataset >= 0);
        status = H5Dclose (dataset);
        status = H5Sclose (dataspace);
    
        /* Create a Section Pointer dataset.  */
        dataspace = H5Screate_simple (1, dims, maxdims); 
        dataset = H5Dcreate2 (file, cell_attribute_path(TREES, pop_name, SEC_PTR).c_str(),
                              SEC_PTR_H5_NATIVE_T, dataspace,
                              lcpl, plist, H5P_DEFAULT);
        assert(dataset >= 0);
        status = H5Dclose (dataset);
        status = H5Sclose (dataspace);
    
        /* Create a Topology Pointer dataset.  */
        dataspace = H5Screate_simple (1, dims, maxdims); 
        dataset = H5Dcreate2 (file, cell_attribute_path(TREES, pop_name, TOPO_PTR).c_str(),
                              TOPO_PTR_H5_NATIVE_T, dataspace,
                              lcpl, plist, H5P_DEFAULT);

        assert(dataset >= 0);
        status = H5Dclose (dataset);
        status = H5Sclose (dataspace);
    
        /* Create a Tree ID dataset. */
        dataspace = H5Screate_simple (1, dims, maxdims); 
        dataset = H5Dcreate2 (file, cell_attribute_path(TREES, pop_name, TREE_ID).c_str(),
                              CELL_IDX_H5_NATIVE_T, dataspace,
                              lcpl, plist, H5P_DEFAULT);
        status = H5Dclose (dataset);
        status = H5Sclose (dataspace);

        /* Create a Source Node dataset. */
        dataspace = H5Screate_simple (1, dims, maxdims); 
        dataset = H5Dcreate2 (file, cell_attribute_path(TREES, pop_name, SRCSEC).c_str(),
                              SECTION_IDX_H5_NATIVE_T, dataspace,
                              lcpl, plist, H5P_DEFAULT);
        status = H5Dclose (dataset);
        status = H5Sclose (dataspace);
    
        /* Create a Destination Node dataset.  */
        dataspace = H5Screate_simple (1, dims, maxdims); 
        dataset = H5Dcreate2 (file, cell_attribute_path(TREES, pop_name, DSTSEC).c_str(),
                              SECTION_IDX_H5_NATIVE_T, dataspace,
                              lcpl, plist, H5P_DEFAULT);
        status = H5Dclose (dataset);
        status = H5Sclose (dataspace);
    
        /* Create a Section dataset.  */
        dataspace = H5Screate_simple (1, dims, maxdims); 
        dataset = H5Dcreate2 (file, cell_attribute_path(TREES, pop_name, SECTION).c_str(),
                              SECTION_IDX_H5_NATIVE_T, dataspace,
                              lcpl, plist, H5P_DEFAULT);
        status = H5Dclose (dataset);
        status = H5Sclose (dataspace);
    
    
        /* Create a X Coordinate dataset.  */
        dataspace = H5Screate_simple (1, dims, maxdims); 
        dataset = H5Dcreate2 (file, cell_attribute_path(TREES, pop_name, X_COORD).c_str(),
                              COORD_H5_NATIVE_T, dataspace,
                              lcpl, plist, H5P_DEFAULT);
        status = H5Dclose (dataset);
        status = H5Sclose (dataspace);
    
        /* Create a Y Coordinate dataset.  */
        dataspace = H5Screate_simple (1, dims, maxdims); 
        dataset = H5Dcreate2 (file, cell_attribute_path(TREES, pop_name, Y_COORD).c_str(),
                              COORD_H5_NATIVE_T, dataspace,
                              lcpl, plist, H5P_DEFAULT);
        status = H5Dclose (dataset);
        status = H5Sclose (dataspace);
    
        /* Create a Z Coordinate dataset.  */
        dataspace = H5Screate_simple (1, dims, maxdims); 
        dataset = H5Dcreate2 (file, cell_attribute_path(TREES, pop_name, Z_COORD).c_str(),
                              COORD_H5_NATIVE_T, dataspace,
                              lcpl, plist, H5P_DEFAULT);
        status = H5Dclose (dataset);
        status = H5Sclose (dataspace);
    
        /* Create a Radius dataset.  */
        dataspace = H5Screate_simple (1, dims, maxdims); 
        dataset = H5Dcreate2 (file, cell_attribute_path(TREES, pop_name, RADIUS).c_str(),
                              REAL_H5_NATIVE_T, dataspace,
                              lcpl, plist, H5P_DEFAULT);
        status = H5Dclose (dataset);
        status = H5Sclose (dataspace);
    
        /* Create a Layer dataset.  */
        dataspace = H5Screate_simple (1, dims, maxdims); 
        dataset = H5Dcreate2 (file, cell_attribute_path(TREES, pop_name, LAYER).c_str(),
                              LAYER_IDX_H5_NATIVE_T, dataspace,
                              lcpl, plist, H5P_DEFAULT);
        status = H5Dclose (dataset);
        status = H5Sclose (dataspace);

        /* Create a Parent Point dataset.  */
        dataspace = H5Screate_simple (1, dims, maxdims); 
        dataset = H5Dcreate2 (file, cell_attribute_path(TREES, pop_name, PARENT).c_str(),
                              PARENT_NODE_IDX_H5_NATIVE_T, dataspace,
                              lcpl, plist, H5P_DEFAULT);
        status = H5Dclose (dataset);
        status = H5Sclose (dataspace);

        /* Create a SWC Type dataset.  */
        dataspace = H5Screate_simple (1, dims, maxdims); 
        dataset = H5Dcreate2 (file, cell_attribute_path(TREES, pop_name, SWCTYPE).c_str(),
                              hdf5_swc_type, dataspace,
                              lcpl, plist, H5P_DEFAULT);
        status = H5Dclose (dataset);
        status = H5Sclose (dataspace);
    
        status = H5Tclose (hdf5_swc_type);
        status = H5Pclose (plist);
        status = H5Pclose (lcpl);
    
        return status;
      }
    }
  }
}
