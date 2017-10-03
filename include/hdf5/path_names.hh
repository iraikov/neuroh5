#ifndef PATH_NAMES_HH
#define PATH_NAMES_HH

#include <string>

namespace neuroh5
{
  namespace hdf5
  {
    
    const std::string POPULATIONS = "Populations";
    const std::string PROJECTIONS = "Projections";
    const std::string NODES       = "Nodes";
    const std::string EDGES       = "Edges";
    const std::string H5_TYPES    = "H5Types";
    const std::string POP_LABELS  = "Population labels";
    const std::string POP_COMBS   = "Valid population projections";
  
    const std::string TREES      = "Trees";
    const std::string X_COORD    = "X Coordinate";
    const std::string Y_COORD    = "Y Coordinate";
    const std::string Z_COORD    = "Z Coordinate";
    const std::string RADIUS     = "Radius";
    const std::string LAYER      = "Point Layer";
    const std::string SECTION    = "Section";
    const std::string SRCSEC     = "Source Section";
    const std::string DSTSEC     = "Destination Section";
    const std::string PARENT     = "Parent Point";
    const std::string SWCTYPE    = "SWC Type";
    const std::string CELL_INDEX = "Cell Index";
    const std::string NODE_INDEX = "Node Index";

    const std::string ATTR_PTR   = "Attribute Pointer";
    const std::string TOPO_PTR   = "Topology Pointer";
    const std::string SEC_PTR    = "Section Pointer";
    const std::string ATTR_VAL   = "Attribute Value";

    const std::string DST_BLK_PTR = "Destination Block Pointer";
    const std::string DST_BLK_IDX = "Destination Block Index";
    const std::string DST_PTR     = "Destination Pointer";
    const std::string SRC_IDX     = "Source Index";

    std::string h5types_path_join(const std::string& name);

    /// @brief Returns the path to a population group
    ///
    /// @param pop_name         Population group name
    ///
    /// @return                  A string containing the full path to the
    ///                          population group
    std::string population_path
    (
     const std::string& pop_name
     );
  
  
    /// @brief Returns the path to cell attributes
    ///
    /// @param pop_name         Population data set name
    ///
    /// @param attr_name        Cell attribute name
    ///
    /// @return                  A string containing the full path to the
    ///                          attribute data set
    std::string cell_attribute_path
    (
     const std::string& name_space,
     const std::string& pop_name,
     const std::string& attr_name
     );
  
    std::string cell_attribute_prefix
    (
     const std::string& name_space,
     const std::string& pop_name
     );
    
    /// @brief Returns the path to projection edge attributes
    ///
    /// @param src_pop_name         Source population name
    ///
    /// @param dst_pop_name         Destination population name
    ///
    /// @param attr_name        Tree attribute name
    ///
    /// @return                  A string containing the full path to the
    ///                          attribute data set
    std::string edge_attribute_path
    (
     //const std::string& name_space,
     const std::string& src_pop_name,
     const std::string& dst_pop_name,
     const std::string& name_space,
     const std::string& attr_name
     );

    std::string edge_attribute_prefix
    (
     const std::string& src_pop_name,
     const std::string& dst_pop_name,
     const std::string& name_space
     );

    std::string projection_prefix
    (
     const std::string& src_pop_name,
     const std::string& dst_pop_name
     );

    std::string node_attribute_path
    (
     const std::string& name_space,
     const std::string& attr_name
     );

    std::string node_attribute_prefix
    (
     const std::string& name_space
     );
    
  }
}

#endif
