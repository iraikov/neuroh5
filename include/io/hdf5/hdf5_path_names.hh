#ifndef HDF5_PATH_NAMES_HH
#define HDF5_PATH_NAMES_HH

#include <string>

namespace neurotrees
{
  const std::string POPS       = "Populations";
  const std::string H5_TYPES   = "H5Types";
  const std::string POPLABELS  = "Population labels";
  
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
  const std::string TREE_ID    = "Tree ID";

  const std::string ATTR_PTR   = "Attribute Pointer";
  const std::string TOPO_PTR   = "Topology Pointer";
  const std::string SEC_PTR    = "Section Pointer";
  
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
  
  /// @brief Returns the path to the trees group of population group
  ///
  /// @param pop_name         Population group name
  ///
  /// @return                  A string containing the full path to the
  ///                          trees group for the given population
  std::string population_trees_path
  (
   const std::string& pop_name
   );
  
  /// @brief Returns the path to cell attributes
  ///
  /// @param pop_name         Population data set name
  ///
  /// @param attr_name        Tree attribute name
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

  std::string h5types_path_join(const std::string& name);

  
}

#endif
