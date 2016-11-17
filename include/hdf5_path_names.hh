#ifndef HDF5_PATH_NAMES_HH
#define HDF5_PATh_NAMES_HH

#include <string>

namespace ngh5
{
  namespace io
  {
    namespace hdf5
    {
      const std::string ATTR        = "Attributes";

      const std::string CONN        = "Connectivity";
      const std::string SRC_IDX     = "Connectivity/Source Index";
      const std::string DST_BLK_IDX = "Connectivity/Destination Block Index";
      const std::string DST_BLK_PTR = "Connectivity/Destination Block Pointer";
      const std::string DST_PTR     = "Connectivity/Destination Pointer";

      const std::string EDGE        = "Edge";

      const std::string H5_TYPES    = "H5Types";

      const std::string POP         = "Populations";

      const std::string PRJ         = "Projections";

      const std::string POP_COMB    = "Valid population projections";

      const std::string DST_POP     = "Destination Population";

      const std::string SRC_POP     = "Source Population";

      const std::string POP_RNG     = "Population range";

      /// @brief Returns the path to the population's edge attributes
      ///
      /// @param proj_name         Projection data set name
      ///
      /// @return                  A string containing the full path to the
      ///                          population attributes
      std::string edge_attribute_path
      (
       const std::string& proj_name
       );

      /// @brief Returns the path to edge attributes
      ///
      /// @param proj_name         Projection data set name
      ///
      /// @param attr_name         Edge attribute name
      ///
      /// @return                  A string containing the full path to the
      ///                          attribute data set
      std::string edge_attribute_path
      (
       const std::string& proj_name,
       const std::string& attr_name
       );

      extern std::string projection_path_join
      (
       const std::string& proj_name,
       const std::string& name
       );

      extern std::string h5types_path_join(const std::string& name);

    }
  }
}

#endif
