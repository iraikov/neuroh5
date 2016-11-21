#ifndef READ_POPULATION_HH
#define READ_POPULATION_HH

#include <mpi.h>

#include <cstdint>
#include <string>

namespace ngh5
{
  namespace io
  {
    void read_destination_population
    (
     MPI_Comm           comm,
     const std::string& file_name,
     const std::string& prj_name,
     uint32_t&          pop
     );

    void read_source_population
    (
     MPI_Comm           comm,
     const std::string& file_name,
     const std::string& prj_name,
     uint32_t&          pop
     );
  }
}

#endif
