
#include "read_population.hh"

#include "hdf5_path_names.hh"
#include "read_singleton_dataset.hh"

using namespace std;

namespace ngh5
{
  namespace io
  {
    void read_destination_population
    (
     MPI_Comm      comm,
     const string& file_name,
     const string& prj_name,
     uint32_t&     pop
     )
    {
      io::hdf5::read_singleton_dataset
        (comm, file_name, io::hdf5::projection_path_join(prj_name,
                                                         io::hdf5::DST_POP),
         H5T_NATIVE_UINT, MPI_UINT32_T, pop);

    };

    void read_source_population
    (
     MPI_Comm      comm,
     const string& file_name,
     const string& prj_name,
     uint32_t&     pop
     )
    {
      io::hdf5::read_singleton_dataset
        (comm, file_name, io::hdf5::projection_path_join(prj_name,
                                                         io::hdf5::SRC_POP),
         H5T_NATIVE_UINT, MPI_UINT32_T, pop);

    };
  }
}
