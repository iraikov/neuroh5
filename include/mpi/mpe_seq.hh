
#include <mpi.h>

using namespace std;

namespace neuroh5
{
  namespace mpi
  {
    void MPE_Seq_begin( MPI_Comm comm, int ng );
    void MPE_Seq_end( MPI_Comm comm, int ng );
  }
}
