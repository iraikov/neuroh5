#include <hdf5.h>
#include <vector>

using namespace std;

namespace neurotrees
{
  // Given a total number of elements and number of ranks, calculate the starting and length for each rank
  void rank_ranges
  (
   const size_t&                    num_elems,
   const size_t&                    size,
   std::vector< std::pair<hsize_t,hsize_t> >& ranges
   );

  
  
}
