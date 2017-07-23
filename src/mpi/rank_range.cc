
#include <hdf5.h>
#include "rank_range.hh"

using namespace std;

namespace neuroh5
{
  namespace mpi
  {
    // Given a total number of elements and number of ranks, calculate the starting and length for each rank
    void rank_ranges
    (
     const size_t&                    num_elems,
     const size_t&                    size,
     vector< pair<hsize_t,hsize_t> >& ranges
     )
    {
      hsize_t remainder=0, offset=0, buckets=0;
      ranges.resize(size);
    
      for (size_t i=0; i<size; i++)
        {
          remainder = num_elems - offset;
          buckets   = (size - i);
          ranges[i] = make_pair(offset, remainder / buckets);
          offset    += ranges[i].second;
        }
    }

  }
  
}
