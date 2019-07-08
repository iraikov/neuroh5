#ifndef RANGE_SAMPLE_HH
#define RANGE_SAMPLE_HH

#include <set>
#include <algorithm>

using namespace std;

namespace neuroh5
{
  namespace data
  {
    void range_sample (size_t N, size_t m, set<size_t>& out);
  }
}
#endif
