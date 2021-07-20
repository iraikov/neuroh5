
#include <set>
#include <vector>
#include <algorithm>
#include <numeric>

#include "range_sample.hh"
#include "throw_assert.hh"

using namespace std;

namespace neuroh5
{
  namespace data
  {


    void range_sample (size_t N, size_t m, set<size_t>& out)
    {
      throw_assert(N > 0, "range_sample: invalid N <= 0");
      throw_assert(m > 0, "range_sample: invalid m <= 0");

      if (N < m)
	m = N;
      size_t h =  N / m;
      std::vector<double> xs(N);
      std::vector<double>::iterator x;
      size_t val;
      for (size_t i = 0, val = 0; i < m; ++i, val += h) 
	{
	  out.insert(val);
	}
      throw_assert(out.size() > 0, "range_sample: invalid output set");

    }

  }
}

