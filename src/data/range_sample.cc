
#include <set>
#include <vector>
#include <algorithm>
#include <numeric>

#include "range_sample.hh"

using namespace std;

namespace neuroh5
{
  namespace data
  {


    void range_sample (size_t N, size_t m, set<size_t>& out)
    {
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

    }

  }
}

