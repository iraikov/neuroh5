#ifndef SPLIT_INTERVALS_HH
#define SPLIT_INTERVALS_HH

#include <algorithm>
#include <iterator>
#include <vector>

using namespace std;

namespace neuroh5
{
  namespace data
  {
    // based on https://stackoverflow.com/questions/40656792/c-best-way-to-split-vector-into-n-vector
    template<typename V>
    auto split_intervals(const V& v, unsigned numitems)  -> std::vector<V>
    {
      using Iterator = typename V::const_iterator;
      std::vector<V> rtn;
      Iterator it = v.cbegin();
      const Iterator end = v.cend();

      while (it != end) {
	V v;
	std::back_insert_iterator<V> inserter(v);
	const auto num_to_copy = std::min(static_cast<unsigned>(std::distance(it, end)), 
					  numitems);
	std::copy(it, it + num_to_copy, inserter);
	rtn.push_back(std::move(v));
	std::advance(it, num_to_copy);
      }

      return rtn;
    }
  }
}
#endif
