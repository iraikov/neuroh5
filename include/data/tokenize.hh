#ifndef TOKENIZE_HH
#define TOKENIZE_HH

#include <string>
#include <vector>
#include <algorithm>
using namespace std;

namespace neuroh5
{
  namespace data
  {
    void tokenize (string str, const string& delimiter, vector<string> &token_vector);

  }
}
#endif
