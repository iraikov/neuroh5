#ifndef TOKENIZE_HH
#define TOKENIZE_HH

#include <string>
#include <vector>
#include <algorithm>
using namespace std;


void tokenize (string str, const string& delimiter, vector<string> &token_vector)
{
    size_t start = str.find_first_not_of(delimiter), end=start;

    while (start != string::npos)
      {
        // Find next occurence of delimiter
        end = str.find(delimiter, start);
        // Push back the token found into vector
        token_vector.push_back(str.substr(start, end-start));
        // Skip all occurrences of the delimiter to find new start
        start = str.find_first_not_of(delimiter, end);
      }
}

#endif
