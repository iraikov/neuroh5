
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <vector>

using namespace std;

// Calculate the starting and stopping block for each rank
void compute_bins (size_t num_blocks, size_t size, vector< pair<size_t,size_t> > &bins)
{
  size_t remainder=0, offset=0, buckets=0;
  
  for (size_t i=0; i<size; i++)
    {
        remainder = num_blocks - offset;
        buckets   = (size - i);
        bins[i] = make_pair(remainder / buckets, offset);
        offset += bins[i].first;
    }

}

int main (int argc, char **argv)
{
  vector< pair<size_t,size_t> > bins;
  size_t size, num_blocks;

  size = 4;
  num_blocks = 317;

  bins.resize(size);
  
  compute_bins(num_blocks, size, bins);

  printf("*** test 1:\n");
  for (size_t i = 0; i<size; i++)
    {
      printf("bins[%lu] = %lu (%lu)\n", i, bins[i].first, bins[i].second);
    }
  
  num_blocks = 424;

  bins.resize(size);

  compute_bins(num_blocks, size, bins);

  printf("*** test 2:\n");
  for (size_t i = 0; i<size; i++)
    {
      printf("bins[%lu] = %lu (%lu)\n", i, bins[i].first, bins[i].second);
    }

  size = 12;
  num_blocks = 10;

  bins.resize(size);

  compute_bins(num_blocks, size, bins);

  printf("*** test 3:\n");
  for (size_t i = 0; i<size; i++)
    {
      printf("bins[%lu] = %lu (%lu)\n", i, bins[i].first, bins[i].second);
    }

}

