#include "debug.hh"
#include "population_reader.hh"

#include "ngh5paths.h"

#include <cstdio>
#include <iostream>
#include <vector>

#undef NDEBUG
#include <cassert>

using namespace std;

namespace ngh5
{

std::string ngh5_pop_path (const char *name) 
{
  std::string result;
  result = std::string("/H5Types/") + name;
  return result;
}

/*****************************************************************************
 * Read the valid population combinations
 *****************************************************************************/

herr_t read_population_combos
(
 MPI_Comm                   comm,
 const char*                fname, 
 set< pair<pop_t,pop_t> >&  pop_pairs
 )
{
  herr_t ierr = 0;

  int rank, size;

  printf("entering read_population_combos\n");
  
  assert(MPI_Comm_size(comm, &size) >= 0);
  assert(MPI_Comm_rank(comm, &rank) >= 0);

  // MPI rank 0 reads and broadcasts the number of pairs 

  uint64_t num_pairs;

  hid_t file = -1, dset = -1;

  // process 0 reads the number of pairs and broadcasts
  if (rank == 0)
    {
      file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
      assert(file >= 0);

      dset = H5Dopen2(file, ngh5_pop_path (POP_COMB_H5_PATH).c_str(), H5P_DEFAULT);
      assert(dset >= 0);
      hid_t fspace = H5Dget_space(dset);
      assert(fspace >= 0);
      num_pairs = (uint64_t) H5Sget_simple_extent_npoints(fspace);
      assert(num_pairs > 0);
      assert(H5Sclose(fspace) >= 0);
    }

  assert(MPI_Bcast(&num_pairs, 1, MPI_UINT64_T, 0, comm) >= 0);

  // allocate buffers
  vector<pop_t> v(2*num_pairs);

  // MPI rank 0 reads and broadcasts the population pairs 

  if (rank == 0)
    {
      vector<pop_comb_t> vpp(num_pairs);
      hid_t ftype = H5Dget_type(dset);
      assert(ftype >= 0);
      hid_t mtype = H5Tget_native_type(ftype, H5T_DIR_ASCEND);

      assert(H5Dread(dset, mtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &vpp[0]) >= 0);
      for (size_t i = 0; i < vpp.size(); ++i)
	{
          v[2*i]   = vpp[i].src;
          v[2*i+1] = vpp[i].dst;
	}

      assert(H5Tclose(mtype) >= 0);
      assert(H5Tclose(ftype) >= 0);

      assert(H5Dclose(dset) >= 0);
      assert(H5Fclose(file) >= 0);
    }

  assert(MPI_Bcast(&v[0], (int)2*num_pairs, MPI_UINT16_T, 0, comm) >= 0);

  // populate the set
  pop_pairs.clear();
  for (size_t i = 0; i < v.size(); i += 2)
    {
      pop_pairs.insert(make_pair(v[i], v[i+1]));
    }

  return ierr;
}

/*****************************************************************************
 * Read the population ranges
 *****************************************************************************/

herr_t read_population_ranges
(
 MPI_Comm                                comm,
 const char*                             fname, 
 map<NODE_IDX_T, pair<uint32_t,pop_t> >& pop_ranges,
 vector<pop_range_t> &pop_vector
 )
{
  herr_t ierr = 0;

  int rank, size;
  assert(MPI_Comm_size(comm, &size) >= 0);
  assert(MPI_Comm_rank(comm, &rank) >= 0);

  // MPI rank 0 reads and broadcasts the number of ranges 

  uint64_t num_ranges;

  hid_t file = -1, dset = -1;

  // process 0 reads the number of ranges and broadcasts
  if (rank == 0)
    {
      file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
      assert(file >= 0);

      dset = H5Dopen2(file, ngh5_pop_path (POP_RANGE_H5_PATH).c_str(), H5P_DEFAULT);
      assert(dset >= 0);
      hid_t fspace = H5Dget_space(dset);
      assert(fspace >= 0);
      num_ranges = (uint64_t) H5Sget_simple_extent_npoints(fspace);
      assert(num_ranges > 0);
      assert(H5Sclose(fspace) >= 0);
    }

  assert(MPI_Bcast(&num_ranges, 1, MPI_UINT64_T, 0, comm) >= 0);

  // allocate buffers
  pop_vector.resize(num_ranges);

  // MPI rank 0 reads and broadcasts the population ranges

  if (rank == 0)
    {
      hid_t ftype = H5Dget_type(dset);
      assert(ftype >= 0);
      hid_t mtype = H5Tget_native_type(ftype, H5T_DIR_ASCEND);

      assert(H5Dread(dset, mtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &pop_vector[0]) >= 0);

      assert(H5Tclose(mtype) >= 0);
      assert(H5Tclose(ftype) >= 0);

      assert(H5Dclose(dset) >= 0);
      assert(H5Fclose(file) >= 0);
    }

  assert(MPI_Bcast(&pop_vector[0], (int)num_ranges*sizeof(pop_range_t), MPI_BYTE, 0,
                   comm) >= 0);

  for(size_t i = 0; i < pop_vector.size(); ++i)
    {
      pop_ranges.insert(make_pair(pop_vector[i].start, make_pair(pop_vector[i].count, pop_vector[i].pop)));
    }

  return ierr;
}

/*****************************************************************************
 * Validate edge populations
 *****************************************************************************/

bool validate_edge_list
(
 NODE_IDX_T&         dst_start,
 NODE_IDX_T&         src_start,
 vector<DST_BLK_PTR_T>&  dst_blk_ptr,
 vector<NODE_IDX_T>& dst_idx,
 vector<DST_PTR_T>&  dst_ptr,
 vector<NODE_IDX_T>& src_idx,
 const pop_range_map_t&           pop_ranges,
 const set< pair<pop_t, pop_t> >& pop_pairs
 )
{
  bool result = true;

  NODE_IDX_T src, dst;

  pop_range_iter_t riter, citer;

  pair<pop_t,pop_t> pp;

  // loop over all edges, look up the node populations, and validate the pairs

  if (dst_blk_ptr.size() > 0) 
    {
      size_t dst_ptr_size = dst_ptr.size();
      for (size_t b = 0; b < dst_blk_ptr.size()-1; ++b)
        {
          size_t low_dst_ptr = dst_blk_ptr[b], high_dst_ptr = dst_blk_ptr[b+1];
          NODE_IDX_T dst_base = dst_idx[b];
          for (size_t i = low_dst_ptr, ii = 0; i < high_dst_ptr; ++i, ++ii)
            {
              if (i < dst_ptr_size-1)
                {
                  dst = dst_base + ii + dst_start;
                  riter = pop_ranges.upper_bound(dst);
                  if (riter == pop_ranges.end())
                    {
                      if (dst < pop_ranges.rbegin()->first+pop_ranges.rbegin()->second.first)
                        {
                          pp.second = pop_ranges.rbegin()->second.second;
                        }
                      else
                        {
                          DEBUG("unable to find population for dst = ",dst,"\n"); 
                          return false;
                        }
                    }
                  else
                    {
                      pp.second = riter->second.second-1;
                    }
                  size_t low = dst_ptr[i], high = dst_ptr[i+1];
                  if ((high-low) == 0)
                    {
                      result = true;
                    }
                  else
                    {
                      for (size_t j = low; j < high; ++j)
                        {
                          src = src_idx[j] + src_start;
                          citer = pop_ranges.upper_bound(src);
                          if (citer == pop_ranges.end())
                            {
                              if (src < pop_ranges.rbegin()->first+pop_ranges.rbegin()->second.first)
                                {
                                  pp.first = pop_ranges.rbegin()->second.second;
                                }
                              else
                                {
                                  DEBUG("unable to find population for src = ",src,"\n"); 
                                  return false;
                                }
                            }
                          else
                            {
                              pp.first = citer->second.second-1;
                            }
                          // check if the population combo is valid
                          result = (pop_pairs.find(pp) != pop_pairs.end());
                          if (!result)
                            {
                              DEBUG("invalid edge: src = ",src," dst = ",dst," pp = ",pp.first,", ",pp.second,"\n"); 
                              return false;
                            }
                        }
                    }
                }
            }
        }
    }

  return result;
}

}
