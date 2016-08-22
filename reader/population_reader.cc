#include "population_reader.hh"

#include "ngh5paths.h"

#include <cassert>
#include <cstdio>
#include <iostream>
#include <vector>

using namespace std;

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

      dset = H5Dopen2(file, POP_COMB_H5_PATH, H5P_DEFAULT);
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
 map<NODE_IDX_T, pair<uint32_t,pop_t> >& pop_ranges
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

      dset = H5Dopen2(file, POP_RANGE_H5_PATH, H5P_DEFAULT);
      assert(dset >= 0);
      hid_t fspace = H5Dget_space(dset);
      assert(fspace >= 0);
      num_ranges = (uint64_t) H5Sget_simple_extent_npoints(fspace);
      assert(num_ranges > 0);
      assert(H5Sclose(fspace) >= 0);
    }

  assert(MPI_Bcast(&num_ranges, 1, MPI_UINT64_T, 0, comm) >= 0);

  // allocate buffers
  vector<pop_range_t> v(num_ranges);

  // MPI rank 0 reads and broadcasts the population ranges

  if (rank == 0)
    {
      hid_t ftype = H5Dget_type(dset);
      assert(ftype >= 0);
      hid_t mtype = H5Tget_native_type(ftype, H5T_DIR_ASCEND);

      assert(H5Dread(dset, mtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v[0]) >= 0);

      assert(H5Tclose(mtype) >= 0);
      assert(H5Tclose(ftype) >= 0);

      assert(H5Dclose(dset) >= 0);
      assert(H5Fclose(file) >= 0);
    }

  assert(MPI_Bcast(&v[0], (int)num_ranges*sizeof(pop_range_t), MPI_BYTE, 0,
                   comm) >= 0);

  for(size_t i = 0; i < v.size(); ++i)
    {
      pop_ranges.insert(make_pair(v[i].start, make_pair(v[i].count, v[i].pop)));
    }

  return ierr;
}

/*****************************************************************************
 * Validate edge populations
 *****************************************************************************/
/*
bool validate_edge_list
(
 const NODE_IDX_T&                base,
 const vector<ROW_PTR_T>&         row_ptr,
 const vector<NODE_IDX_T>&        col_idx,
 const pop_range_map_t&           pop_ranges,
 const set< pair<pop_t, pop_t> >& pop_pairs
 )
{
  bool result = false;

  NODE_IDX_T row, col;

  pop_range_iter_t riter, citer;

  pair<pop_t,pop_t> pp;

  // loop over all edges, look up the node populations, and validate the pairs

  for (size_t i = 0; i < row_ptr.size(); ++i)
    {
      row = base + (NODE_IDX_T)i;
      riter = pop_ranges.lower_bound(row);
      if (riter == pop_ranges.end()) { return false; }
      pp.first = riter->second.second;

      if (i < row_ptr.size()-1)
	{
	  size_t low = row_ptr[i], high = row_ptr[i+1];
	  for (size_t j = low; j < high; ++j)
	    {
	      col = col_idx[j];
	      citer = pop_ranges.lower_bound(col);
	      if (citer == pop_ranges.end()) { return false; }
	      pp.second = citer->second.second;

	      // check if the population combo is valid
	      result = (pop_pairs.find(pp) != pop_pairs.end());

	      if (!result) { return false; }
	    }
	}
    }

  return result;
}
*/
