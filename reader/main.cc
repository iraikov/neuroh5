
#include "ngh5types.h"

#include "hdf5.h"

#include <cassert>
#include <cstdio>
#include <iostream>
#include <vector>
#include <mpi.h>

using namespace std;

// names

#define FNAME "crs.h5"

#define ROW_PTR_H5_PATH "row_ptr"
#define COL_IDX_H5_PATH "col_idx"


/*****************************************************************************
 * Read the basic CRS graph structure
 *****************************************************************************/

herr_t read_graph
(
MPI_Comm            comm,
const char*         fname, 
NODE_IDX_T&         base,     /* global index of the first node */
vector<ROW_PTR_T>&  row_ptr,  /* one element longer than owned nodes count */
vector<NODE_IDX_T>& col_idx
)
{
  herr_t ierr = 0;

  int rank, size;
  assert(MPI_Comm_size(MPI_COMM_WORLD, &size) >= 0);
  assert(MPI_Comm_rank(MPI_COMM_WORLD, &rank) >= 0);

  /***************************************************************************
   * MPI rank 0 reads and broadcasts the number of nodes
   ***************************************************************************/

  uint64_t num_nodes;

  // process 0 reads the size of row_ptr (= num_nodes+1)
  if (rank == 0)
  {
      hid_t file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
      assert(file >= 0);

      hid_t dset = H5Dopen2(file, ROW_PTR_H5_PATH, H5P_DEFAULT);
      assert(dset >= 0);
      hid_t fspace = H5Dget_space(dset);
      assert(fspace >= 0);
      num_nodes = (uint64_t) H5Sget_simple_extent_npoints(fspace) - 1;
      assert(num_nodes > 0);

      assert(H5Sclose(fspace) >= 0);
      assert(H5Dclose(dset) >= 0);
      assert(H5Fclose(file) >= 0);
  }

  assert(MPI_Bcast(&num_nodes, 1, MPI_UINT64_T, 0, comm) >= 0);

  /***************************************************************************
   * read ROW_PTR
   ***************************************************************************/

  // determine my block of row_ptr
  hsize_t start = (hsize_t) rank*num_nodes/size;
  base = (NODE_IDX_T) start;
  hsize_t stop = (hsize_t) (rank+1)*num_nodes/size + 1;
  // patch the last rank
  if (rank == size-1) { stop = (hsize_t) num_nodes+1; }

  hsize_t block = stop - start;

  // allocate buffer and memory dataspace
  row_ptr.resize(block);
  assert(row_ptr.size() > 0);

  hid_t mspace = H5Screate_simple(1, &block, NULL);
  assert(mspace >= 0);
  ierr = H5Sselect_all(mspace);
  assert(ierr >= 0);

  // open the file (independent I/O)
  hid_t file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
  assert(file >= 0);
  hid_t dset = H5Dopen2(file, ROW_PTR_H5_PATH, H5P_DEFAULT);
  assert(dset >= 0);

  // make hyperslab selection
  hid_t fspace = H5Dget_space(dset);
  assert(fspace >= 0);
  hsize_t one = 1;
  ierr = H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL, &one, &block);
  assert(ierr >= 0);

  ierr = H5Dread(dset, ROW_PTR_H5_NATIVE_T, mspace, fspace, H5P_DEFAULT, &row_ptr[0]);
  assert(ierr >= 0);

  assert(H5Sclose(fspace) >= 0);
  assert(H5Dclose(dset) >= 0);
  assert(H5Sclose(mspace) >= 0);

  // rebase the row_ptr array to local offsets
  ROW_PTR_T rebase = row_ptr[0];
  for (size_t i = 0; i < row_ptr.size(); ++i)
  {
    row_ptr[i] -= rebase;
  }

  /***************************************************************************
   * read COL_IDX
   ***************************************************************************/

  // determine my read block of col_idx
  start = (hsize_t) row_ptr.front();
  stop = (hsize_t) row_ptr.back();
  block = stop - start;

  // allocate buffer and memory dataspace
  col_idx.resize(block);
  assert(col_idx.size() > 0);

  mspace = H5Screate_simple(1, &block, NULL);
  assert(mspace >= 0);
  ierr = H5Sselect_all(mspace);
  assert(ierr >= 0);

  dset = H5Dopen2(file, COL_IDX_H5_PATH, H5P_DEFAULT);
  assert(dset >= 0);

  // make hyperslab selection
  fspace = H5Dget_space(dset);
  assert(fspace >= 0);
  ierr = H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL, &one, &block);
  assert(ierr >= 0);

  ierr = H5Dread(dset, NODE_IDX_H5_NATIVE_T , mspace, fspace, H5P_DEFAULT, &col_idx[0]);
  assert(ierr >= 0);

  assert(H5Sclose(fspace) >= 0);
  assert(H5Dclose(dset) >= 0);
  assert(H5Sclose(mspace) >= 0);

  assert(H5Fclose(file) >= 0);

  return ierr;
}

/*****************************************************************************
 * Create a list of edges (represented as src/dst node pairs)
 *****************************************************************************/

int create_edge_list
(
  const NODE_IDX_T&         base,
  const vector<ROW_PTR_T>&  row_ptr,
  const vector<NODE_IDX_T>& col_idx,
  vector<NODE_IDX_T>&       edge_list
)
{
  int ierr = 0;

  for (size_t i = 0; i < row_ptr.size(); ++i)
  {
    NODE_IDX_T row = base + (NODE_IDX_T)i;
    if (i < row_ptr.size()-1)
    {
      size_t low = row_ptr[i], high = row_ptr[i+1];
      for (size_t j = low; j < high; ++j)
      {
        NODE_IDX_T col = col_idx[j];
        edge_list.push_back(row);
        edge_list.push_back(col);
      }
    }
  }

  return ierr;
}


/*****************************************************************************
 * Main driver
 *****************************************************************************/

int main(int argc, char** argv)
{
  assert(MPI_Init(&argc, &argv) >= 0);

  int rank, size;
  assert(MPI_Comm_size(MPI_COMM_WORLD, &size) >= 0);
  assert(MPI_Comm_rank(MPI_COMM_WORLD, &rank) >= 0);

  // parse arguments

  // read the file

  NODE_IDX_T base;
  vector<ROW_PTR_T> row_ptr;
  vector<NODE_IDX_T> col_idx;
  assert(read_graph(MPI_COMM_WORLD, FNAME, base, row_ptr, col_idx) >= 0);

  // create the partitioner input

  vector<NODE_IDX_T> edge_list;
  assert(create_edge_list(base, row_ptr, col_idx, edge_list) >= 0);
  assert(edge_list.size()%2 == 0);

  // partition the graph

  
  MPI_Finalize();
  return 0;
}
