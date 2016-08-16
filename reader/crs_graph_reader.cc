#include "crs_graph_reader.hh"

#include "ngh5paths.h"

#include <cassert>
#include <cstdio>
#include <iostream>

using namespace std;

/*****************************************************************************
 * Read the basic CRS graph structure
 *****************************************************************************/

herr_t read_crs_graph
(
 MPI_Comm            comm,
 const char*         fname, 
 NODE_IDX_T&         base,
 vector<ROW_PTR_T>&  row_ptr,
 vector<NODE_IDX_T>& col_idx
 )
{
  herr_t ierr = 0;

  int rank, size;
  assert(MPI_Comm_size(comm, &size) >= 0);
  assert(MPI_Comm_rank(comm, &rank) >= 0);

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
  hsize_t ppcount = (hsize_t) num_nodes/size;
  hsize_t start = (hsize_t) rank*ppcount;
  base = (NODE_IDX_T) start;
  hsize_t stop = (hsize_t) (rank+1)*ppcount + 1;
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
  hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
  assert(fapl >= 0);
  assert(H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL) >= 0);
  hid_t file = H5Fopen(fname, H5F_ACC_RDONLY, fapl);
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
  // REBASE is going to be the start offset for the hyperslab
  ROW_PTR_T rebase = row_ptr[0];
  for (size_t i = 0; i < row_ptr.size(); ++i)
    {
      row_ptr[i] -= rebase;
    }

  /***************************************************************************
   * read COL_IDX
   ***************************************************************************/

  // determine my read block of col_idx
  block = (hsize_t)(row_ptr.back() - row_ptr.front());
  start = (hsize_t)rebase;

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
