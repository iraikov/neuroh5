#include "csc_graph_reader.hh"

#include "ngh5paths.h"

#include <cassert>
#include <cstdio>
#include <iostream>

using namespace std;

/*****************************************************************************
 * Read the basic CSC graph structure
 *****************************************************************************/

herr_t read_csc_graph
(
 MPI_Comm            comm,
 const char*         fname, 
 NODE_IDX_T&         base,
 vector<BLOCK_PTR_T>&  block_ptr,
 vector<COL_PTR_T>&  col_ptr,
 vector<NODE_IDX_T>& row_idx
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

  // process 0 reads the size of col_ptr (= num_nodes+1)
  if (rank == 0)
    {
      hid_t file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
      assert(file >= 0);

      hid_t dset = H5Dopen2(file, BLOCK_PTR_H5_PATH, H5P_DEFAULT);
      assert(dset >= 0);
      hid_t fspace = H5Dget_space(dset);
      assert(fspace >= 0);
      num_blocks = (uint64_t) H5Sget_simple_extent_npoints(fspace) - 1;
      assert(num_blocks > 0);

      assert(H5Sclose(fspace) >= 0);
      assert(H5Dclose(dset) >= 0);
      assert(H5Fclose(file) >= 0);
    }

  assert(MPI_Bcast(&num_blocks, 1, MPI_UINT64_T, 0, comm) >= 0);

  /***************************************************************************
   * read BLOCK_PTR
   ***************************************************************************/

  // determine my block of block_ptr
  hsize_t ppcount = (hsize_t) num_blocks/size;
  hsize_t start = (hsize_t) rank*ppcount;
  base = (NODE_IDX_T) start;
  hsize_t stop = (hsize_t) (rank+1)*ppcount + 1;
  // patch the last rank
  if (rank == size-1) { stop = (hsize_t) num_blocks+1; }

  hsize_t block = stop - start;

  // allocate buffer and memory dataspace
  block_ptr.resize(block);
  assert(block_ptr.size() > 0);

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
  hid_t dset = H5Dopen2(file, BLOCK_PTR_H5_PATH, H5P_DEFAULT);
  assert(dset >= 0);

  // make hyperslab selection
  hid_t fspace = H5Dget_space(dset);
  assert(fspace >= 0);
  hsize_t one = 1;
  ierr = H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL, &one, &block);
  assert(ierr >= 0);

  ierr = H5Dread(dset, BLOCK_PTR_H5_NATIVE_T, mspace, fspace, H5P_DEFAULT, &row_ptr[0]);
  assert(ierr >= 0);

  assert(H5Sclose(fspace) >= 0);
  assert(H5Dclose(dset) >= 0);
  assert(H5Sclose(mspace) >= 0);

  // rebase the block_ptr array to local offsets
  // REBASE is going to be the start offset for the hyperslab
  BLOCK_PTR_T block_rebase = block_ptr[0];
  for (size_t i = 0; i < block_ptr.size(); ++i)
    {
      block_ptr[i] -= block_rebase;
    }

  /***************************************************************************
   * read COL_PTR
   ***************************************************************************/

  // determine my read block of col_idx
  block = (hsize_t)(block_ptr.back() - block_ptr.front());
  start = (hsize_t)block_rebase;

  // allocate buffer and memory dataspace
  col_ptr.resize(block);
  assert(col_ptr.size() > 0);

  mspace = H5Screate_simple(1, &block, NULL);
  assert(mspace >= 0);
  ierr = H5Sselect_all(mspace);
  assert(ierr >= 0);

  dset = H5Dopen2(file, COL_PTR_H5_PATH, H5P_DEFAULT);
  assert(dset >= 0);

  // make hyperslab selection
  fspace = H5Dget_space(dset);
  assert(fspace >= 0);
  ierr = H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL, &one, &block);
  assert(ierr >= 0);

  ierr = H5Dread(dset, NODE_IDX_H5_NATIVE_T, mspace, fspace, H5P_DEFAULT, &col_ptr[0]);
  assert(ierr >= 0);

  assert(H5Sclose(fspace) >= 0);
  assert(H5Dclose(dset) >= 0);
  assert(H5Sclose(mspace) >= 0);

  BLOCK_PTR_T col_rebase = col_ptr[0];
  for (size_t i = 0; i < col_ptr.size(); ++i)
    {
      col_ptr[i] -= col_rebase;
    }

  /***************************************************************************
   * read ROW_IDX
   ***************************************************************************/

  // determine my read block of col_idx
  block = (hsize_t)(col_ptr.back() - col_ptr.front());
  start = (hsize_t)col_rebase;

  // allocate buffer and memory dataspace
  row_idx.resize(block);
  assert(row_idx.size() > 0);

  mspace = H5Screate_simple(1, &block, NULL);
  assert(mspace >= 0);
  ierr = H5Sselect_all(mspace);
  assert(ierr >= 0);

  dset = H5Dopen2(file, ROW_IDX_H5_PATH, H5P_DEFAULT);
  assert(dset >= 0);

  // make hyperslab selection
  fspace = H5Dget_space(dset);
  assert(fspace >= 0);
  ierr = H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL, &one, &block);
  assert(ierr >= 0);

  ierr = H5Dread(dset, NODE_IDX_H5_NATIVE_T, mspace, fspace, H5P_DEFAULT, &row_idx[0]);
  assert(ierr >= 0);

  assert(H5Sclose(fspace) >= 0);
  assert(H5Dclose(dset) >= 0);
  assert(H5Sclose(mspace) >= 0);

  assert(H5Fclose(file) >= 0);

  return ierr;
}
