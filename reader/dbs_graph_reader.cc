#include "dbs_graph_reader.hh"

#include "ngh5paths.h"

#include <cassert>
#include <cstdio>
#include <iostream>

using namespace std;

std::string ngh5_prj_path (const char *dsetname, const char *name) 
{
  std::string result;
  result = std::string("/Projections/") + dsetname + "/Connectivity/" + name;
  return result;
}

/*****************************************************************************
 * Read the basic DBS graph structure
 *****************************************************************************/

herr_t read_dbs_graph
(
 MPI_Comm            comm,
 const char*         fname, 
 const char*         dsetname, 
 NODE_IDX_T&         base,
 vector<DST_BLK_PTR_T>&  dst_blk_ptr,
 vector<NODE_IDX_T>& dst_idx,
 vector<DST_PTR_T>&  dst_ptr,
 vector<NODE_IDX_T>& src_idx
 )
{
  herr_t ierr = 0;
  int rank, size;
  assert(MPI_Comm_size(comm, &size) >= 0);
  assert(MPI_Comm_rank(comm, &rank) >= 0);

  /***************************************************************************
   * MPI rank 0 reads and broadcasts the number of nodes
   ***************************************************************************/

  uint64_t num_blocks;
  printf("rank = %d\n", rank);
  printf("size = %d\n", size);

  // process 0 reads the size of dst_blk_ptr (= num_nodes+1)
  if (rank == 0)
    {
      hid_t file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
      assert(file >= 0);
      hid_t dset = H5Dopen2(file, ngh5_prj_path(dsetname, DST_BLK_PTR_H5_PATH).c_str(), H5P_DEFAULT);
      assert(dset >= 0);
      hid_t fspace = H5Dget_space(dset);
      assert(fspace >= 0);
      num_blocks = (uint64_t) H5Sget_simple_extent_npoints(fspace) - 1;
      assert(num_blocks > 0);

      assert(H5Sclose(fspace) >= 0);
      assert(H5Dclose(dset) >= 0);
      assert(H5Fclose(file) >= 0);
    }
  printf("num_blocks = %lu\n", num_blocks);

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
  printf("ppcount = %llu\n", ppcount);
  printf("start = %llu\n", start);
  printf("stop = %llu\n", stop);

  hsize_t block = stop - start;
  printf("block = %llu\n", block);

  // allocate buffer and memory dataspace
  dst_blk_ptr.resize(block);
  assert(dst_blk_ptr.size() > 0);

  hid_t mspace = H5Screate_simple(1, &block, NULL);
  assert(mspace >= 0);
  ierr = H5Sselect_all(mspace);
  assert(ierr >= 0);

  printf("after create_simple: block = %llu\n", block);

  // open the file (independent I/O)
  hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
  assert(fapl >= 0);
  assert(H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL) >= 0);
  hid_t file = H5Fopen(fname, H5F_ACC_RDONLY, fapl);
  assert(file >= 0);
  hid_t dset = H5Dopen2(file, ngh5_prj_path(dsetname, DST_BLK_PTR_H5_PATH).c_str(), H5P_DEFAULT);
  assert(dset >= 0);

  // make hyperslab selection
  hid_t fspace = H5Dget_space(dset);
  assert(fspace >= 0);
  hsize_t one = 1;
  
  printf("before select_hyperslab: start = %llu\n", start);
  printf("before select_hyperslab: block = %llu\n", block);
  //ierr = H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL, &one, &block);
  ierr = H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL, &block, NULL);
  assert(ierr >= 0);
  printf("after select_hyperslab: ierr = %d\n", ierr);

  printf("dst_blk_ptr.size() = %lu\n", dst_blk_ptr.size());
  ierr = H5Dread(dset, DST_BLK_PTR_H5_NATIVE_T, mspace, fspace, H5P_DEFAULT, &dst_blk_ptr[0]);
  assert(ierr >= 0);

  assert(H5Sclose(fspace) >= 0);
  assert(H5Dclose(dset) >= 0);

  // rebase the block_ptr array to local offsets
  // REBASE is going to be the start offset for the hyperslab
  DST_BLK_PTR_T block_rebase = dst_blk_ptr[0];
  printf("block_rebase = %u\n", block_rebase);
  for (size_t i = 0; i < dst_blk_ptr.size(); ++i)
    {
      printf("before: dst_blk_ptr[%lu] = %u\n", i, dst_blk_ptr[i]);
      dst_blk_ptr[i] -= block_rebase;
      printf("after: dst_blk_ptr[%lu] = %u\n", i, dst_blk_ptr[i]);
    }

  /***************************************************************************
   * read DST_IDX
   ***************************************************************************/

  // determine my read block of dst_idx
  block = (hsize_t)(dst_blk_ptr.back() - dst_blk_ptr.front());
  start = (hsize_t)block_rebase;
  
  printf("block = %llu\n", block);
  printf("start = %llu\n", start);

  dst_idx.resize(block);
  assert(dst_idx.size() > 0);

  dset = H5Dopen2(file, ngh5_prj_path(dsetname, DST_IDX_H5_PATH).c_str(), H5P_DEFAULT);
  assert(dset >= 0);

  // make hyperslab selection
  fspace = H5Dget_space(dset);
  assert(fspace >= 0);
  ierr = H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL, &one, &block);
  assert(ierr >= 0);

  ierr = H5Dread(dset, NODE_IDX_H5_NATIVE_T, mspace, fspace, H5P_DEFAULT, &dst_idx[0]);
  assert(ierr >= 0);

  assert(H5Sclose(fspace) >= 0);
  assert(H5Dclose(dset) >= 0);
  assert(H5Sclose(mspace) >= 0);

  /***************************************************************************
   * read DST_PTR
   ***************************************************************************/

  // determine my read block of dst_ptr
  block = (hsize_t)(dst_blk_ptr.back() - dst_blk_ptr.front());
  start = (hsize_t)block_rebase;

  // allocate buffer and memory dataspace
  dst_ptr.resize(block);
  assert(dst_ptr.size() > 0);

  mspace = H5Screate_simple(1, &block, NULL);
  assert(mspace >= 0);
  ierr = H5Sselect_all(mspace);
  assert(ierr >= 0);

  dset = H5Dopen2(file, ngh5_prj_path(dsetname, DST_PTR_H5_PATH).c_str(), H5P_DEFAULT);
  assert(dset >= 0);

  // make hyperslab selection
  fspace = H5Dget_space(dset);
  assert(fspace >= 0);
  ierr = H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL, &one, &block);
  assert(ierr >= 0);

  ierr = H5Dread(dset, NODE_IDX_H5_NATIVE_T, mspace, fspace, H5P_DEFAULT, &dst_ptr[0]);
  assert(ierr >= 0);

  assert(H5Sclose(fspace) >= 0);
  assert(H5Dclose(dset) >= 0);
  assert(H5Sclose(mspace) >= 0);

  DST_PTR_T dst_rebase = dst_ptr[0];
  for (size_t i = 0; i < dst_ptr.size(); ++i)
    {
      dst_ptr[i] -= dst_rebase;
    }

  /***************************************************************************
   * read SRC_IDX
   ***************************************************************************/

  // determine my read block of dst_idx
  block = (hsize_t)(dst_ptr.back() - dst_ptr.front());
  start = (hsize_t)dst_rebase;

  // allocate buffer and memory dataspace
  src_idx.resize(block);
  assert(src_idx.size() > 0);

  mspace = H5Screate_simple(1, &block, NULL);
  assert(mspace >= 0);
  ierr = H5Sselect_all(mspace);
  assert(ierr >= 0);

  dset = H5Dopen2(file, ngh5_prj_path(dsetname, SRC_IDX_H5_PATH).c_str(), H5P_DEFAULT);
  assert(dset >= 0);

  // make hyperslab selection
  fspace = H5Dget_space(dset);
  assert(fspace >= 0);
  ierr = H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, NULL, &one, &block);
  assert(ierr >= 0);

  ierr = H5Dread(dset, NODE_IDX_H5_NATIVE_T, mspace, fspace, H5P_DEFAULT, &src_idx[0]);
  assert(ierr >= 0);

  assert(H5Sclose(fspace) >= 0);
  assert(H5Dclose(dset) >= 0);
  assert(H5Sclose(mspace) >= 0);

  assert(H5Fclose(file) >= 0);

  return ierr;
}
