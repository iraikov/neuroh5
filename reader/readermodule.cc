#include "debug.hh"
#include "ngh5paths.h"
#include "ngh5types.hh"

#include "dbs_graph_reader.hh"
#include "population_reader.hh"
#include "edge_reader.hh"

#include <Python.h>
#include <numpy/numpyconfig.h>
#include <numpy/arrayobject.h>

#include <getopt.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <vector>

#include <hdf5.h>
#include <mpi.h>
#include <algorithm>

#undef NDEBUG
#include <cassert>

using namespace std;

void throw_err(char const* err_message)
{
  fprintf(stderr, "Error: %s\n", err_message);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

void throw_err(char const* err_message, int32_t task)
{
  fprintf(stderr, "Task %d Error: %s\n", task, err_message);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

void throw_err(char const* err_message, int32_t task, int32_t thread)
{
  fprintf(stderr, "Task %d Thread %d Error: %s\n", task, thread, err_message);
  MPI_Abort(MPI_COMM_WORLD, 1);
}


void print_usage_full(char** argv)
{
  printf("Usage: %s [graphfile] [options]\n\n", argv[0]);
  printf("Options:\n");
  printf("\t-s:\n");
  printf("\t\tPrint only edge summary\n");
}




/*****************************************************************************
 * Append src/dst node indices to a vector of edges
 *****************************************************************************/

int append_prj_list
(
 const NODE_IDX_T&         dst_start,
 const NODE_IDX_T&         src_start,
 const vector<DST_BLK_PTR_T>&  dst_blk_ptr,
 const vector<NODE_IDX_T>& dst_idx,
 const vector<DST_PTR_T>&  dst_ptr,
 const vector<NODE_IDX_T>& src_idx,
 const vector<float>&      longitudinal_distance,
 const vector<float>&      transverse_distance,
 const vector<float>&      distance,
 const vector<float>&      synaptic_weight,
 const vector<uint16_t>&   segment_index,
 const vector<uint16_t>&   segment_point_index,
 const vector<uint8_t>&    layer,
 size_t&                   num_edges,
 vector<prj_tuple_t>&      prj_list
 )
{
  int ierr = 0; size_t dst_ptr_size;
  num_edges = 0;
  vector<NODE_IDX_T> src_vec, dst_vec;
  
  
  if (dst_blk_ptr.size() > 0) 
    {
      dst_ptr_size = dst_ptr.size();
      for (size_t b = 0; b < dst_blk_ptr.size()-1; ++b)
        {
          size_t low_dst_ptr = dst_blk_ptr[b], high_dst_ptr = dst_blk_ptr[b+1];
          NODE_IDX_T dst_base = dst_idx[b];
          for (size_t i = low_dst_ptr, ii = 0; i < high_dst_ptr; ++i, ++ii)
            {
              if (i < dst_ptr_size-1) 
                {
                  NODE_IDX_T dst = dst_base + ii + dst_start;
                  size_t low = dst_ptr[i], high = dst_ptr[i+1];
                  for (size_t j = low; j < high; ++j)
                    {
                      NODE_IDX_T src = src_idx[j] + src_start;
                      src_vec.push_back(src);
                      dst_vec.push_back(dst);
		      num_edges++;
                    }
                }
            }
        }
    }

  prj_list.push_back(make_tuple(src_vec, dst_vec,
                                longitudinal_distance,
                                transverse_distance,
                                distance,
                                synaptic_weight,
                                segment_index,
                                segment_point_index,
                                layer));

  return ierr;
}




/*****************************************************************************
 * Read edge attributes
 *****************************************************************************/

int read_all_edge_attributes
(
 MPI_Comm comm,
 const char *input_file_name,
 const char *prj_name,
 const DST_PTR_T edge_base,
 const DST_PTR_T edge_count,
 const vector<string>&      edge_attr_names,
 vector<float>&      longitudinal_distance,
 vector<float>&      transverse_distance,
 vector<float>&      distance,
 vector<float>&      synaptic_weight,
 vector<uint16_t>&   segment_index,
 vector<uint16_t>&   segment_point_index,
 vector<uint8_t>&    layer
 )
{
  int ierr = 0; 
  vector<NODE_IDX_T> src_vec, dst_vec;

  for (size_t j = 0; j < edge_attr_names.size(); j++)
    {
      if (edge_attr_names[j].compare(string("Longitudinal Distance")) == 0)
        {
          assert(read_edge_attributes<float>(comm,
                                             input_file_name,
                                             prj_name,
                                             "Longitudinal Distance",
                                             edge_base, edge_count,
                                             LONG_DISTANCE_H5_NATIVE_T,
                                             longitudinal_distance) >= 0);
          continue;
        }
      if (edge_attr_names[j].compare(string("Transverse Distance")) == 0)
        {
          assert(read_edge_attributes<float>(comm,
                                             input_file_name,
                                             prj_name,
                                             "Transverse Distance",
                                             edge_base, edge_count,
                                             TRANS_DISTANCE_H5_NATIVE_T,
                                             transverse_distance)  >= 0);
          continue;
        }
      if (edge_attr_names[j].compare(string("Distance")) == 0)
        {
          assert(read_edge_attributes<float>(comm,
                                             input_file_name,
                                             prj_name,
                                             "Distance",
                                             edge_base, edge_count,
                                             DISTANCE_H5_NATIVE_T,
                                             distance)  >= 0);
          continue;
        }
      if (edge_attr_names[j].compare(string("Segment Index")) == 0)
        {
          assert(read_edge_attributes<uint16_t>(comm,
                                                input_file_name,
                                                prj_name,
                                                "Segment Index",
                                                edge_base, edge_count,
                                                SEGMENT_INDEX_H5_NATIVE_T,
                                                segment_index) >= 0);
          continue;
        }
      if (edge_attr_names[j].compare(string("Segment Point Index")) == 0)
        {
          assert(read_edge_attributes<uint16_t>(comm,
                                                input_file_name,
                                                prj_name,
                                                "Segment Point Index",
                                                edge_base, edge_count,
                                                SEGMENT_POINT_INDEX_H5_NATIVE_T,
                                                segment_point_index) >= 0);
          continue;
        }
      if (edge_attr_names[j].compare(string("Layer")) == 0)
        {
          assert(read_edge_attributes<uint8_t>(comm,
                                               input_file_name,
                                               prj_name,
                                               "Layer",
                                               edge_base, edge_count,
                                               LAYER_H5_NATIVE_T,
                                               layer) >= 0);
          continue;
        }

    }
  return ierr;
}

/*****************************************************************************
 * Main driver
 *****************************************************************************/

int read_edges(void *commptr,
               const char *input_file_name,
               const bool opt_attrs,
               vector<string> &prj_names,
               vector<prj_tuple_t> &prj_list)
{
  int rank;

  MPI_Comm comm = *((MPI_Comm*)commptr);
  
  MPI_Comm_rank(comm, &rank);

  // read the population info
  set< pair<pop_t, pop_t> > pop_pairs;
  assert(read_population_combos(comm, input_file_name, pop_pairs) >= 0);

  vector<pop_range_t> pop_vector;
  map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;

  assert(read_population_ranges(comm, input_file_name, pop_ranges, pop_vector) >= 0);
  assert(read_projection_names(comm, input_file_name, prj_names) >= 0);

  printf("Task %d: total number of projections is %lu\n", rank,  prj_names.size());
 
  size_t total_num_edges = 0, local_num_edges = 0;
  
  // read the edges
  for (size_t i = 0; i < prj_names.size(); i++)
    {
      size_t local_prj_num_edges = 0, total_prj_num_edges = 0;
      DST_BLK_PTR_T block_base;
      DST_PTR_T edge_base, edge_count = 0;
      NODE_IDX_T dst_start, src_start;
      vector<DST_BLK_PTR_T> dst_blk_ptr;
      vector<NODE_IDX_T> dst_idx;
      vector<DST_PTR_T> dst_ptr;
      vector<NODE_IDX_T> src_idx;
      vector<string> edge_attr_names;
      vector<float> longitudinal_distance;
      vector<float> transverse_distance;
      vector<float> distance;
      vector<float> synaptic_weight;
      vector<uint16_t> segment_index;
      vector<uint16_t> segment_point_index;
      vector<uint8_t> layer;
      
      printf("Task %d reading projection %lu (%s)\n", rank, i, prj_names[i].c_str());

      assert(read_dbs_projection(comm, input_file_name, prj_names[i].c_str(), 
                                 pop_vector, dst_start, src_start, total_prj_num_edges, block_base, edge_base,
                                 dst_blk_ptr, dst_idx, dst_ptr, src_idx) >= 0);
      
      // validate the edges
      assert(validate_edge_list(dst_start, src_start, dst_blk_ptr, dst_idx, dst_ptr, src_idx, pop_ranges, pop_pairs) == true);
      
      if (opt_attrs)
        {
          assert(read_edge_attribute_names(comm, input_file_name, prj_names[i].c_str(), edge_attr_names) >= 0);

          assert(read_all_edge_attributes(comm, input_file_name, prj_names[i].c_str(), edge_base, edge_count,
                                          edge_attr_names, longitudinal_distance, transverse_distance, distance,
                                          synaptic_weight, segment_index, segment_point_index, layer) >= 0);
        }

      // append to the vectors representing a projection (sources, destinations, edge attributes)
      assert(append_prj_list(dst_start, src_start, dst_blk_ptr, dst_idx, dst_ptr, src_idx, 
                             longitudinal_distance, transverse_distance, distance,
                             synaptic_weight, segment_index, segment_point_index, layer,
                             local_prj_num_edges, prj_list) >= 0);


      // ensure that all edges in the projection have been read and appended to edge_list
      assert(local_prj_num_edges == src_idx.size());

      printf("Task %d has read %lu edges in projection %lu (%s)\n", rank,  local_prj_num_edges, i, prj_names[i].c_str());

      total_num_edges = total_num_edges + total_prj_num_edges;
      local_num_edges = local_num_edges + local_prj_num_edges;

    }

  printf("Task %d has read a total of %lu projections\n", rank,  prj_list.size());
  printf("Task %d has read a total of %lu edges\n", rank,  local_num_edges);
  printf("Task %d: total number of edges is %lu\n", rank,  total_num_edges);
  
  size_t sum_local_num_edges = 0;
  MPI_Reduce(&local_num_edges, &sum_local_num_edges, 1,
	     MPI_INT64_T, MPI_SUM, 0, comm);
  if (rank == 0)
    {
      assert(sum_local_num_edges == total_num_edges);
    }


  return 0;
}

extern "C"
{

  static PyObject *py_read_edges (PyObject *self, PyObject *args)
  {
    vector<prj_tuple_t> prj_vector;
    vector<string> prj_names;
    PyObject *prj_dict = PyDict_New();
    unsigned long commptr;
    char *input_file_name;
    
    if (!PyArg_ParseTuple(args, "ks", &commptr, &input_file_name))
      return NULL;

    read_edges((void *)(long(commptr)), input_file_name, true, prj_names, prj_vector);
    
    for (size_t i = 0; i < prj_vector.size(); i++)
      {
        const prj_tuple_t& prj = prj_vector[i];
        
        const vector<NODE_IDX_T>& src_vector = get<0>(prj);
        const vector<NODE_IDX_T>& dst_vector = get<1>(prj);
        
        const vector<float>&      longitudinal_distance = get<2>(prj);
        const vector<float>&      transverse_distance   = get<3>(prj);
        const vector<float>&      distance              = get<4>(prj);
        const vector<float>&      synaptic_weight       = get<5>(prj);
        const vector<uint16_t>&   segment_index         = get<6>(prj);
        const vector<uint16_t>&   segment_point_index   = get<7>(prj);
        const vector<uint8_t>&    layer                 = get<8>(prj);
        
        bool has_longitudinal_distance = longitudinal_distance.size() > 0;
        bool has_transverse_distance   = transverse_distance.size() > 0;
        bool has_distance              = distance.size() > 0;
        bool has_synaptic_weight       = synaptic_weight.size() > 0;
        bool has_segment_index         = segment_index.size() > 0;
        bool has_segment_point_index   = segment_point_index.size() > 0;
        bool has_layer                 = layer.size() > 0;
        
        npy_intp dims[1];
        dims[0] = src_vector.size();

        PyObject *src_arr = PyArray_SimpleNew(1, dims, NPY_UINT32);
        PyObject *dst_arr = PyArray_SimpleNew(1, dims, NPY_UINT32);

        dims[0] = longitudinal_distance.size();
        PyObject *longitudinal_distance_arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_FLOAT);
        dims[0] = transverse_distance.size();
        PyObject *transverse_distance_arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_FLOAT);
        dims[0] = distance.size();
        PyObject *distance_arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_FLOAT);
        dims[0] = synaptic_weight.size();
        PyObject *synaptic_weight_arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_FLOAT);
        dims[0] = segment_index.size();
        PyObject *segment_index_arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_UINT16);
        dims[0] = segment_point_index.size();
        PyObject *segment_point_index_arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_UINT16);
        dims[0] = layer.size();
        PyObject *layer_arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_UINT8);

        npy_intp ind = 0;
        uint32_t *src_ptr = (uint32_t *)PyArray_GetPtr((PyArrayObject *)src_arr, &ind);
        uint32_t *dst_ptr = (uint32_t *)PyArray_GetPtr((PyArrayObject *)dst_arr, &ind);
        float *longitudinal_distance_ptr = (float *)PyArray_GetPtr((PyArrayObject *)longitudinal_distance_arr, &ind);
        float *transverse_distance_ptr = (float *)PyArray_GetPtr((PyArrayObject *)transverse_distance_arr, &ind);
        float *distance_ptr = (float *)PyArray_GetPtr((PyArrayObject *)distance_arr, &ind);
        float *synaptic_weight_ptr = (float *)PyArray_GetPtr((PyArrayObject *)synaptic_weight_arr, &ind);
        uint16_t *segment_index_ptr = (uint16_t *)PyArray_GetPtr((PyArrayObject *)segment_index_arr, &ind);
        uint16_t *segment_point_index_ptr = (uint16_t *)PyArray_GetPtr((PyArrayObject *)segment_point_index_arr, &ind);
        uint8_t *layer_ptr = (uint8_t *)PyArray_GetPtr((PyArrayObject *)layer_arr, &ind);
        
        
        for (size_t j = 0; j < src_vector.size(); j++)
          {
            src_ptr[j] = src_vector[j];
            dst_ptr[j] = dst_vector[j];
            if (has_longitudinal_distance)
              longitudinal_distance_ptr[j] = longitudinal_distance[j];
            if (has_transverse_distance)
              transverse_distance_ptr[j] = transverse_distance[j];
            if (has_distance)
              distance_ptr[j] = distance[j];
            if (has_synaptic_weight)
              synaptic_weight_ptr[j] = synaptic_weight[j];
            if (has_segment_index)
              segment_index_ptr[j] = segment_index[j];
            if (has_segment_point_index)
              segment_point_index_ptr[j] = segment_point_index[j];
            if (has_layer)
              layer_ptr[j] = layer[j];
          }
        
        PyObject *prjval = PyTuple_Pack(9,
                                        src_arr,
                                        dst_arr,
                                        longitudinal_distance_arr,
                                        transverse_distance_arr,
                                        distance_arr,
                                        synaptic_weight_arr,
                                        segment_index_arr,
                                        segment_point_index_arr,
                                        layer_arr);
        
        PyDict_SetItemString(prj_dict, prj_names[i].c_str(), prjval);
        
        
      }

    return prj_dict;
  }

  static PyMethodDef module_methods[] = {
    { "read_edges", (PyCFunction)py_read_edges, METH_VARARGS, NULL },
    { NULL, NULL, 0, NULL }
  };
}

PyMODINIT_FUNC
initneurograph_reader(void) {
  import_array();
  Py_InitModule3("neurograph_reader", module_methods, "HDF5 graph reader");
}

  

