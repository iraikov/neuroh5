#include "debug.hh"
#include "ngh5paths.h"
#include "ngh5types.hh"

#include "dbs_edge_reader.hh"
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
using namespace ngh5;

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




extern "C"
{

  static PyObject *py_read_graph (PyObject *self, PyObject *args)
  {
    vector<prj_tuple_t> prj_vector;
    vector<string> prj_names;
    PyObject *prj_dict = PyDict_New();
    unsigned long commptr;
    char *input_file_name;
    size_t total_num_edges = 0, local_num_edges = 0;

    if (!PyArg_ParseTuple(args, "ks", &commptr, &input_file_name))
      return NULL;

    assert(read_projection_names(MPI_COMM_WORLD, input_file_name, prj_names) >= 0);

    read_graph((void *)(long(commptr)), input_file_name, true,
               prj_names, prj_vector, local_num_edges, total_num_edges);
    
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
    { "read_graph", (PyCFunction)py_read_graph, METH_VARARGS, NULL },
    { NULL, NULL, 0, NULL }
  };
}

PyMODINIT_FUNC
initneurograph_reader(void) {
  import_array();
  Py_InitModule3("neurograph_reader", module_methods, "HDF5 graph reader");
}

  

