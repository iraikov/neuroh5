#include "debug.hh"
#include "ngh5paths.hh"
#include "ngh5types.hh"

#include "dbs_edge_reader.hh"
#include "population_reader.hh"
#include "attributes.hh"
#include "graph_reader.hh"

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
    int status;
    vector<prj_tuple_t> prj_vector;
    vector<string> prj_names;
    PyObject *py_prj_dict = PyDict_New();
    unsigned long commptr;
    char *input_file_name;
    size_t total_num_nodes, total_num_edges = 0, local_num_edges = 0;

    if (!PyArg_ParseTuple(args, "ks", &commptr, &input_file_name))
      return NULL;

    assert(read_projection_names(*((MPI_Comm *)(commptr)), input_file_name, prj_names) >= 0);

    read_graph(*((MPI_Comm *)(commptr)), std::string(input_file_name), true,
               prj_names, prj_vector,
               total_num_nodes, local_num_edges, total_num_edges);
    
    for (size_t i = 0; i < prj_vector.size(); i++)
      {
        const prj_tuple_t& prj = prj_vector[i];
        
        const vector<NODE_IDX_T>& src_vector = get<0>(prj);
        const vector<NODE_IDX_T>& dst_vector = get<1>(prj);
        const EdgeAttr&  edge_attr_values    = get<2>(prj);

        std::vector <PyObject*> py_float_edge_attrs;
        std::vector <PyObject*> py_uint8_edge_attrs;
        std::vector <PyObject*> py_uint16_edge_attrs;
        std::vector <PyObject*> py_uint32_edge_attrs;

        std::vector <float*> py_float_edge_attrs_ptr;
        std::vector <uint8_t*> py_uint8_edge_attrs_ptr;
        std::vector <uint16_t*> py_uint16_edge_attrs_ptr;
        std::vector <uint32_t*> py_uint32_edge_attrs_ptr;
        
        npy_intp dims[1], ind = 0;
        dims[0] = src_vector.size();
        
        PyObject *src_arr = PyArray_SimpleNew(1, dims, NPY_UINT32);
        PyObject *dst_arr = PyArray_SimpleNew(1, dims, NPY_UINT32);
        uint32_t *src_ptr = (uint32_t *)PyArray_GetPtr((PyArrayObject *)src_arr, &ind);
        uint32_t *dst_ptr = (uint32_t *)PyArray_GetPtr((PyArrayObject *)dst_arr, &ind);
        
        for (size_t j = 0; j < edge_attr_values.size_attr_vec<float>(); j++)
          {
            PyObject *arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_FLOAT);
            float *ptr = (float *)PyArray_GetPtr((PyArrayObject *)arr, &ind);
            py_float_edge_attrs.push_back(arr);
            py_float_edge_attrs_ptr.push_back(ptr);
          }
        for (size_t j = 0; j < edge_attr_values.size_attr_vec<uint8_t>(); j++)
          {
            PyObject *arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_UINT8);
            uint8_t *ptr = (uint8_t *)PyArray_GetPtr((PyArrayObject *)arr, &ind);
            py_uint8_edge_attrs.push_back(arr);
            py_uint8_edge_attrs_ptr.push_back(ptr);
          }
        for (size_t j = 0; j < edge_attr_values.size_attr_vec<uint16_t>(); j++)
          {
            PyObject *arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_UINT16);
            uint16_t *ptr = (uint16_t *)PyArray_GetPtr((PyArrayObject *)arr, &ind);
            py_uint16_edge_attrs.push_back(arr);
            py_uint16_edge_attrs_ptr.push_back(ptr);
          }
        for (size_t j = 0; j < edge_attr_values.size_attr_vec<uint32_t>(); j++)
          {
            PyObject *arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_UINT32);
            uint32_t *ptr = (uint32_t *)PyArray_GetPtr((PyArrayObject *)arr, &ind);
            py_uint32_edge_attrs.push_back(arr);
            py_uint32_edge_attrs_ptr.push_back(ptr);
          }
        
        
        for (size_t j = 0; j < src_vector.size(); j++)
          {
            src_ptr[j] = src_vector[j];
            dst_ptr[j] = dst_vector[j];
            for (size_t k = 0; k < edge_attr_values.size_attr_vec<float>(); k++)
              {
                py_float_edge_attrs_ptr[k][j] = edge_attr_values.at<float>(k,j); 
              }
            for (size_t k = 0; k < edge_attr_values.size_attr_vec<uint8_t>(); k++)
              {
                py_uint8_edge_attrs_ptr[k][j] = edge_attr_values.at<uint8_t>(k,j); 
              }
            for (size_t k = 0; k < edge_attr_values.size_attr_vec<uint16_t>(); k++)
              {
                py_uint16_edge_attrs_ptr[k][j] = edge_attr_values.at<uint16_t>(k,j); 
              }
            for (size_t k = 0; k < edge_attr_values.size_attr_vec<uint32_t>(); k++)
              {
                py_uint32_edge_attrs_ptr[k][j] = edge_attr_values.at<uint32_t>(k,j); 
              }
          }
        
        PyObject *py_prjval  = PyList_New(0);
        status = PyList_Append(py_prjval, src_arr);
        assert (status == 0);
        status = PyList_Append(py_prjval, dst_arr);
        assert (status == 0);
        for (size_t j = 0; j < edge_attr_values.size_attr_vec<float>(); j++)
          {
            status = PyList_Append(py_prjval, py_float_edge_attrs[j]);
            assert(status == 0);
          }
        for (size_t j = 0; j < edge_attr_values.size_attr_vec<uint8_t>(); j++)
          {
            status = PyList_Append(py_prjval, py_uint8_edge_attrs[j]);
            assert(status == 0);
          }
        for (size_t j = 0; j < edge_attr_values.size_attr_vec<uint16_t>(); j++)
          {
            status = PyList_Append(py_prjval, py_uint16_edge_attrs[j]);
            assert(status == 0);
          }
        for (size_t j = 0; j < edge_attr_values.size_attr_vec<uint32_t>(); j++)
          {
            status = PyList_Append(py_prjval, py_uint32_edge_attrs[j]);
            assert(status == 0);
          }
        
        PyDict_SetItemString(py_prj_dict, prj_names[i].c_str(), py_prjval);
        
      }

    return py_prj_dict;
  }

  
  static PyObject *py_scatter_graph (PyObject *self, PyObject *args)
  {
    int status;
    // A vector that maps nodes to compute ranks
    vector<rank_t> node_rank_vector;
    vector < edge_map_t > prj_vector;
    vector<pop_range_t> pop_vector;
    map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
    vector<string> prj_names;
    PyObject *py_prj_dict = PyDict_New();
    unsigned long commptr; unsigned long io_size; int size;
    char *input_file_name;
    size_t total_num_nodes, total_num_edges = 0, local_num_edges = 0;

    MPI_Comm_size(*((MPI_Comm *)(commptr)), &size);
    
    if (!PyArg_ParseTuple(args, "ksk", &commptr, &input_file_name, io_size))
      return NULL;

    assert(read_projection_names(*((MPI_Comm *)(commptr)), input_file_name, prj_names) >= 0);

    // Read population info to determine total_num_nodes
    assert(read_population_ranges(*((MPI_Comm *)(commptr)), input_file_name, pop_ranges, pop_vector, total_num_nodes) >= 0);
    
    // Determine which nodes are assigned to which compute ranks
    node_rank_vector.resize(total_num_nodes);
    // round-robin node to rank assignment from file
    for (size_t i = 0; i < total_num_nodes; i++)
      {
        node_rank_vector[i] = i%size;
    }

    scatter_graph(*((MPI_Comm *)(commptr)), std::string(input_file_name),
                  io_size, true, prj_names, node_rank_vector, prj_vector,
                  total_num_nodes);
    
    for (size_t i = 0; i < prj_vector.size(); i++)
      {
        PyObject *py_edge_dict = PyDict_New();
        edge_map_t prj_edge_map = prj_vector[i];
        
        if (prj_edge_map.size() > 0)
          {
            for (auto it = prj_edge_map.begin(); it != prj_edge_map.end(); it++)
              {
                NODE_IDX_T dst   = it->first;
                edge_tuple_t& et = it->second;
                
                std::vector <PyObject*> py_float_edge_attrs;
                std::vector <PyObject*> py_uint8_edge_attrs;
                std::vector <PyObject*> py_uint16_edge_attrs;
                std::vector <PyObject*> py_uint32_edge_attrs;
                
                std::vector <float*> py_float_edge_attrs_ptr;
                std::vector <uint8_t*> py_uint8_edge_attrs_ptr;
                std::vector <uint16_t*> py_uint16_edge_attrs_ptr;
                std::vector <uint32_t*> py_uint32_edge_attrs_ptr;
                
                vector<NODE_IDX_T> src_vector = get<0>(et);
                const EdgeAttr&   edge_attr_values = get<1>(et);

                npy_intp dims[1], ind = 0;
                dims[0] = src_vector.size();
                
                PyObject *src_arr = PyArray_SimpleNew(1, dims, NPY_UINT32);
                uint32_t *src_ptr = (uint32_t *)PyArray_GetPtr((PyArrayObject *)src_arr, &ind);

                for (size_t j = 0; j < edge_attr_values.size_attr_vec<float>(); j++)
                  {
                    PyObject *arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_FLOAT);
                    float *ptr = (float *)PyArray_GetPtr((PyArrayObject *)arr, &ind);
                    py_float_edge_attrs.push_back(arr);
                    py_float_edge_attrs_ptr.push_back(ptr);
                  }
                for (size_t j = 0; j < edge_attr_values.size_attr_vec<uint8_t>(); j++)
                  {
                    PyObject *arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_UINT8);
                    uint8_t *ptr = (uint8_t *)PyArray_GetPtr((PyArrayObject *)arr, &ind);
                    py_uint8_edge_attrs.push_back(arr);
                    py_uint8_edge_attrs_ptr.push_back(ptr);
                  }
                for (size_t j = 0; j < edge_attr_values.size_attr_vec<uint16_t>(); j++)
                  {
                    PyObject *arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_UINT16);
                    uint16_t *ptr = (uint16_t *)PyArray_GetPtr((PyArrayObject *)arr, &ind);
                    py_uint16_edge_attrs.push_back(arr);
                    py_uint16_edge_attrs_ptr.push_back(ptr);
                  }
                for (size_t j = 0; j < edge_attr_values.size_attr_vec<uint32_t>(); j++)
                  {
                    PyObject *arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_UINT32);
                    uint32_t *ptr = (uint32_t *)PyArray_GetPtr((PyArrayObject *)arr, &ind);
                    py_uint32_edge_attrs.push_back(arr);
                    py_uint32_edge_attrs_ptr.push_back(ptr);
                  }
                        
                for (size_t j = 0; j < src_vector.size(); j++)
                  {
                    src_ptr[j] = src_vector[j];
                    for (size_t k = 0; k < edge_attr_values.size_attr_vec<float>(); k++)
                      {
                        py_float_edge_attrs_ptr[k][j] = edge_attr_values.at<float>(k,j); 
                      }
                    for (size_t k = 0; k < edge_attr_values.size_attr_vec<uint8_t>(); k++)
                      {
                        py_uint8_edge_attrs_ptr[k][j] = edge_attr_values.at<uint8_t>(k,j); 
                      }
                    for (size_t k = 0; k < edge_attr_values.size_attr_vec<uint16_t>(); k++)
                      {
                        py_uint16_edge_attrs_ptr[k][j] = edge_attr_values.at<uint16_t>(k,j); 
                      }
                    for (size_t k = 0; k < edge_attr_values.size_attr_vec<uint32_t>(); k++)
                      {
                        py_uint32_edge_attrs_ptr[k][j] = edge_attr_values.at<uint32_t>(k,j); 
                      }
                    
                  }
                
                PyObject *py_edgeval  = PyList_New(0);
                status = PyList_Append(py_edgeval, src_arr);
                assert (status == 0);
                for (size_t j = 0; j < edge_attr_values.size_attr_vec<float>(); j++)
                  {
                    status = PyList_Append(py_edgeval, py_float_edge_attrs[j]);
                    assert(status == 0);
                  }
                for (size_t j = 0; j < edge_attr_values.size_attr_vec<uint8_t>(); j++)
                  {
                    status = PyList_Append(py_edgeval, py_uint8_edge_attrs[j]);
                    assert(status == 0);
                  }
                for (size_t j = 0; j < edge_attr_values.size_attr_vec<uint16_t>(); j++)
                  {
                    status = PyList_Append(py_edgeval, py_uint16_edge_attrs[j]);
                    assert(status == 0);
                  }
                for (size_t j = 0; j < edge_attr_values.size_attr_vec<uint32_t>(); j++)
                  {
                    status = PyList_Append(py_edgeval, py_uint32_edge_attrs[j]);
                    assert(status == 0);
                  }
                
                PyObject *key = PyInt_FromLong(dst);
                PyDict_SetItem(py_edge_dict, key, py_edgeval);
              }
          }
        
         PyDict_SetItemString(py_prj_dict, prj_names[i].c_str(), py_edge_dict);
        
      }

    return py_prj_dict;
  }

  
  static PyMethodDef module_methods[] = {
    { "read_graph", (PyCFunction)py_read_graph, METH_VARARGS, NULL },
    { "scatter_graph", (PyCFunction)py_scatter_graph, METH_VARARGS, NULL },
    { NULL, NULL, 0, NULL }
  };
}

PyMODINIT_FUNC
initreader(void) {
  import_array();
  Py_InitModule3("reader", module_methods, "HDF5 graph reader");
}

  

