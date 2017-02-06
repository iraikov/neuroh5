// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file iomodule.cc
///
///  Python module for reading edge information in DBS (Destination Block Sparse) format.
///
///  Copyright (C) 2016-2017 Project Neurograph.
//==============================================================================

#include "debug.hh"

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

#include "model_types.hh"
#include "read_dbs_projection.hh"
#include "population_reader.hh"
#include "read_graph.hh"
#include "write_graph.hh"
#include "graph_reader.hh"
#include "read_population.hh"
#include "projection_names.hh"

using namespace std;
using namespace ngh5;
using namespace ngh5::model;

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

  static PyObject *py_append_connections (PyObject *self, PyObject *args, PyObject *kwds)
  {
    int status; 
    PyObject *gid_values;
    const unsigned long default_cache_size = 4*1024*1024;
    const unsigned long default_chunk_size = 4000;
    const unsigned long default_value_chunk_size = 4000;
    unsigned long commptr;
    unsigned long chunk_size = default_chunk_size;
    unsigned long value_chunk_size = default_value_chunk_size;
    unsigned long cache_size = default_cache_size;
    char *file_name_arg, *prj_name_arg, *src_pop_arg, *dst_pop_arg;
    
    static const char *kwlist[] = {"commptr",
                                   "file_name",
                                   "src_pop_name",
                                   "dst_pop_name",
                                   "prj_name",
                                   "edges",
                                   "attributes",
                                   "chunk_size",
                                   "value_chunk_size",
                                   "cache_size",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "kssO|skkk", (char **)kwlist,
                                     &commptr, &file_name_arg, &pop_name_arg, &gid_values,
                                     &name_space_arg, &chunk_size, &value_chunk_size, &cache_size))
        return NULL;

    string file_name      = string(file_name_arg);
    string pop_name       = string(pop_name_arg);
    string attr_namespace = string(name_space_arg);
    
    int npy_type=0;
    
    vector<string> attr_names;
    vector<int> attr_types;
        
    vector< map<TREE_IDX_T, vector<uint32_t> >> all_attr_values_uint32;
    vector< map<TREE_IDX_T, vector<uint16_t> >> all_attr_values_uint16;
    vector< map<TREE_IDX_T, vector<uint8_t> >>  all_attr_values_uint8;
    vector< map<TREE_IDX_T, vector<float> >>  all_attr_values_float;
    

    create_value_maps(gid_values,
                      attr_names,
                      attr_types,
                      all_attr_values_uint32,
                      all_attr_values_uint16,
                      all_attr_values_uint8,
                      all_attr_values_float);

    size_t attr_idx=0;
    vector<size_t> attr_type_idx(AttrMap::num_attr_types);
    for(auto it = attr_names.begin(); it != attr_names.end(); ++it, attr_idx++) 
      {
        const string attr_name = *it;
        npy_type=attr_types[attr_idx];

        switch (npy_type)
          {
          case NPY_UINT32:
            {
              append_tree_attribute_map<uint32_t> (*((MPI_Comm *)(commptr)), file_name, attr_namespace, pop_name, 
                                                  attr_name, all_attr_values_uint32[attr_type_idx[AttrMap::attr_index_uint32]]);
              attr_type_idx[AttrMap::attr_index_uint32]++;
              break;
            }
          case NPY_UINT16:
            {
              append_tree_attribute_map<uint16_t> (*((MPI_Comm *)(commptr)), file_name, attr_namespace, pop_name, 
                                                  attr_name, all_attr_values_uint16[attr_type_idx[AttrMap::attr_index_uint16]]);
              attr_type_idx[AttrMap::attr_index_uint16]++;
              break;
            }
          case NPY_UINT8:
            {
              append_tree_attribute_map<uint8_t> (*((MPI_Comm *)(commptr)), file_name, attr_namespace, pop_name, 
                                                 attr_name, all_attr_values_uint8[attr_type_idx[AttrMap::attr_index_uint8]]);
              attr_type_idx[AttrMap::attr_index_uint8]++;
              break;
            }
          case NPY_FLOAT:
            {
              append_tree_attribute_map<float> (*((MPI_Comm *)(commptr)), file_name, attr_namespace, pop_name, 
                                                attr_name, all_attr_values_float[attr_type_idx[AttrMap::attr_index_float]]);
              attr_type_idx[AttrMap::attr_index_float]++;
              break;
            }
          default:
            throw runtime_error("Unsupported attribute type");
            break;
          }
      }
    
    return Py_None;
  }


  
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

    assert(io::hdf5::read_projection_names(*((MPI_Comm *)(commptr)), input_file_name, prj_names) >= 0);

    graph::read_graph(*((MPI_Comm *)(commptr)), std::string(input_file_name), true,
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

  
  static PyObject *py_scatter_read_graph (PyObject *self, PyObject *args)
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
    
    if (!PyArg_ParseTuple(args, "ksk", &commptr, &input_file_name, &io_size))
      return NULL;

    assert(MPI_Comm_size(*((MPI_Comm *)(commptr)), &size) >= 0);

    assert(io::hdf5::read_projection_names(*((MPI_Comm *)(commptr)), input_file_name, prj_names) >= 0);

    // Read population info to determine total_num_nodes
    assert(io::hdf5::read_population_ranges(*((MPI_Comm *)(commptr)), input_file_name, pop_ranges, pop_vector, total_num_nodes) >= 0);

    // Determine which nodes are assigned to which compute ranks
    node_rank_vector.resize(total_num_nodes);
    // round-robin node to rank assignment from file
    for (size_t i = 0; i < total_num_nodes; i++)
      {
        node_rank_vector[i] = i%size;
      }

    graph::scatter_graph(*((MPI_Comm *)(commptr)), std::string(input_file_name),
                         io_size, true, prj_names, node_rank_vector, prj_vector,
                         total_num_nodes, local_num_edges, total_num_edges);

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

  static PyObject *py_write_graph (PyObject *self, PyObject *args, PyObject *kwds)
  {
    PyObject *gid_values;
    unsigned long commptr;
    char *file_name_arg, *src_pop_name_arg, *dst_pop_name_arg, *prj_name_arg;
    
    static const char *kwlist[] = {"commptr",
                                   "file_name",
                                   "src_pop_name",
                                   "dst_pop_name",
                                   "prj_name",
                                   "edges",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "kssO|s", (char **)kwlist,
                                     &commptr, &file_name_arg,
                                     &src_pop_name_arg, &dst_pop_name_arg,
                                     &prj_name_arg, 
                                     &edge_values_arg,
                                     &attributes_arg))
        return NULL;

    string file_name = string(file_name_arg);
    string pop_name = string(pop_name_arg);
    string attr_namespace = string(name_space_arg);
    
    int npy_type=0;
    
    vector<string> attr_names;
    vector<int> attr_types;
        
    vector< map<TREE_IDX_T, vector<uint32_t> >> all_attr_values_uint32;
    vector< map<TREE_IDX_T, vector<uint16_t> >> all_attr_values_uint16;
    vector< map<TREE_IDX_T, vector<uint8_t> >>  all_attr_values_uint8;
    vector< map<TREE_IDX_T, vector<float> >>  all_attr_values_float;
    

    create_value_maps(gid_values,
                      attr_names,
                      attr_types,
                      all_attr_values_uint32,
                      all_attr_values_uint16,
                      all_attr_values_uint8,
                      all_attr_values_float);
  }
  
  
  static PyMethodDef module_methods[] = {
    { "read_graph", (PyCFunction)py_read_graph, METH_VARARGS,
      "Reads graph connectivity in Destination Block Sparse format." },
    { "scatter_read_graph", (PyCFunction)py_scatter_read_graph, METH_VARARGS,
      "Reads and scatters graph connectivity in Destination Block Sparse format." },
    { NULL, NULL, 0, NULL }
  };
}

PyMODINIT_FUNC
initio(void) {
  import_array();
  Py_InitModule3("io", module_methods, "HDF5 graph I/O module");
}

  

