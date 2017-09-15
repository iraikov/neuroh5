// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file iomodule.cc
///
///  Python module for reading and writing neuronal connectivity and morphological information.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================

#include "debug.hh"

#define MPICH_SKIP_MPICXX 1
#include <Python.h>
#include <bytesobject.h>
#include <mpi4py/mpi4py.h>
#include <numpy/numpyconfig.h>
#include <numpy/arrayobject.h>

#include <getopt.h>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <unistd.h>

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

#include "neuroh5_types.hh"
#include "cell_populations.hh"
#include "cell_attributes.hh"
#include "path_names.hh"
#include "create_file_toplevel.hh"
#include "read_tree.hh"
#include "append_tree.hh"
#include "scatter_read_tree.hh"
#include "cell_index.hh"
#include "dataset_num_elements.hh"
#include "num_projection_blocks.hh"
#include "attr_map.hh"

#include "read_projection.hh"
#include "read_graph.hh"
#include "scatter_read_graph.hh"
#include "bcast_graph.hh"
#include "write_graph.hh"
#include "append_graph.hh"
#include "projection_names.hh"
#include "edge_attributes.hh"

using namespace std;
using namespace neuroh5;
using namespace neuroh5::data;

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

  
void build_node_rank_map (PyObject *py_node_rank_map,
                          map<NODE_IDX_T, rank_t>& node_rank_map)
{
  PyObject *idx_key, *idx_value;
  Py_ssize_t map_pos = 0;
  
  while (PyDict_Next(py_node_rank_map, &map_pos, &idx_key, &idx_value))
    {
      NODE_IDX_T idx = PyLong_AsLong(idx_key);
      rank_t rank = PyLong_AsLong(idx_value);
      node_rank_map.insert(make_pair(idx,rank));
    }
}
  


template<class T>
void py_array_to_vector (PyObject *pyval,
                         vector<T>& value_vector)
{
  npy_intp *dims, ind = 0;
  assert(PyArray_Check(pyval));
  PyArrayObject* pyarr = (PyArrayObject*)PyArray_FROM_OTF(pyval, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
  T *pyarr_ptr = (T *)PyArray_GetPtr(pyarr, &ind);
  dims = PyArray_DIMS(pyarr);
  assert(dims != NULL);
  size_t value_size = dims[0];
  value_vector.resize(value_size);
  for (size_t j=0; j<value_size; j++)
    {
      value_vector[j] = pyarr_ptr[j];
    }
  Py_DECREF(pyarr);
}


template<class T>
void append_value_map (CELL_IDX_T idx,
                       PyObject *pyval,
                       size_t attr_pos,
                       map<CELL_IDX_T, vector<T> >& all_attr_values)
{
  npy_intp *dims, ind = 0;
  assert(PyArray_Check(pyval));
  PyArrayObject* pyarr = (PyArrayObject*)PyArray_FROM_OTF(pyval, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
  dims = PyArray_DIMS(pyarr);
  assert(dims != NULL);
  size_t value_size = dims[0];
  T *pyarr_ptr = (T *)PyArray_GetPtr(pyarr, &ind);
  vector<T> attr_values(value_size);
  for (size_t j=0; j<value_size; j++)
    {
      attr_values[j] = pyarr_ptr[j];
    }
  all_attr_values.insert(make_pair(idx, attr_values));
  Py_DECREF(pyarr);
}


void build_cell_attr_value_maps (PyObject *idx_values,
                                 vector<string>& attr_names,
                                 vector<int>& attr_types,
                                 vector<map<CELL_IDX_T, vector<uint32_t>>>& all_attr_values_uint32,
                                 vector<map<CELL_IDX_T, vector<uint16_t>>>& all_attr_values_uint16,
                                 vector<map<CELL_IDX_T, vector<uint8_t>>>& all_attr_values_uint8,
                                 vector<map<CELL_IDX_T, vector<int32_t>>>& all_attr_values_int32,
                                 vector<map<CELL_IDX_T, vector<int16_t>>>& all_attr_values_int16,
                                 vector<map<CELL_IDX_T, vector<int8_t>>>& all_attr_values_int8,
                                 vector<map<CELL_IDX_T, vector<float>>>& all_attr_values_float)
{
  PyObject *idx_key, *idx_value;
  Py_ssize_t idx_pos = 0;
  int npy_type=0;
  
  while (PyDict_Next(idx_values, &idx_pos, &idx_key, &idx_value))
    {
      assert(idx_key != Py_None);
      assert(idx_value != Py_None);

      CELL_IDX_T idx = PyLong_AsLong(idx_key);

      PyObject *attr_key, *attr_values;
      Py_ssize_t attr_pos = 0;
      size_t attr_idx = 0;
      vector<size_t> attr_type_idx(AttrMap::num_attr_types);
        
      while (PyDict_Next(idx_value, &attr_pos, &attr_key, &attr_values))
        {
          assert(attr_key != Py_None);
          assert(attr_values != Py_None);

          npy_type = PyArray_TYPE((PyArrayObject *)attr_values);
          if (attr_names.size() < (size_t)attr_idx+1)
            {
              string attr_name = string(PyBytes_AsString(attr_key));
              attr_names.push_back(attr_name);
              attr_types.push_back(npy_type);
            }
          else
            {
              assert(attr_names[attr_idx] == string(PyBytes_AsString(attr_key)));
              assert(attr_types[attr_idx] == npy_type);
            }
            
          switch (npy_type)
            {
            case NPY_UINT32:
              {
                if (all_attr_values_uint32.size() < (size_t)attr_type_idx[AttrMap::attr_index_uint32]+1)
                  {
                    all_attr_values_uint32.resize(attr_type_idx[AttrMap::attr_index_uint32]+1);
                  }
                append_value_map<uint32_t> (idx, attr_values, attr_idx,
                                            all_attr_values_uint32[attr_type_idx[AttrMap::attr_index_uint32]]);
                attr_type_idx[AttrMap::attr_index_uint32]++;
                break;
              }
            case NPY_INT32:
              {
                if (all_attr_values_int32.size() < (size_t)attr_type_idx[AttrMap::attr_index_int32]+1)
                  {
                    all_attr_values_int32.resize(attr_type_idx[AttrMap::attr_index_int32]+1);
                  }
                append_value_map<int32_t> (idx, attr_values, attr_idx,
                                           all_attr_values_int32[attr_type_idx[AttrMap::attr_index_int32]]);
                attr_type_idx[AttrMap::attr_index_int32]++;
                break;
              }
            case NPY_UINT16:
              {
                if (all_attr_values_uint16.size() < (size_t)attr_type_idx[AttrMap::attr_index_uint16]+1)
                  {
                    all_attr_values_uint16.resize(attr_type_idx[AttrMap::attr_index_uint16]+1);
                  }
                append_value_map<uint16_t> (idx, attr_values, attr_idx,
                                            all_attr_values_uint16[attr_type_idx[AttrMap::attr_index_uint16]]);
                attr_type_idx[AttrMap::attr_index_uint16]++;
                break;
              }
            case NPY_INT16:
              {
                if (all_attr_values_int16.size() < (size_t)attr_type_idx[AttrMap::attr_index_int16]+1)
                  {
                    all_attr_values_int16.resize(attr_type_idx[AttrMap::attr_index_int16]+1);
                  }
                append_value_map<int16_t> (idx, attr_values, attr_idx,
                                           all_attr_values_int16[attr_type_idx[AttrMap::attr_index_int16]]);
                attr_type_idx[AttrMap::attr_index_int16]++;
                break;
              }
            case NPY_UINT8:
              {
                if (all_attr_values_uint8.size() < (size_t)attr_type_idx[AttrMap::attr_index_uint8]+1)
                  {
                    all_attr_values_uint8.resize(attr_type_idx[AttrMap::attr_index_uint8]+1);
                  }
                append_value_map<uint8_t> (idx, attr_values, attr_idx,
                                           all_attr_values_uint8[attr_type_idx[AttrMap::attr_index_uint8]]);
                attr_type_idx[AttrMap::attr_index_uint8]++;
                break;
              }
            case NPY_INT8:
              {
                if (all_attr_values_int8.size() < (size_t)attr_type_idx[AttrMap::attr_index_int8]+1)
                  {
                    all_attr_values_int8.resize(attr_type_idx[AttrMap::attr_index_int8]+1);
                  }
                append_value_map<int8_t> (idx, attr_values, attr_idx,
                                          all_attr_values_int8[attr_type_idx[AttrMap::attr_index_int8]]);
                attr_type_idx[AttrMap::attr_index_int8]++;
                break;
              }
            case NPY_FLOAT:
              {
                if (all_attr_values_float.size() < (size_t)attr_type_idx[AttrMap::attr_index_float]+1)
                  {
                    all_attr_values_float.resize(attr_type_idx[AttrMap::attr_index_float]+1);
                  }
                append_value_map<float> (idx, attr_values, attr_idx,
                                         all_attr_values_float[attr_type_idx[AttrMap::attr_index_float]]);
                attr_type_idx[AttrMap::attr_index_float]++;
                break;
              }
            default:
              throw runtime_error("Unsupported attribute type");
              break;
            }
          attr_idx = attr_idx+1;
        }
    }
}



void build_edge_map (PyObject *py_edge_values,
                     map <string, vector< vector <string> > >& attr_names,
                     edge_map_t& edge_map)
{
  PyObject *py_edge_key, *py_edge_value;
  Py_ssize_t edge_pos = 0;
  int npy_type=0;
  
  while (PyDict_Next(py_edge_values, &edge_pos, &py_edge_key, &py_edge_value))
    {
      assert(edge_key != Py_None);
      assert(edge_value != Py_None);

      NODE_IDX_T node_idx = PyLong_AsLong(py_edge_key);

      PyObject *py_attr_namespaces = PyTuple_GetItem(py_edge_value, 1);
      PyObject *py_adj_values = PyTuple_GetItem(py_edge_value, 0);

      Py_ssize_t attr_namespace_pos = 0;
      PyObject *py_attr_namespace, *py_attr_namespace_value;
      map <string, data::AttrVal> edge_attr_map;

      vector<NODE_IDX_T>  adj_values;

      npy_type = PyArray_TYPE((PyArrayObject *)py_adj_values);
      
      switch (npy_type)
        {
        case NPY_UINT32:
          {
            py_array_to_vector<NODE_IDX_T> (py_attr_values, adj_values);
            break;
          }
        default:
          throw runtime_error("Unsupported source vertex type");
          break;
        }
      
      while (PyDict_Next(py_attr_namespaces, &attr_namespace_pos, &py_attr_namespace, &py_attr_namespace_value))
        {
          vector<size_t> attr_type_idx(AttrMap::num_attr_types);
          PyObject *py_attr_key, *py_attr_values;
          Py_ssize_t attr_pos = 0;
          size_t attr_idx = 0;

          char *str = PyBytes_AsString (py_attr_namespace);
          string attr_namespace = string(str);
          
          attr_names[attr_namespace].resize(AttrMap::num_attr_types);

          vector<uint32_t>    attr_values_uint32;
          vector<uint16_t>    attr_values_uint16;
          vector<uint8_t>     attr_values_uint8;
          vector<int32_t>     attr_values_int32;
          vector<int16_t>     attr_values_int16;
          vector<int8_t>      attr_values_int8;
          vector<float>       attr_values_float;
          
          data::AttrVal edge_attr_values;

          while (PyDict_Next(py_attr_namespace_value, &attr_pos, &py_attr_key, &py_attr_values))
            {
              assert(py_attr_key != Py_None);
              assert(py_attr_values != Py_None);
              string attr_name = string(PyBytes_AsString(py_attr_key));
              
              npy_type = PyArray_TYPE((PyArrayObject *)py_attr_values);
              
              switch (npy_type)
                {
                case NPY_UINT32:
                  {
                    if (attr_names[AttrMap::attr_index_uint32].size() < (size_t)attr_type_idx[AttrMap::attr_index_uint32]+1)
                      {
                        attr_names[AttrMap::attr_index_uint32].push_back(attr_name);
                      }
                    else
                      {
                        size_t idx = attr_type_idx[AttrMap::attr_index_uint32];
                        assert(attr_names[AttrMap::attr_index_uint32][idx].compare(string(PyBytes_AsString(py_attr_key))) != 0);
                      }
                    
                    py_array_to_vector<uint32_t>(py_attr_values, attr_values_uint32);
                    edge_attr_values.insert(attr_values_uint32);
                    attr_type_idx[AttrMap::attr_index_uint32]++;
                    break;
                  }
                case NPY_UINT16:
                  {
                    if (attr_names[AttrMap::attr_index_uint16].size() < (size_t)attr_type_idx[AttrMap::attr_index_uint16]+1)
                      {
                        attr_names[AttrMap::attr_index_uint16].push_back(attr_name);
                      }
                    else
                      {
                        size_t idx = attr_type_idx[AttrMap::attr_index_uint16];
                        assert(attr_names[AttrMap::attr_index_uint16][idx].compare(string(PyBytes_AsString(py_attr_key))) != 0);
                      }
                    
                    py_array_to_vector<uint16_t>(py_attr_values, attr_values_uint16);
                    edge_attr_values.insert(attr_values_uint16);
                    attr_type_idx[AttrMap::attr_index_uint16]++;
                    break;
                  }
                case NPY_UINT8:
                  {
                    if (attr_names[AttrMap::attr_index_uint8].size() < (size_t)attr_type_idx[AttrMap::attr_index_uint8]+1)
                      {
                        attr_names[AttrMap::attr_index_uint8].push_back(attr_name);
                      }
                    else
                      {
                        size_t idx = attr_type_idx[AttrMap::attr_index_uint8];
                        assert(attr_names[AttrMap::attr_index_uint8][idx].compare(string(PyBytes_AsString(py_attr_key))) != 0);
                      }
                    
                    py_array_to_vector<uint8_t>(py_attr_values, attr_values_uint8);
                    edge_attr_values.insert(attr_values_uint8);
                    attr_type_idx[AttrMap::attr_index_uint8]++;
                    break;
                  }
                case NPY_INT32:
                  {
                    if (attr_names[AttrMap::attr_index_int32].size() < (size_t)attr_type_idx[AttrMap::attr_index_int32]+1)
                      {
                        attr_names[AttrMap::attr_index_int32].push_back(attr_name);
                      }
                    else
                      {
                        size_t idx = attr_type_idx[AttrMap::attr_index_int32];
                        assert(attr_names[AttrMap::attr_index_int32][idx].compare(string(PyBytes_AsString(py_attr_key))) != 0);
                      }
                    
                    py_array_to_vector<int32_t>(py_attr_values, attr_values_int32);
                    edge_attr_values.insert(attr_values_int32);
                    attr_type_idx[AttrMap::attr_index_int32]++;
                    break;
                  }
                case NPY_INT16:
                  {
                    if (attr_names[AttrMap::attr_index_int16].size() < (size_t)attr_type_idx[AttrMap::attr_index_int16]+1)
                      {
                        attr_names[AttrMap::attr_index_int16].push_back(attr_name);
                      }
                    else
                      {
                        size_t idx = attr_type_idx[AttrMap::attr_index_int16];
                        assert(attr_names[AttrMap::attr_index_int16][idx].compare(string(PyBytes_AsString(py_attr_key))) != 0);
                      }
                    
                    py_array_to_vector<int16_t>(py_attr_values, attr_values_int16);
                    edge_attr_values.insert(attr_values_int16);
                    attr_type_idx[AttrMap::attr_index_int16]++;
                    break;
                  }
                case NPY_INT8:
                  {
                    if (attr_names[AttrMap::attr_index_int8].size() < (size_t)attr_type_idx[AttrMap::attr_index_int8]+1)
                      {
                        attr_names[AttrMap::attr_index_int8].push_back(attr_name);
                      }
                    else
                      {
                        size_t idx = attr_type_idx[AttrMap::attr_index_int8];
                        assert(attr_names[AttrMap::attr_index_int8][idx].compare(string(PyBytes_AsString(py_attr_key))) != 0);
                      }
                    
                    py_array_to_vector<int8_t>(py_attr_values, attr_values_int8);
                    edge_attr_values.insert(attr_values_int8);
                    attr_type_idx[AttrMap::attr_index_int8]++;
                    break;
                  }
                case NPY_FLOAT:
                  {
                    if (attr_names[AttrMap::attr_index_float].size() < (size_t)attr_type_idx[AttrMap::attr_index_float]+1)
                      {
                        attr_names[AttrMap::attr_index_float].push_back(attr_name);
                      }
                    else
                      {
                        size_t idx = attr_type_idx[AttrMap::attr_index_float];
                        assert(attr_names[AttrMap::attr_index_float][idx].compare(string(PyBytes_AsString(py_attr_key))) != 0);
                      }
                    
                    py_array_to_vector<float>(py_attr_values, attr_values_float);
                    edge_attr_values.insert(attr_values_float);
                    attr_type_idx[AttrMap::attr_index_float]++;
                    break;
                  }
                default:
                  throw runtime_error("Unsupported attribute type");
                  break;
                }
              attr_idx = attr_idx+1;
            }
          
          edge_attr_map[attr_namespace] = edge_attr_values;
        }
      edge_map[node_idx] = make_tuple(adj_values, edge_attr_map);
    }
}



PyObject* py_build_tree_value(const CELL_IDX_T key, const neurotree_t &tree,
                              map <string, NamedAttrMap>& attr_maps)
{
  const CELL_IDX_T idx = get<0>(tree);
  assert(idx == key);

  const vector<SECTION_IDX_T> & src_vector=get<1>(tree);
  const vector<SECTION_IDX_T> & dst_vector=get<2>(tree);
  const vector<SECTION_IDX_T> & sections=get<3>(tree);
  const vector<COORD_T> & xcoords=get<4>(tree);
  const vector<COORD_T> & ycoords=get<5>(tree);
  const vector<COORD_T> & zcoords=get<6>(tree);
  const vector<REALVAL_T> & radiuses=get<7>(tree);
  const vector<LAYER_IDX_T> & layers=get<8>(tree);
  const vector<PARENT_NODE_IDX_T> & parents=get<9>(tree);
  const vector<SWC_TYPE_T> & swc_types=get<10>(tree);
  
  size_t num_nodes = xcoords.size();
        
  PyObject *py_section_topology = PyDict_New();
  npy_intp ind = 0;
  npy_intp dims[1];
  dims[0] = xcoords.size();
  PyObject *py_section_vector = (PyObject *)PyArray_SimpleNew(1, dims, NPY_UINT16);
  SECTION_IDX_T *section_vector_ptr = (SECTION_IDX_T *)PyArray_GetPtr((PyArrayObject *)py_section_vector, &ind);
  size_t sections_ptr=0;
  SECTION_IDX_T section_idx = 0;
  PyObject *py_section_node_map = PyDict_New();
  PyObject *py_section_key; PyObject *py_section_nodes;
  set<NODE_IDX_T> marked_nodes;
  size_t num_sections = sections[sections_ptr];
  sections_ptr++;
  while (sections_ptr < sections.size())
    {
      vector<NODE_IDX_T> section_nodes;
      size_t num_section_nodes = sections[sections_ptr];
      npy_intp nodes_dims[1], nodes_ind = 0;
      nodes_dims[0]    = num_section_nodes;
      py_section_key   = PyLong_FromLong((long)section_idx);
      py_section_nodes = (PyObject *)PyArray_SimpleNew(1, nodes_dims, NPY_UINT32);
      NODE_IDX_T *section_nodes_ptr = (NODE_IDX_T *)PyArray_GetPtr((PyArrayObject *)py_section_nodes, &nodes_ind);
      sections_ptr++;
      for (size_t p = 0; p < num_section_nodes; p++)
        {
          NODE_IDX_T node_idx = sections[sections_ptr];
          assert(node_idx <= num_nodes);
          section_nodes_ptr[p] = node_idx;
          if (marked_nodes.find(node_idx) == marked_nodes.end())
            {
              section_vector_ptr[node_idx] = section_idx;
              marked_nodes.insert(node_idx);
            }
          sections_ptr++;
        }
      PyDict_SetItem(py_section_node_map, py_section_key, py_section_nodes);
      section_idx++;
    }
  assert(section_idx == num_sections);
  
  dims[0] = src_vector.size();
  PyObject *py_section_src = (PyObject *)PyArray_SimpleNew(1, dims, NPY_UINT16);
  SECTION_IDX_T *section_src_ptr = (SECTION_IDX_T *)PyArray_GetPtr((PyArrayObject *)py_section_src, &ind);
  PyObject *py_section_dst = (PyObject *)PyArray_SimpleNew(1, dims, NPY_UINT16);
  SECTION_IDX_T *section_dst_ptr = (SECTION_IDX_T *)PyArray_GetPtr((PyArrayObject *)py_section_dst, &ind);
  for (size_t s = 0; s < src_vector.size(); s++)
    {
      section_src_ptr[s] = src_vector[s];
      section_dst_ptr[s] = dst_vector[s];
    }
  
  PyDict_SetItemString(py_section_topology, "num_sections", PyLong_FromUnsignedLong(num_sections));
  PyDict_SetItemString(py_section_topology, "nodes", py_section_node_map);
  PyDict_SetItemString(py_section_topology, "src", py_section_src);
  PyDict_SetItemString(py_section_topology, "dst", py_section_dst);
  
  dims[0] = xcoords.size();
  
  PyObject *py_xcoords = (PyObject *)PyArray_SimpleNew(1, dims, NPY_FLOAT);
  float *xcoords_ptr = (float *)PyArray_GetPtr((PyArrayObject *)py_xcoords, &ind);
  PyObject *py_ycoords = (PyObject *)PyArray_SimpleNew(1, dims, NPY_FLOAT);
  float *ycoords_ptr = (float *)PyArray_GetPtr((PyArrayObject *)py_ycoords, &ind);
  PyObject *py_zcoords = (PyObject *)PyArray_SimpleNew(1, dims, NPY_FLOAT);
  float *zcoords_ptr = (float *)PyArray_GetPtr((PyArrayObject *)py_zcoords, &ind);
  PyObject *py_radiuses = (PyObject *)PyArray_SimpleNew(1, dims, NPY_FLOAT);
  float *radiuses_ptr = (float *)PyArray_GetPtr((PyArrayObject *)py_radiuses, &ind);
  PyObject *py_layers = (PyObject *)PyArray_SimpleNew(1, dims, NPY_INT8);
  LAYER_IDX_T *layers_ptr = (LAYER_IDX_T *)PyArray_GetPtr((PyArrayObject *)py_layers, &ind);
  PyObject *py_parents = (PyObject *)PyArray_SimpleNew(1, dims, NPY_INT32);
  PARENT_NODE_IDX_T *parents_ptr = (PARENT_NODE_IDX_T *)PyArray_GetPtr((PyArrayObject *)py_parents, &ind);
  PyObject *py_swc_types = (PyObject *)PyArray_SimpleNew(1, dims, NPY_INT8);
  SWC_TYPE_T *swc_types_ptr = (SWC_TYPE_T *)PyArray_GetPtr((PyArrayObject *)py_swc_types, &ind);
  for (size_t j = 0; j < xcoords.size(); j++)
    {
      xcoords_ptr[j]   = xcoords[j];
      ycoords_ptr[j]   = ycoords[j];
      zcoords_ptr[j]   = zcoords[j];
      radiuses_ptr[j]  = radiuses[j];
      layers_ptr[j]    = layers[j];
      parents_ptr[j]   = parents[j];
      swc_types_ptr[j] = swc_types[j];
    }
  
  PyObject *py_treeval = PyDict_New();
  PyDict_SetItemString(py_treeval, "x", py_xcoords);
  PyDict_SetItemString(py_treeval, "y", py_ycoords);
  PyDict_SetItemString(py_treeval, "z", py_zcoords);
  PyDict_SetItemString(py_treeval, "radius", py_radiuses);
  PyDict_SetItemString(py_treeval, "layer", py_layers);
  PyDict_SetItemString(py_treeval, "parent", py_parents);
  PyDict_SetItemString(py_treeval, "swc_type", py_swc_types);
  PyDict_SetItemString(py_treeval, "section", py_section_vector);
  PyDict_SetItemString(py_treeval, "section_topology", py_section_topology);

  
  for (auto const& attr_map_entry : attr_maps)
    {
      const string& attr_namespace  = attr_map_entry.first;
      data::NamedAttrMap attr_map  = attr_map_entry.second;
      vector <vector<string> > attr_names;

      attr_map.attr_names(attr_names);
        
      PyObject *py_namespace_dict = PyDict_New();

      const vector <vector <float>> &float_attrs     = attr_map.find<float>(idx);
      const vector <vector <uint8_t>> &uint8_attrs   = attr_map.find<uint8_t>(idx);
      const vector <vector <int8_t>> &int8_attrs     = attr_map.find<int8_t>(idx);
      const vector <vector <uint16_t>> &uint16_attrs = attr_map.find<uint16_t>(idx);
      const vector <vector <uint32_t>> &uint32_attrs = attr_map.find<uint32_t>(idx);
      const vector <vector <int32_t>> &int32_attrs   = attr_map.find<int32_t>(idx);

      for (size_t i=0; i<float_attrs.size(); i++)
        {
          const vector<float> &attr_value = float_attrs[i];
          dims[0] = attr_value.size();
          PyObject *py_value = (PyObject *)PyArray_SimpleNew(1, dims, NPY_FLOAT);
          float *py_value_ptr = (float *)PyArray_GetPtr((PyArrayObject *)py_value, &ind);
          for (size_t j = 0; j < attr_value.size(); j++)
            {
              py_value_ptr[j]   = attr_value[j];
            }
          
          PyDict_SetItemString(py_namespace_dict,
                               (attr_names[AttrMap::attr_index_float][i]).c_str(),
                               py_value);
        }
      for (size_t i=0; i<uint8_attrs.size(); i++)
        {
          const vector<uint8_t> &attr_value = uint8_attrs[i];
          dims[0] = attr_value.size();
          PyObject *py_value = (PyObject *)PyArray_SimpleNew(1, dims, NPY_UINT8);
          uint8_t *py_value_ptr = (uint8_t *)PyArray_GetPtr((PyArrayObject *)py_value, &ind);
          for (size_t j = 0; j < attr_value.size(); j++)
            {
              py_value_ptr[j]   = attr_value[j];
            }
          
          PyDict_SetItemString(py_namespace_dict,
                               (attr_names[AttrMap::attr_index_uint8][i]).c_str(),
                               py_value);
        }
      for (size_t i=0; i<int8_attrs.size(); i++)
        {
          const vector<int8_t> &attr_value = int8_attrs[i];
          dims[0] = attr_value.size();
          PyObject *py_value = (PyObject *)PyArray_SimpleNew(1, dims, NPY_INT8);
          int8_t *py_value_ptr = (int8_t *)PyArray_GetPtr((PyArrayObject *)py_value, &ind);
          for (size_t j = 0; j < attr_value.size(); j++)
            {
              py_value_ptr[j]   = attr_value[j];
            }
          
          PyDict_SetItemString(py_namespace_dict,
                               (attr_names[AttrMap::attr_index_int8][i]).c_str(),
                               py_value);
        }
      for (size_t i=0; i<uint16_attrs.size(); i++)
        {
          const vector<uint16_t> &attr_value = uint16_attrs[i];
          dims[0] = attr_value.size();
          PyObject *py_value = (PyObject *)PyArray_SimpleNew(1, dims, NPY_UINT16);
          uint16_t *py_value_ptr = (uint16_t *)PyArray_GetPtr((PyArrayObject *)py_value, &ind);
          for (size_t j = 0; j < attr_value.size(); j++)
            {
              py_value_ptr[j]   = attr_value[j];
            }
          
          PyDict_SetItemString(py_namespace_dict,
                               (attr_names[AttrMap::attr_index_uint16][i]).c_str(),
                               py_value);
        }
      for (size_t i=0; i<uint32_attrs.size(); i++)
        {
          const vector<uint32_t> &attr_value = uint32_attrs[i];
          dims[0] = attr_value.size();
          PyObject *py_value = (PyObject *)PyArray_SimpleNew(1, dims, NPY_UINT32);
          uint32_t *py_value_ptr = (uint32_t *)PyArray_GetPtr((PyArrayObject *)py_value, &ind);
          for (size_t j = 0; j < attr_value.size(); j++)
            {
              py_value_ptr[j]   = attr_value[j];
            }
          
          PyDict_SetItemString(py_namespace_dict,
                               (attr_names[AttrMap::attr_index_uint32][i]).c_str(),
                               py_value);
        }
      for (size_t i=0; i<int32_attrs.size(); i++)
        {
          const vector<int32_t> &attr_value = int32_attrs[i];
          dims[0] = attr_value.size();
          PyObject *py_value = (PyObject *)PyArray_SimpleNew(1, dims, NPY_INT32);
          int32_t *py_value_ptr = (int32_t *)PyArray_GetPtr((PyArrayObject *)py_value, &ind);
          for (size_t j = 0; j < attr_value.size(); j++)
            {
              py_value_ptr[j]   = attr_value[j];
            }
          
          PyDict_SetItemString(py_namespace_dict,
                               (attr_names[AttrMap::attr_index_int32][i]).c_str(),
                               py_value);
        }

      PyDict_SetItemString(py_treeval,
                           attr_namespace.c_str(),
                           py_namespace_dict);
    }
  

  return py_treeval;
}


PyObject* py_build_cell_attr_value(const CELL_IDX_T key, 
                                   NamedAttrMap& attr_map,
                                   const string& attr_namespace,
                                   const vector <vector<string> >& attr_names)
{
  PyObject *py_result = PyDict_New();
  PyObject *py_attrval = PyDict_New();
  npy_intp dims[1];
  npy_intp ind = 0;
  
  const vector < vector <float>> &float_attrs      = attr_map.find<float>(key);
  const vector < vector <uint8_t> > &uint8_attrs   = attr_map.find<uint8_t>(key);
  const vector < vector <int8_t> > &int8_attrs     = attr_map.find<int8_t>(key);
  const vector < vector <uint16_t> > &uint16_attrs = attr_map.find<uint16_t>(key);
  const vector < vector <int16_t> > &int16_attrs   = attr_map.find<int16_t>(key);
  const vector < vector <uint32_t> > &uint32_attrs = attr_map.find<uint32_t>(key);
  const vector < vector <int32_t> > &int32_attrs   = attr_map.find<int32_t>(key);
  
  for (size_t i=0; i<float_attrs.size(); i++)
    {
      const vector<float> &attr_value = float_attrs[i];
      dims[0] = attr_value.size();
      PyObject *py_value = (PyObject *)PyArray_SimpleNew(1, dims, NPY_FLOAT);
      float *py_value_ptr = (float *)PyArray_GetPtr((PyArrayObject *)py_value, &ind);
      for (size_t j = 0; j < attr_value.size(); j++)
        {
          py_value_ptr[j]   = attr_value[j];
        }
          
      PyDict_SetItemString(py_attrval,
                           (attr_names[AttrMap::attr_index_float][i]).c_str(),
                           py_value);
    }

  for (size_t i=0; i<uint8_attrs.size(); i++)
    {
      const vector<uint8_t> &attr_value = uint8_attrs[i];
      dims[0] = attr_value.size();
      PyObject *py_value = (PyObject *)PyArray_SimpleNew(1, dims, NPY_UINT8);
      uint8_t *py_value_ptr = (uint8_t *)PyArray_GetPtr((PyArrayObject *)py_value, &ind);
      for (size_t j = 0; j < attr_value.size(); j++)
        {
          py_value_ptr[j]   = attr_value[j];
        }
      
      PyDict_SetItemString(py_attrval,
                           (attr_names[AttrMap::attr_index_uint8][i]).c_str(),
                           py_value);
    }
  
  for (size_t i=0; i<int8_attrs.size(); i++)
    {
      const vector<int8_t> &attr_value = int8_attrs[i];
      dims[0] = attr_value.size();
      PyObject *py_value = (PyObject *)PyArray_SimpleNew(1, dims, NPY_INT8);
      int8_t *py_value_ptr = (int8_t *)PyArray_GetPtr((PyArrayObject *)py_value, &ind);
      for (size_t j = 0; j < attr_value.size(); j++)
        {
          py_value_ptr[j]   = attr_value[j];
        }
      
      PyDict_SetItemString(py_attrval,
                           (attr_names[AttrMap::attr_index_int8][i]).c_str(),
                           py_value);
    }
  
  for (size_t i=0; i<uint16_attrs.size(); i++)
    {
      const vector<uint16_t> &attr_value = uint16_attrs[i];
      dims[0] = attr_value.size();
      PyObject *py_value = (PyObject *)PyArray_SimpleNew(1, dims, NPY_UINT16);
      uint16_t *py_value_ptr = (uint16_t *)PyArray_GetPtr((PyArrayObject *)py_value, &ind);
      for (size_t j = 0; j < attr_value.size(); j++)
        {
          py_value_ptr[j]   = attr_value[j];
        }
      
      PyDict_SetItemString(py_attrval,
                           (attr_names[AttrMap::attr_index_uint16][i]).c_str(),
                           py_value);
    }

  for (size_t i=0; i<uint32_attrs.size(); i++)
    {
      const vector<uint32_t> &attr_value = uint32_attrs[i];
      dims[0] = attr_value.size();
      PyObject *py_value = (PyObject *)PyArray_SimpleNew(1, dims, NPY_UINT32);
      uint32_t *py_value_ptr = (uint32_t *)PyArray_GetPtr((PyArrayObject *)py_value, &ind);
      for (size_t j = 0; j < attr_value.size(); j++)
        {
          py_value_ptr[j]   = attr_value[j];
        }
      
      PyDict_SetItemString(py_attrval,
                           (attr_names[AttrMap::attr_index_uint32][i]).c_str(),
                           py_value);
    }
  
  for (size_t i=0; i<int32_attrs.size(); i++)
    {
      const vector<int32_t> &attr_value = int32_attrs[i];
      dims[0] = attr_value.size();
      PyObject *py_value = (PyObject *)PyArray_SimpleNew(1, dims, NPY_INT32);
      int32_t *py_value_ptr = (int32_t *)PyArray_GetPtr((PyArrayObject *)py_value, &ind);
      for (size_t j = 0; j < attr_value.size(); j++)
        {
          py_value_ptr[j]   = attr_value[j];
        }
      
      PyDict_SetItemString(py_attrval,
                           (attr_names[AttrMap::attr_index_int32][i]).c_str(),
                           py_value);
    }

  PyDict_SetItemString(py_result,
                       attr_namespace.c_str(),
                       py_attrval);

  return py_result;
}


template <class T>
void build_edge_attr_vec (const AttrVal& attr_val, const size_t pos,
                          vector < vector<T> >& edge_attr_values)
{
  for (size_t attr_index=0; attr_index<attr_val.size_attr_vec<T>(); attr_index++)
    {
      vector<T> vec_value;
      vec_value.push_back(attr_val.at<T>(attr_index, pos));
      edge_attr_values.push_back(vec_value);
    }
}
      
      
template <class T>
void py_build_edge_attr_value (const vector < vector<T> >& edge_attr_values,
                               const NPY_TYPES numpy_type,
                               const size_t attr_index,
                               const vector <vector<string> >& attr_names,
                               PyObject *py_attrval)
{
  for (size_t i=0; i<edge_attr_values.size(); i++)
    {
      npy_intp dims[1]; npy_intp ind = 0;
      const vector<T> &attr_value = edge_attr_values[i];
      dims[0] = attr_value.size();
      PyObject *py_value = (PyObject *)PyArray_SimpleNew(1, dims, numpy_type);
      T *py_value_ptr = (T *)PyArray_GetPtr((PyArrayObject *)py_value, &ind);
      for (size_t j = 0; j < attr_value.size(); j++)
        {
          py_value_ptr[j]   = attr_value[j];
        }
      
      PyDict_SetItemString(py_attrval,
                           (attr_names[attr_index][i]).c_str(),
                           py_value);
    }
}


PyObject* py_build_edge_value(const NODE_IDX_T key,
                              const NODE_IDX_T adj, 
                              const size_t pos, 
                              const map <string, AttrVal&>& attr_val_map,
                              const map <string, vector <vector<string> > >& attr_names)
{
  PyObject *py_attrval = Py_None;
  PyObject *py_attrmap = Py_None;

  for ( auto it = attr_val_map.cbegin(); it != attr_val_map.cend(); it++ )
    {
      const string& attr_namespace = it->first;

      const AttrVal& attr_val = it->second;
      
      vector < vector <float> >    float_attrs;
      vector < vector <uint8_t> >  uint8_attrs;
      vector < vector <int8_t> >   int8_attrs;
      vector < vector <uint16_t> > uint16_attrs;
      vector < vector <int16_t> >  int16_attrs;
      vector < vector <uint32_t> > uint32_attrs;
      vector < vector <int32_t> >  int32_attrs;
          
      build_edge_attr_vec<float>(attr_val, pos, float_attrs);
      build_edge_attr_vec<uint8_t>(attr_val, pos, uint8_attrs);
      build_edge_attr_vec<uint16_t>(attr_val, pos, uint16_attrs);
      build_edge_attr_vec<uint32_t>(attr_val, pos, uint32_attrs);
      build_edge_attr_vec<int8_t>(attr_val, pos, int8_attrs);
      build_edge_attr_vec<int16_t>(attr_val, pos, int16_attrs);
      build_edge_attr_vec<int32_t>(attr_val, pos, int32_attrs);
      
      py_attrval = PyDict_New();
      py_attrmap = PyDict_New();
      
      py_build_edge_attr_value (float_attrs, NPY_FLOAT, AttrMap::attr_index_float,
                                attr_names[attr_namespace], py_attrval);
      py_build_edge_attr_value (uint8_attrs, NPY_UINT8, AttrMap::attr_index_uint8,
                                attr_names[attr_namespace], py_attrval);
      py_build_edge_attr_value (uint16_attrs, NPY_UINT16, AttrMap::attr_index_uint16,
                                attr_names[attr_namespace], py_attrval);
      py_build_edge_attr_value (uint32_attrs, NPY_UINT32, AttrMap::attr_index_uint32,
                                attr_names[attr_namespace], py_attrval);
      py_build_edge_attr_value (int8_attrs,   NPY_INT8, AttrMap::attr_index_int8,
                                attr_names[attr_namespace], py_attrval);
      py_build_edge_attr_value (int16_attrs,  NPY_INT16, AttrMap::attr_index_int16,
                                attr_names[attr_namespace], py_attrval);
      py_build_edge_attr_value (int32_attrs,  NPY_INT32, AttrMap::attr_index_int32,
                                attr_names[attr_namespace], py_attrval);
      
      PyDict_SetItemString(py_attrmap,
                           attr_namespace.c_str(),
                           py_attrval);
    }
  
  PyObject *py_result = PyTuple_New(3);

  PyObject *py_key = PyLong_FromLong(key);
  PyObject *py_adj = PyLong_FromLong(adj);
  
  PyTuple_SetItem(py_result, 0, py_key);
  PyTuple_SetItem(py_result, 1, py_adj);
  PyTuple_SetItem(py_result, 2, (attr_names.size()>0) ? py_attrmap : (Py_INCREF(Py_None), Py_None));
  
  return py_result;
}


PyObject* py_build_edge_tuple_value (const edge_tuple_t& et)
{
  const vector<NODE_IDX_T>& adj_vector = get<0>(et);
  const map<string, AttrVal>& edge_attr_map = get<1>(et);

  npy_intp dims[1], ind = 0;
  dims[0] = adj_vector.size();
                
  PyObject *adj_arr = PyArray_SimpleNew(1, dims, NPY_UINT32);
  uint32_t *adj_ptr = (uint32_t *)PyArray_GetPtr((PyArrayObject *)adj_arr, &ind);
  
  for (size_t j = 0; j < adj_vector.size(); j++)
    {
      adj_ptr[j] = adj_vector[j];
    }
  
  PyObject *py_attrmap  = PyDict_New();

  for (auto iter : edge_attr_map)
    {
      const string & edge_attr_namespace = iter->first;
      const AttrVal& edge_attr_values = iter->second;
          
      PyObject *py_attrval  = PyList_New(0);
          
      vector <PyObject*> py_float_edge_attrs;
      vector <PyObject*> py_uint8_edge_attrs;
      vector <PyObject*> py_uint16_edge_attrs;
      vector <PyObject*> py_uint32_edge_attrs;
      vector <PyObject*> py_int8_edge_attrs;
      vector <PyObject*> py_int16_edge_attrs;
      vector <PyObject*> py_int32_edge_attrs;
          
      vector <float*>    py_float_edge_attrs_ptr;
      vector <uint8_t*>  py_uint8_edge_attrs_ptr;
      vector <uint16_t*> py_uint16_edge_attrs_ptr;
      vector <uint32_t*> py_uint32_edge_attrs_ptr;
      vector <uint8_t*>  py_int8_edge_attrs_ptr;
      vector <uint16_t*> py_int16_edge_attrs_ptr;
      vector <uint32_t*> py_int32_edge_attrs_ptr;
          
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
      for (size_t j = 0; j < edge_attr_values.size_attr_vec<int8_t>(); j++)
        {
          PyObject *arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_INT8);
          uint8_t *ptr = (int8_t *)PyArray_GetPtr((PyArrayObject *)arr, &ind);
          py_int8_edge_attrs.push_back(arr);
          py_int8_edge_attrs_ptr.push_back(ptr);
        }
      for (size_t j = 0; j < edge_attr_values.size_attr_vec<int16_t>(); j++)
        {
          PyObject *arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_INT16);
          uint16_t *ptr = (int16_t *)PyArray_GetPtr((PyArrayObject *)arr, &ind);
          py_int16_edge_attrs.push_back(arr);
          py_int16_edge_attrs_ptr.push_back(ptr);
        }
      for (size_t j = 0; j < edge_attr_values.size_attr_vec<int32_t>(); j++)
        {
          PyObject *arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_INT32);
          int32_t *ptr = (uint32_t *)PyArray_GetPtr((PyArrayObject *)arr, &ind);
          py_int32_edge_attrs.push_back(arr);
          py_int32_edge_attrs_ptr.push_back(ptr);
        }
          
      for (size_t j = 0; j < adj_vector.size(); j++)
        {
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
          for (size_t k = 0; k < edge_attr_values.size_attr_vec<int8_t>(); k++)
            {
              py_int8_edge_attrs_ptr[k][j] = edge_attr_values.at<int8_t>(k,j); 
            }
          for (size_t k = 0; k < edge_attr_values.size_attr_vec<int16_t>(); k++)
            {
              py_int16_edge_attrs_ptr[k][j] = edge_attr_values.at<int16_t>(k,j); 
            }
          for (size_t k = 0; k < edge_attr_values.size_attr_vec<int32_t>(); k++)
            {
              py_int32_edge_attrs_ptr[k][j] = edge_attr_values.at<int32_t>(k,j); 
            }
        }
          
      for (size_t j = 0; j < edge_attr_values.size_attr_vec<float>(); j++)
        {
          status = PyList_Append(py_attrval, py_float_edge_attrs[j]);
          assert(status == 0);
        }
      for (size_t j = 0; j < edge_attr_values.size_attr_vec<uint8_t>(); j++)
        {
          status = PyList_Append(py_attrval, py_uint8_edge_attrs[j]);
          assert(status == 0);
        }
      for (size_t j = 0; j < edge_attr_values.size_attr_vec<uint16_t>(); j++)
        {
          status = PyList_Append(py_attrval, py_uint16_edge_attrs[j]);
          assert(status == 0);
        }
      for (size_t j = 0; j < edge_attr_values.size_attr_vec<uint32_t>(); j++)
        {
          status = PyList_Append(py_attrval, py_uint32_edge_attrs[j]);
          assert(status == 0);
        }
      for (size_t j = 0; j < edge_attr_values.size_attr_vec<int8_t>(); j++)
        {
          status = PyList_Append(py_attrval, py_int8_edge_attrs[j]);
          assert(status == 0);
        }
      for (size_t j = 0; j < edge_attr_values.size_attr_vec<int16_t>(); j++)
        {
          status = PyList_Append(py_attrval, py_int16_edge_attrs[j]);
          assert(status == 0);
        }
      for (size_t j = 0; j < edge_attr_values.size_attr_vec<int32_t>(); j++)
        {
          status = PyList_Append(py_attrval, py_int32_edge_attrs[j]);
          assert(status == 0);
        }
          
      PyDict_SetItemString(py_attrmap, attr_namespace.c_str(), py_attrval);
    }
                
  PyObject *py_edgeval  = PyTuple_New(2);
  PyTuple_SetItem(py_edgeval, 0, adj_arr);
  PyTuple_SetItem(py_edgeval, 1, py_attrmap);

  return py_edgeval;

}



extern "C"
{

  
  static PyObject *py_read_graph (PyObject *self, PyObject *args)
  {
    int status;
    map <string, vector < vector < vector<string> > > > edge_attr_name_vector;
    vector<prj_tuple_t> prj_vector;
    vector< pair<string,string> > prj_names;
    char *input_file_name;
    PyObject *py_attr_namespaces=NULL;
    PyObject *py_comm = NULL;
    MPI_Comm *comm_ptr  = NULL;
    size_t total_num_nodes, total_num_edges = 0, local_num_edges = 0;

    if (!PyArg_ParseTuple(args, "Os|O", &py_comm, &input_file_name, &py_attr_namespaces))
      return NULL;

    PyObject *py_prj_dict = PyDict_New();

    assert(py_comm != NULL);
    comm_ptr = PyMPIComm_Get(py_comm);
    assert(comm_ptr != NULL);
    assert(*comm_ptr != MPI_COMM_NULL);

    vector <string> attr_namespaces;
    // Create C++ vector of namespace strings:
    if (py_attr_namespaces != NULL)
      {
        for (size_t i = 0; (Py_ssize_t)i < PyList_Size(py_attr_namespaces); i++)
          {
            PyObject *pyval = PyList_GetItem(py_attr_namespaces, (Py_ssize_t)i);
            char *str = PyBytes_AsString (pyval);
            attr_namespaces.push_back(string(str));
          }
      }

    assert(graph::read_projection_names(*comm_ptr, input_file_name, prj_names) >= 0);

    graph::read_graph(*comm_ptr, std::string(input_file_name), attr_namespaces,
                      prj_names, prj_vector, edge_attr_name_vector,
                      total_num_nodes, local_num_edges, total_num_edges);
    
    for (size_t i = 0; i < prj_vector.size(); i++)
      {
        const prj_tuple_t& prj = prj_vector[i];
        const vector < vector<string> > >& edge_attr_names = edge_attr_name_vector[i];
        
        const vector<NODE_IDX_T>& src_vector       = get<0>(prj);
        const vector<NODE_IDX_T>& dst_vector       = get<1>(prj);
        const map<string, AttrVal>&  edge_attr_map = get<2>(prj);

        PyObject *py_prjval = PyList_New(0);
        
        for (size_t j = 0; j < src_vector.size(); j++)
          {
            NODE_IDX_T src = src_vector[i];
            NODE_IDX_T dst = dst_vector[i];
            
            PyObject* py_edge_val =  py_build_edge_value(src, dst, j, edge_attr_map, edge_attr_names);

            PyList_Append(py_prjval, py_edge_val);
          }


        PyObject *py_src_dict = PyDict_GetItemString(py_prj_dict, prj_names[i].second.c_str());
        if (py_src_dict == NULL)
          {
            py_src_dict = PyDict_New();
            PyDict_SetItemString(py_src_dict, prj_names[i].first.c_str(), py_prjval);
            PyDict_SetItemString(py_prj_dict, prj_names[i].second.c_str(), py_src_dict);
          }
        else
          {
            PyDict_SetItemString(py_src_dict, prj_names[i].first.c_str(), py_prjval);
          }
        
      }

    return py_prj_dict;
  }

  
  static PyObject *py_read_graph_serial (PyObject *self, PyObject *args)
  {
    int status;
    vector<prj_tuple_t> prj_vector;
    vector< pair<string,string> > prj_names;
    PyObject *py_prj_dict = PyDict_New();
    char *input_file_name;
    size_t total_num_nodes, total_num_edges = 0;

    if (!PyArg_ParseTuple(args, "s", &input_file_name))
      return NULL;

    assert(graph::read_projection_names_serial(input_file_name, prj_names) >= 0);

    graph::read_graph_serial(std::string(input_file_name), true,
                             prj_names, prj_vector,
                             total_num_nodes, total_num_edges);
    for (size_t i = 0; i < prj_vector.size(); i++)
      {
        const prj_tuple_t& prj = prj_vector[i];
        const vector < vector<string> > >& edge_attr_names = edge_attr_name_vector[i];
        
        const vector<NODE_IDX_T>& src_vector       = get<0>(prj);
        const vector<NODE_IDX_T>& dst_vector       = get<1>(prj);
        const map<string, AttrVal>&  edge_attr_map = get<2>(prj);

        PyObject *py_prjval = PyList_New(0);
        
        for (size_t j = 0; j < src_vector.size(); j++)
          {
            NODE_IDX_T src = src_vector[i];
            NODE_IDX_T dst = dst_vector[i];
            
            PyObject* py_edge_val =  py_build_edge_value(src, dst, j, edge_attr_map, edge_attr_names);

            PyList_Append(py_prjval, py_edge_val);
          }


        PyObject *py_src_dict = PyDict_GetItemString(py_prj_dict, prj_names[i].second.c_str());
        if (py_src_dict == NULL)
          {
            py_src_dict = PyDict_New();
            PyDict_SetItemString(py_src_dict, prj_names[i].first.c_str(), py_prjval);
            PyDict_SetItemString(py_prj_dict, prj_names[i].second.c_str(), py_src_dict);
          }
        else
          {
            PyDict_SetItemString(py_src_dict, prj_names[i].first.c_str(), py_prjval);
          }
      }

    return py_prj_dict;
  }

  
  static PyObject *py_scatter_read_graph (PyObject *self, PyObject *args, PyObject *kwds)
  {
    int status; int opt_edge_map_type=0;
    graph::EdgeMapType edge_map_type = graph::EdgeMapDst;
    // A vector that maps nodes to compute ranks
    PyObject *py_node_rank_map=NULL;
    PyObject *py_attr_namespaces=NULL;
    map<NODE_IDX_T, rank_t> node_rank_map;
    vector < edge_map_t > prj_vector;
    map <string, vector < vector < vector<string> > > > edge_attr_name_vector;
    vector<pop_range_t> pop_vector;
    map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
    vector<pair<string,string> > prj_names;
    PyObject *py_prj_dict = PyDict_New();
    unsigned long io_size; int size;
    PyObject *py_comm = NULL;
    MPI_Comm *comm_ptr = NULL;
    
    char *input_file_name;
    size_t local_num_nodes = 0, total_num_nodes = 0, total_num_edges = 0, local_num_edges = 0;
    
    static const char *kwlist[] = {"comm",
                                   "file_name",
                                   "node_rank_map",
                                   "namespaces",
                                   "map_type",
                                   "io_size",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Os|OOik", (char **)kwlist,
                                     &py_comm, &input_file_name, 
                                     &py_node_rank_map, &py_attr_namespaces,
                                     &opt_edge_map_type, &io_size))
      return NULL;

    assert(py_comm != NULL);
    comm_ptr = PyMPIComm_Get(py_comm);
    assert(comm_ptr != NULL);
    assert(*comm_ptr != MPI_COMM_NULL);
    
    if (opt_edge_map_type == 1)
      {
        edge_map_type = graph::EdgeMapSrc;
      }
    
    assert(MPI_Comm_size(*comm_ptr, &size) >= 0);

    if (io_size == 0)
      {
        io_size = size;
      }

    vector <string> attr_namespaces;
    // Create C++ vector of namespace strings:
    if (py_attr_namespaces != NULL)
      {
        for (size_t i = 0; (Py_ssize_t)i < PyList_Size(py_attr_namespaces); i++)
          {
            PyObject *pyval = PyList_GetItem(py_attr_namespaces, (Py_ssize_t)i);
            char *str = PyBytes_AsString (pyval);
            attr_namespaces.push_back(string(str));
          }
      }
    
    assert(graph::read_projection_names(*comm_ptr, input_file_name, prj_names) >= 0);

    // Read population info to determine total_num_nodes
    assert(cell::read_population_ranges(*comm_ptr, input_file_name, pop_ranges, pop_vector, total_num_nodes) >= 0);

    // Create C++ map for node_rank_map:
    if ((py_node_rank_map != NULL) && (py_node_rank_map != Py_None))
      {
        build_node_rank_map(py_node_rank_map, node_rank_map);
      }
    else
      {
        // round-robin node to rank assignment from file
        for (size_t i = 0; i < total_num_nodes; i++)
          {
            node_rank_map.insert(make_pair(i, i%size));
          }
      }

    graph::scatter_read_graph(*comm_ptr, edge_map_type, std::string(input_file_name),
                              io_size, attr_namespaces, prj_names, node_rank_map, prj_vector, edge_attr_name_vector, 
                              local_num_nodes, total_num_nodes, local_num_edges, total_num_edges);

    PyObject *py_attribute_info = PyDict_New();
    for (string& attr_namespace : attr_namespaces) 
      {
        for (size_t p = 0; p<edge_attr_name_vector[attr_namespace].size(); p++)
          {
            PyObject *py_prj_attr_info  = PyDict_New();
            int attr_index=0;
            for (size_t n = 0; n<edge_attr_name_vector[attr_namespace][p].size(); n++)
              {
                for (size_t t = 0; t<edge_attr_name_vector[attr_namespace][p][n].size(); t++)
                  {
                    PyObject *py_attr_key = PyBytes_FromString(edge_attr_name_vector[attr_namespace][p][n][t].c_str());
                    PyObject *py_attr_index = PyLong_FromLong(attr_index);
                    
                    PyDict_SetItem(py_prj_attr_info, py_attr_key, py_attr_index);
                    attr_index++;
                  }
              }
            PyObject *py_prj_key = PyTuple_New(2);
            PyTuple_SetItem(py_prj_key, 0, PyBytes_FromString(prj_names[p].first.c_str()));
            PyTuple_SetItem(py_prj_key, 1, PyBytes_FromString(prj_names[p].second.c_str()));
            PyDict_SetItem(py_attribute_info, py_prj_key, py_prj_attr_info);
          }
      }

    
    for (size_t i = 0; i < prj_vector.size(); i++)
      {
        PyObject *py_edge_dict = PyDict_New();
        edge_map_t prj_edge_map = prj_vector[i];
        
        if (prj_edge_map.size() > 0)
          {
            for (auto it = prj_edge_map.begin(); it != prj_edge_map.end(); it++)
              {
                NODE_IDX_T key_node = it->first;
                edge_tuple_t& et    = it->second;

                PyObject* py_edge_tuple_value = py_build_edge_tuple_value (et);

                PyObject *key = PyLong_FromLong(key_node);
                PyDict_SetItem(py_edge_dict, key, py_edge_tuple_value);
              }
          }
        
         PyObject *py_src_dict = PyDict_GetItemString(py_prj_dict, prj_names[i].second.c_str());
         if (py_src_dict == NULL)
           {
             py_src_dict = PyDict_New();
             PyDict_SetItemString(py_src_dict, prj_names[i].first.c_str(), py_edge_dict);
             PyDict_SetItemString(py_prj_dict, prj_names[i].second.c_str(), py_src_dict);
           }
         else
           {
             PyDict_SetItemString(py_src_dict, prj_names[i].first.c_str(), py_edge_dict);
           }
        
      }

    PyObject *py_prj_tuple = PyTuple_New(2);
    PyTuple_SetItem(py_prj_tuple, 0, py_prj_dict);
    PyTuple_SetItem(py_prj_tuple, 1, py_attribute_info);
    return py_prj_tuple;

  }

  
  static PyObject *py_bcast_graph (PyObject *self, PyObject *args, PyObject *kwds)
  {
    int status; int opt_edge_map_type=0;
    graph::EdgeMapType edge_map_type = graph::EdgeMapDst;
    vector < edge_map_t > prj_vector;
    vector < vector <vector<string>> > edge_attr_name_vector;
    vector<pop_range_t> pop_vector;
    map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
    vector< pair<string,string> > prj_names;
    PyObject *py_prj_dict = PyDict_New();
    int size;
    char *input_file_name;
    PyObject *py_comm = NULL;
    MPI_Comm *comm_ptr  = NULL;
    PyObject *py_attr_namespaces=NULL;
    size_t total_num_nodes, total_num_edges = 0, local_num_edges = 0;
    
    static const char *kwlist[] = {"comm",
                                   "file_name",
                                   "namespaces",
                                   "map_type",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Os|ii", (char **)kwlist,
                                     &py_comm, &input_file_name, 
                                     &py_attr_namespaces, &opt_edge_map_type))
      return NULL;

    assert(py_comm != NULL);
    comm_ptr = PyMPIComm_Get(py_comm);
    assert(comm_ptr != NULL);
    assert(*comm_ptr != MPI_COMM_NULL);

    if (opt_edge_map_type == 1)
      {
        edge_map_type = graph::EdgeMapSrc;
      }
    
    assert(MPI_Comm_size(*comm_ptr, &size) >= 0);

    vector <string> attr_namespaces;
    // Create C++ vector of namespace strings:
    if (py_attr_namespaces != NULL)
      {
        for (size_t i = 0; (Py_ssize_t)i < PyList_Size(py_attr_namespaces); i++)
          {
            PyObject *pyval = PyList_GetItem(py_attr_namespaces, (Py_ssize_t)i);
            char *str = PyBytes_AsString (pyval);
            attr_namespaces.push_back(string(str));
          }
      }

    assert(graph::read_projection_names(*comm_ptr, input_file_name, prj_names) >= 0);

    // Read population info to determine total_num_nodes
    assert(cell::read_population_ranges(*comm_ptr, input_file_name, pop_ranges, pop_vector, total_num_nodes) >= 0);

    graph::bcast_graph(*comm_ptr, edge_map_type, std::string(input_file_name),
                       attr_namespaces, prj_names, prj_vector, edge_attr_name_vector, 
                       total_num_nodes, local_num_edges, total_num_edges);

    PyObject *py_attribute_info = PyDict_New();
    if (attr_namespaces.size()>0)
      {
        for (size_t p = 0; p<edge_attr_name_vector.size(); p++)
          {
            PyObject *py_prj_attr_info  = PyDict_New();
            int attr_index=0;
            for (size_t n = 0; n<edge_attr_name_vector[p].size(); n++)
              {
                for (size_t t = 0; t<edge_attr_name_vector[p][n].size(); t++)
                  {
                    PyObject *py_attr_key = PyBytes_FromString(edge_attr_name_vector[p][n][t].c_str());
                    PyObject *py_attr_index = PyLong_FromLong(attr_index);
                    
                    PyDict_SetItem(py_prj_attr_info, py_attr_key, py_attr_index);
                    attr_index++;
                  }
              }
            PyObject *py_prj_key = PyTuple_New(2);
            PyTuple_SetItem(py_prj_key, 0, PyBytes_FromString(prj_names[p].first.c_str()));
            PyTuple_SetItem(py_prj_key, 1, PyBytes_FromString(prj_names[p].second.c_str()));
            PyDict_SetItem(py_attribute_info, py_prj_key, py_prj_attr_info);
          }
      }

    for (size_t i = 0; i < prj_vector.size(); i++)
      {
        PyObject *py_edge_dict = PyDict_New();
        edge_map_t prj_edge_map = prj_vector[i];
        
        if (prj_edge_map.size() > 0)
          {
            for (auto it = prj_edge_map.begin(); it != prj_edge_map.end(); it++)
              {
                NODE_IDX_T key_node = it->first;
                edge_tuple_t& et    = it->second;

                PyObject* py_edge_tuple_value = py_build_edge_tuple_value (et);

                PyObject *key = PyLong_FromLong(key_node);
                PyDict_SetItem(py_edge_dict, key, py_edge_tuple_value);
              }
          }
        
         PyObject *py_src_dict = PyDict_GetItemString(py_prj_dict, prj_names[i].second.c_str());
         if (py_src_dict == NULL)
           {
             py_src_dict = PyDict_New();
             PyDict_SetItemString(py_src_dict, prj_names[i].first.c_str(), py_edge_dict);
             PyDict_SetItemString(py_prj_dict, prj_names[i].second.c_str(), py_src_dict);
           }
         else
           {
             PyDict_SetItemString(py_src_dict, prj_names[i].first.c_str(), py_edge_dict);
           }
        
      }

    PyObject *py_prj_tuple = PyTuple_New(2);
    PyTuple_SetItem(py_prj_tuple, 0, py_prj_dict);
    PyTuple_SetItem(py_prj_tuple, 1, py_attribute_info);
    return py_prj_tuple;
  }


  static PyObject *py_write_graph (PyObject *self, PyObject *args, PyObject *kwds)
  {
    PyObject *edge_values = NULL;
    PyObject *py_comm  = NULL;
    MPI_Comm *comm_ptr = NULL;
    char *file_name_arg, *src_pop_name_arg, *dst_pop_name_arg;
    unsigned long io_size = 0;
    
    static const char *kwlist[] = {"comm",
                                   "file_name",
                                   "src_pop_name",
                                   "dst_pop_name",
                                   "edges",
                                   "io_size",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OsssO|k", (char **)kwlist,
                                     &py_comm, &file_name_arg,
                                     &src_pop_name_arg, &dst_pop_name_arg,
                                     &edge_values, &io_size))
        return NULL;

    assert(py_comm != NULL);
    comm_ptr = PyMPIComm_Get(py_comm);
    assert(comm_ptr != NULL);
    assert(*comm_ptr != MPI_COMM_NULL);

    int rank, size;
    assert(MPI_Comm_size(*comm_ptr, &size) >= 0);
    assert(MPI_Comm_rank(*comm_ptr, &rank) >= 0);

    if (io_size == 0)
      {
        io_size = size;
      }
    
    string file_name = string(file_name_arg);
    string src_pop_name = string(src_pop_name_arg);
    string dst_pop_name = string(dst_pop_name_arg);
    
    map <string, vector <vector <string> > > edge_attr_names;
    edge_map_t edge_map;
    
    build_edge_map(edge_values, edge_attr_names, edge_map);

    assert(graph::write_graph(*comm_ptr, io_size, file_name, src_pop_name, dst_pop_name,
                              edge_attr_names, edge_map) >= 0);
    Py_INCREF(Py_None);
    return Py_None;
  }
  

  static PyObject *py_append_graph (PyObject *self, PyObject *args, PyObject *kwds)
  {
    PyObject *edge_values = NULL;
    PyObject *py_comm  = NULL;
    MPI_Comm *comm_ptr = NULL;
    char *file_name_arg, *src_pop_name_arg, *dst_pop_name_arg;
    unsigned long io_size = 0;
    
    static const char *kwlist[] = {"comm",
                                   "file_name",
                                   "src_pop_name",
                                   "dst_pop_name",
                                   "edges",
                                   "io_size",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OsssO|k", (char **)kwlist,
                                     &py_comm, &file_name_arg,
                                     &src_pop_name_arg, &dst_pop_name_arg,
                                     &edge_values, &io_size))
        return NULL;

    assert(py_comm != NULL);
    comm_ptr = PyMPIComm_Get(py_comm);
    assert(comm_ptr != NULL);
    assert(*comm_ptr != MPI_COMM_NULL);

    int rank, size;
    assert(MPI_Comm_size(*comm_ptr, &size) >= 0);
    assert(MPI_Comm_rank(*comm_ptr, &rank) >= 0);

    if (io_size == 0)
      {
        io_size = size;
      }
    
    string file_name = string(file_name_arg);
    string src_pop_name = string(src_pop_name_arg);
    string dst_pop_name = string(dst_pop_name_arg);
    
    map <string, vector<vector<string> > > edge_attr_names;
    edge_map_t edge_map;
    
    build_edge_map(edge_values, edge_attr_names, edge_map);

    assert(graph::append_graph(*comm_ptr, io_size, file_name, src_pop_name, dst_pop_name,
                              edge_attr_names, edge_map) >= 0);
    Py_INCREF(Py_None);
    return Py_None;
  }
  
  
  static PyObject *py_read_population_names (PyObject *self, PyObject *args)
  {
    int status; 
    char *input_file_name;
    PyObject *py_comm = NULL;
    MPI_Comm *comm_ptr  = NULL;

    if (!PyArg_ParseTuple(args, "Os", &py_comm, &input_file_name))
      return NULL;

    assert(py_comm != NULL);
    comm_ptr = PyMPIComm_Get(py_comm);
    assert(comm_ptr != NULL);
    assert(*comm_ptr != MPI_COMM_NULL);

    int rank, size;
    assert(MPI_Comm_size(*comm_ptr, &size) >= 0);
    assert(MPI_Comm_rank(*comm_ptr, &rank) >= 0);

    vector <string> pop_names;
    status = cell::read_population_names(*comm_ptr, input_file_name, pop_names);
    assert (status >= 0);


    PyObject *py_population_names = PyList_New(0);
    for (size_t i=0; i<pop_names.size(); i++)
      {
        PyList_Append(py_population_names, PyBytes_FromString(pop_names[i].c_str()));
      }
    
    return py_population_names;
  }

  
  static PyObject *py_read_population_ranges (PyObject *self, PyObject *args)
  {
    int status; 
    vector< pair<pop_t, string> > pop_labels;
    PyObject *py_comm;
    MPI_Comm *comm_ptr  = NULL;
    
    char *input_file_name;

    if (!PyArg_ParseTuple(args, "Os", &py_comm, &input_file_name))
      return NULL;

    assert(py_comm != NULL);
    comm_ptr = PyMPIComm_Get(py_comm);
    assert(comm_ptr != NULL);
    assert(*comm_ptr != MPI_COMM_NULL);

    int rank, size;
    assert(MPI_Comm_size(*comm_ptr, &size) >= 0);
    assert(MPI_Comm_rank(*comm_ptr, &rank) >= 0);

    status = cell::read_population_labels(*comm_ptr, input_file_name, pop_labels);
    assert (status >= 0);

    size_t n_nodes;
    map<CELL_IDX_T, pair<uint32_t,pop_t> > pop_ranges;
    vector<pop_range_t> pop_vector;
    assert(cell::read_population_ranges(*comm_ptr,
                                        string(input_file_name),
                                        pop_ranges, pop_vector,
                                        n_nodes) >= 0);

    PyObject *py_population_ranges_dict = PyDict_New();
    for (auto range: pop_ranges)
      {
        PyObject *py_range_tuple = PyTuple_New(2);
        PyTuple_SetItem(py_range_tuple, 0, PyLong_FromLong((long)range.first));
        PyTuple_SetItem(py_range_tuple, 1, PyLong_FromLong((long)range.second.first));

        if (range.second.second < pop_labels.size())
          {
            PyDict_SetItemString(py_population_ranges_dict,
                                 get<1>(pop_labels[range.second.second]).c_str(),
                                 py_range_tuple);
          }
      }

    PyObject *py_result_tuple = PyTuple_New(2);

    PyTuple_SetItem(py_result_tuple, 0, py_population_ranges_dict);
    PyTuple_SetItem(py_result_tuple, 1, PyLong_FromLong((long)n_nodes));
    
    return py_result_tuple;
  }

  
  static PyObject *py_read_projection_names (PyObject *self, PyObject *args)
  {
    int status;
    vector< pair<string,string> > prj_names;
    char *input_file_name;
    PyObject *py_comm = NULL;
    MPI_Comm *comm_ptr  = NULL;

    if (!PyArg_ParseTuple(args, "Os", &py_comm, &input_file_name))
      return NULL;

    assert(py_comm != NULL);
    comm_ptr = PyMPIComm_Get(py_comm);
    assert(comm_ptr != NULL);
    assert(*comm_ptr != MPI_COMM_NULL);

    assert(graph::read_projection_names(*comm_ptr, string(input_file_name), prj_names) >= 0);

    PyObject *py_result  = PyList_New(0);

    for (auto name_pair: prj_names)
      {
        PyObject *py_pairval = PyTuple_New(2);
        PyTuple_SetItem(py_pairval, 0, PyBytes_FromString(name_pair.first.c_str()));
        PyTuple_SetItem(py_pairval, 1, PyBytes_FromString(name_pair.second.c_str()));
        status = PyList_Append(py_result, py_pairval);
        assert (status == 0);
      }
    
    return py_result;
  }

  
  static PyObject *py_read_trees (PyObject *self, PyObject *args)
  {
    int status; size_t start=0, end=0;
    PyObject *py_cell_dict = PyDict_New();
    PyObject *py_comm = NULL;
    MPI_Comm *comm_ptr  = NULL;
    char *file_name, *pop_name;
    PyObject *py_attr_namespaces=NULL;

    if (!PyArg_ParseTuple(args, "Oss|O", &py_comm, &file_name,  &pop_name, &py_attr_namespaces))
      return NULL;

    assert(py_comm != NULL);
    comm_ptr = PyMPIComm_Get(py_comm);
    assert(comm_ptr != NULL);
    assert(*comm_ptr != MPI_COMM_NULL);

    vector <string> attr_namespaces;
    // Create C++ vector of namespace strings:
    if (py_attr_namespaces != NULL)
      {
        for (size_t i = 0; (Py_ssize_t)i < PyList_Size(py_attr_namespaces); i++)
          {
            PyObject *pyval = PyList_GetItem(py_attr_namespaces, (Py_ssize_t)i);
            char *str = PyBytes_AsString (pyval);
            attr_namespaces.push_back(string(str));
          }
      }

    vector<pair <pop_t, string> > pop_labels;
    status = cell::read_population_labels(*comm_ptr, string(file_name), pop_labels);
    assert (status >= 0);
    
    // Determine index of population to be read
    size_t pop_idx=0; bool pop_idx_set=false;
    for (size_t i=0; i<pop_labels.size(); i++)
      {
        if (get<1>(pop_labels[i]) == pop_name)
          {
            pop_idx = get<0>(pop_labels[i]);
            pop_idx_set = true;
          }
      }
    if (!pop_idx_set)
      {
        throw_err("Population not found");
      }

    map<CELL_IDX_T, pair<uint32_t,pop_t> > pop_ranges;
    vector<pop_range_t> pop_vector;
    size_t n_nodes;
    
    // Read population info
    assert(cell::read_population_ranges(*comm_ptr, string(file_name),
                                        pop_ranges, pop_vector,
                                        n_nodes) >= 0);


    vector<neurotree_t> tree_list;

    status = cell::read_trees (*comm_ptr, string(file_name),
                               string(pop_name), pop_vector[pop_idx].start,
                               tree_list, start, end);
    assert (status >= 0);
    map <string, NamedAttrMap> attr_maps;
    
    for (string attr_namespace : attr_namespaces)
      {
        data::NamedAttrMap attr_map;
        cell::read_cell_attributes(*comm_ptr, string(file_name), 
                                   attr_namespace, pop_name,
                                   pop_vector[pop_idx].start, attr_map);
        attr_maps.insert(make_pair(attr_namespace, attr_map));
      }

    for (size_t i = 0; i < tree_list.size(); i++)
      {
        const CELL_IDX_T idx = get<0>(tree_list[i]);
        const neurotree_t &tree = tree_list[i];
          
        PyObject *py_treeval = py_build_tree_value(idx, tree, attr_maps);

        PyDict_SetItem(py_cell_dict, PyLong_FromUnsignedLong(idx), py_treeval);
      }

    PyObject *py_result_tuple = PyTuple_New(2);
    PyTuple_SetItem(py_result_tuple, 0, py_cell_dict);
    PyTuple_SetItem(py_result_tuple, 1, PyLong_FromLong((long)n_nodes));

    return py_result_tuple;
  }


  
  static PyObject *py_scatter_read_trees (PyObject *self, PyObject *args, PyObject *kwds)
  {
    int status;
    PyObject *py_cell_dict = PyDict_New();
    PyObject *py_comm = NULL;
    MPI_Comm *comm_ptr  = NULL;
    unsigned long io_size = 0;
    char *file_name, *pop_name;
    PyObject *py_node_rank_map=NULL;
    PyObject *py_attr_namespaces=NULL;
    map<CELL_IDX_T, rank_t> node_rank_map;
    static const char *kwlist[] = {"comm",
                                   "file_name",
                                   "pop_name",
                                   "node_rank_map",
                                   "namespaces",
                                   "io_size",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Oss|OOk", (char **)kwlist,
                                     &py_comm, &file_name, &pop_name
                                     &py_node_rank_map, &py_attr_namespaces,
                                     &io_size))
      return NULL;

    assert(py_comm != NULL);
    comm_ptr = PyMPIComm_Get(py_comm);
    assert(comm_ptr != NULL);
    assert(*comm_ptr != MPI_COMM_NULL);

    int rank, size;
    assert(MPI_Comm_size(*comm_ptr, &size) >= 0);
    assert(MPI_Comm_rank(*comm_ptr, &rank) >= 0);

    if (io_size == 0)
      {
        io_size = size;
      }
    
    vector <string> attr_namespaces;
    map<CELL_IDX_T, pair<uint32_t,pop_t> > pop_ranges;
    vector<pop_range_t> pop_vector;
    vector< vector <string> > attr_names;
    size_t n_nodes;
    
    // Read population info
    assert(cell::read_population_ranges(*comm_ptr, string(file_name),
                                        pop_ranges, pop_vector,
                                        n_nodes) >= 0);
    // Create C++ vector of namespace strings:
    if (py_attr_namespaces != NULL)
      {
        for (size_t i = 0; (Py_ssize_t)i < PyList_Size(py_attr_namespaces); i++)
          {
            PyObject *pyval = PyList_GetItem(py_attr_namespaces, (Py_ssize_t)i);
            char *str = PyBytes_AsString (pyval);
            attr_namespaces.push_back(string(str));
          }
      }
    
    // Create C++ map for node_rank_map:
    if ((py_node_rank_map != NULL) && (py_node_rank_map != Py_None))
      {
        build_node_rank_map(py_node_rank_map, node_rank_map);
      }
    else
      {
        // round-robin node to rank assignment from file
        for (size_t i = 0; i < n_nodes; i++)
          {
            node_rank_map.insert(make_pair(i, i%size));
          }
      }
    
    vector<pair <pop_t, string> > pop_labels;
    status = cell::read_population_labels(*comm_ptr, string(file_name), pop_labels);
    assert (status >= 0);
    
    // Determine index of population to be read
    size_t pop_idx=0; bool pop_idx_set=false;
    for (size_t i=0; i<pop_labels.size(); i++)
      {
        if (get<1>(pop_labels[i]) == pop_name)
          {
            pop_idx = get<0>(pop_labels[i]);
            pop_idx_set = true;
          }
      }
    if (!pop_idx_set)
      {
        throw_err("Population not found");
      }
    

    map<CELL_IDX_T, neurotree_t> tree_map;
    map<string, NamedAttrMap> attr_maps;

    status = cell::scatter_read_trees (*comm_ptr, string(file_name),
                                       io_size, attr_namespaces,
                                       node_rank_map, string(pop_name),
                                       pop_vector[pop_idx].start,
                                       tree_map, attr_maps);
    assert (status >= 0);
    
    for (auto const& element : tree_map)
      {
        const CELL_IDX_T key = element.first;
        const neurotree_t &tree = element.second;

        PyObject *py_treeval = py_build_tree_value(key, tree, attr_maps);
        PyDict_SetItem(py_cell_dict, PyLong_FromUnsignedLong(key), py_treeval);
      }

    PyObject *py_result_tuple = PyTuple_New(2);
    PyTuple_SetItem(py_result_tuple, 0, py_cell_dict);
    PyTuple_SetItem(py_result_tuple, 1, PyLong_FromLong((long)n_nodes));

    return py_result_tuple;
  }
  
  static PyObject *py_read_tree_selection (PyObject *self, PyObject *args)
  {
    int status; 
    PyObject *py_cell_dict = PyDict_New();
    PyObject *py_comm = NULL;
    MPI_Comm *comm_ptr  = NULL;
    char *file_name, *pop_name;
    PyObject *py_attr_namespaces=NULL;
    PyObject *py_selection=NULL;
    vector <CELL_IDX_T> selection;

    if (!PyArg_ParseTuple(args, "OssO|O", &py_comm, &file_name,  &pop_name,
                          &py_selection, &py_attr_namespaces))
      return NULL;

    assert(py_comm != NULL);
    comm_ptr = PyMPIComm_Get(py_comm);
    assert(comm_ptr != NULL);
    assert(*comm_ptr != MPI_COMM_NULL);

    vector <string> attr_namespaces;
    // Create C++ vector of namespace strings:
    if (py_attr_namespaces != NULL)
      {
        for (size_t i = 0; (Py_ssize_t)i < PyList_Size(py_attr_namespaces); i++)
          {
            PyObject *pyval = PyList_GetItem(py_attr_namespaces, (Py_ssize_t)i);
            char *str = PyBytes_AsString (pyval);
            attr_namespaces.push_back(string(str));
          }
      }

    // Create C++ vector of selection indices:
    if (py_selection != NULL)
      {
        for (size_t i = 0; (Py_ssize_t)i < PyList_Size(py_selection); i++)
          {
            PyObject *pyval = PyList_GetItem(py_selection, (Py_ssize_t)i);
            CELL_IDX_T n = PyLong_AsLong(pyval);
            selection.push_back(n);
          }
      }

    vector<pair <pop_t, string> > pop_labels;
    status = cell::read_population_labels(*comm_ptr, string(file_name), pop_labels);
    assert (status >= 0);
    
    // Determine index of population to be read
    size_t pop_idx=0; bool pop_idx_set=false;
    for (size_t i=0; i<pop_labels.size(); i++)
      {
        if (get<1>(pop_labels[i]) == pop_name)
          {
            pop_idx = get<0>(pop_labels[i]);
            pop_idx_set = true;
          }
      }
    if (!pop_idx_set)
      {
        throw_err("Population not found");
      }

    map<CELL_IDX_T, pair<uint32_t,pop_t> > pop_ranges;
    vector<pop_range_t> pop_vector;
    size_t n_nodes;
    
    // Read population info
    assert(cell::read_population_ranges(*comm_ptr, string(file_name),
                                        pop_ranges, pop_vector,
                                        n_nodes) >= 0);


    vector<neurotree_t> tree_list;

    status = cell::read_tree_selection (string(file_name),
                                        string(pop_name), pop_vector[pop_idx].start,
                                        tree_list, selection);
    assert (status >= 0);
    map <string, NamedAttrMap> attr_maps;
    
    for (string attr_namespace : attr_namespaces)
      {
        data::NamedAttrMap attr_map;
        cell::read_cell_attribute_selection(string(file_name), 
                                            attr_namespace, pop_name,
                                            pop_vector[pop_idx].start,
                                            selection, attr_map);
        attr_maps.insert(make_pair(attr_namespace, attr_map));
      }

    for (size_t i = 0; i < tree_list.size(); i++)
      {
        const CELL_IDX_T idx = get<0>(tree_list[i]);
        const neurotree_t &tree = tree_list[i];

        PyObject *py_treeval = py_build_tree_value(idx, tree, attr_maps);

        PyDict_SetItem(py_cell_dict, PyLong_FromUnsignedLong(idx), py_treeval);
      }

    PyObject *py_result_tuple = PyTuple_New(2);
    PyTuple_SetItem(py_result_tuple, 0, py_cell_dict);
    PyTuple_SetItem(py_result_tuple, 1, PyLong_FromLong((long)n_nodes));

    return py_result_tuple;
  }

  
  static PyObject *py_read_cell_attributes (PyObject *self, PyObject *args, PyObject *kwds)
  {
    herr_t status;
    PyObject *py_comm = NULL;
    MPI_Comm *comm_ptr  = NULL;
    const string default_namespace = "Attributes";
    char *file_name, *pop_name, *namespace = (char *)default_namespace.c_str();
    
    static const char *kwlist[] = {"comm",
                                   "file_name",
                                   "pop_name",
                                   "namespace",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Oss|s", (char **)kwlist,
                                     &py_comm, &file_name,
                                     &pop_name, &namespace))
        return NULL;

    assert(py_comm != NULL);
    comm_ptr = PyMPIComm_Get(py_comm);
    assert(comm_ptr != NULL);
    assert(*comm_ptr != MPI_COMM_NULL);

    vector<pair <pop_t, string> > pop_labels;
    status = cell::read_population_labels(*comm_ptr, string(file_name), pop_labels);
    assert (status >= 0);
    
    // Determine index of population to be read
    size_t pop_idx=0; bool pop_idx_set=false;
    for (size_t i=0; i<pop_labels.size(); i++)
      {
        if (get<1>(pop_labels[i]) == pop_name)
          {
            pop_idx = get<0>(pop_labels[i]);
            pop_idx_set = true;
          }
      }
    if (!pop_idx_set)
      {
        throw_err("Population not found");
      }

    size_t n_nodes;
    map<CELL_IDX_T, pair<uint32_t,pop_t> > pop_ranges;
    vector<pop_range_t> pop_vector;
    assert(cell::read_population_ranges(*comm_ptr,
                                        string(file_name),
                                        pop_ranges, pop_vector,
                                        n_nodes) >= 0);


    NamedAttrMap attr_values;
    cell::read_cell_attributes (*comm_ptr,
                                string(file_name), string(namespace),
                                string(pop_name), pop_vector[pop_idx].start,
                                attr_values);
    vector<vector<string>> attr_names;
    attr_values.attr_names(attr_names);

    
    PyObject *py_idx_dict = PyDict_New();
    for (auto it = attr_values.index_set.begin(); it != attr_values.index_set.end(); ++it)
      {
        CELL_IDX_T idx = *it;

        PyObject *py_attr_dict = PyDict_New();

        py_attr_values<float> (idx,
                               attr_names[AttrMap::attr_index_float],
                               attr_values.attr_maps<float>(),
                               NPY_FLOAT,
                               py_attr_dict);
        py_attr_values<uint8_t> (idx,
                                 attr_names[AttrMap::attr_index_uint8],
                                 attr_values.attr_maps<uint8_t>(),
                                 NPY_UINT8,
                                 py_attr_dict);
        py_attr_values<int8_t> (idx,
                                attr_names[AttrMap::attr_index_int8],
                                attr_values.attr_maps<int8_t>(),
                                NPY_INT8,
                                py_attr_dict);
        py_attr_values<uint16_t> (idx,
                                  attr_names[AttrMap::attr_index_uint16],
                                  attr_values.attr_maps<uint16_t>(),
                                  NPY_UINT16,
                                  py_attr_dict);
        py_attr_values<int16_t> (idx,
                                 attr_names[AttrMap::attr_index_int16],
                                 attr_values.attr_maps<int16_t>(),
                                 NPY_INT16,
                                 py_attr_dict);
        py_attr_values<uint32_t> (idx,
                                  attr_names[AttrMap::attr_index_uint32],
                                  attr_values.attr_maps<uint32_t>(),
                                  NPY_UINT32,
                                  py_attr_dict);
        py_attr_values<int32_t> (idx,
                                 attr_names[AttrMap::attr_index_int32],
                                 attr_values.attr_maps<int32_t>(),
                                 NPY_INT32,
                                 py_attr_dict);

        PyDict_SetItem(py_idx_dict, PyLong_FromUnsignedLong(idx), py_attr_dict);

      }
    
    return py_idx_dict;
  }

  
  static PyObject *py_read_cell_attribute_selection (PyObject *self, PyObject *args, PyObject *kwds)
  {
    herr_t status;
    PyObject *py_comm = NULL;
    MPI_Comm *comm_ptr  = NULL;
    const string default_namespace = "Attributes";
    char *file_name, *pop_name, *namespace = (char *)default_namespace.c_str();
    PyObject *py_selection = NULL;
    vector <CELL_IDX_T> selection;
    
    static const char *kwlist[] = {"comm",
                                   "file_name",
                                   "pop_name",
                                   "selection",
                                   "namespace",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OssO|s", (char **)kwlist,
                                     &py_comm, &file_name,
                                     &pop_name, &py_selection,
                                     &namespace))
        return NULL;

    assert(py_comm != NULL);
    comm_ptr = PyMPIComm_Get(py_comm);
    assert(comm_ptr != NULL);
    assert(*comm_ptr != MPI_COMM_NULL);
    
    vector<pair <pop_t, string> > pop_labels;
    status = cell::read_population_labels(*comm_ptr, string(file_name), pop_labels);
    assert (status >= 0);
    
    // Determine index of population to be read
    size_t pop_idx=0; bool pop_idx_set=false;
    for (size_t i=0; i<pop_labels.size(); i++)
      {
        if (get<1>(pop_labels[i]) == pop_name)
          {
            pop_idx = get<0>(pop_labels[i]);
            pop_idx_set = true;
          }
      }
    if (!pop_idx_set)
      {
        throw_err("Population not found");
      }

    // Create C++ vector of selection indices:
    if (py_selection != NULL)
      {
        for (size_t i = 0; (Py_ssize_t)i < PyList_Size(py_selection); i++)
          {
            PyObject *pyval = PyList_GetItem(py_selection, (Py_ssize_t)i);
            CELL_IDX_T n = PyLong_AsLong(pyval);
            selection.push_back(n);
          }
      }

    size_t n_nodes;
    map<CELL_IDX_T, pair<uint32_t,pop_t> > pop_ranges;
    vector<pop_range_t> pop_vector;
    assert(cell::read_population_ranges(*comm_ptr,
                                        string(file_name),
                                        pop_ranges, pop_vector,
                                        n_nodes) >= 0);


    NamedAttrMap attr_values;
    cell::read_cell_attribute_selection (string(file_name), string(namespace),
                                         string(pop_name), pop_vector[pop_idx].start,
                                         selection, attr_values);
    vector<vector<string>> attr_names;
    attr_values.attr_names(attr_names);

    
    PyObject *py_idx_dict = PyDict_New();
    for (auto it = attr_values.index_set.begin(); it != attr_values.index_set.end(); ++it)
      {
        CELL_IDX_T idx = *it;

        PyObject *py_attr_dict = PyDict_New();

        py_attr_values<float> (idx,
                               attr_names[AttrMap::attr_index_float],
                               attr_values.attr_maps<float>(),
                               NPY_FLOAT,
                               py_attr_dict);
        py_attr_values<uint8_t> (idx,
                                 attr_names[AttrMap::attr_index_uint8],
                                 attr_values.attr_maps<uint8_t>(),
                                 NPY_UINT8,
                                 py_attr_dict);
        py_attr_values<int8_t> (idx,
                                attr_names[AttrMap::attr_index_int8],
                                attr_values.attr_maps<int8_t>(),
                                NPY_INT8,
                                py_attr_dict);
        py_attr_values<uint16_t> (idx,
                                  attr_names[AttrMap::attr_index_uint16],
                                  attr_values.attr_maps<uint16_t>(),
                                  NPY_UINT16,
                                  py_attr_dict);
        py_attr_values<int16_t> (idx,
                                 attr_names[AttrMap::attr_index_int16],
                                 attr_values.attr_maps<int16_t>(),
                                 NPY_INT16,
                                 py_attr_dict);
        py_attr_values<uint32_t> (idx,
                                  attr_names[AttrMap::attr_index_uint32],
                                  attr_values.attr_maps<uint32_t>(),
                                  NPY_UINT32,
                                  py_attr_dict);
        py_attr_values<int32_t> (idx,
                                 attr_names[AttrMap::attr_index_int32],
                                 attr_values.attr_maps<int32_t>(),
                                 NPY_INT32,
                                 py_attr_dict);

        PyDict_SetItem(py_idx_dict, PyLong_FromUnsignedLong(idx), py_attr_dict);

      }
    
    return py_idx_dict;
  }

  
  static PyObject *py_bcast_cell_attributes (PyObject *self, PyObject *args, PyObject *kwds)
  {
    herr_t status;
    unsigned long root;
    PyObject *py_comm = NULL;
    MPI_Comm *comm_ptr  = NULL;
    const string default_namespace = "Attributes";
    char *file_name, *pop_name, *namespace = (char *)default_namespace.c_str();
    NamedAttrMap attr_values;
    
    static const char *kwlist[] = {"comm",
                                   "root",
                                   "file_name",
                                   "pop_name",
                                   "namespace",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Pkss|s", (char **)kwlist,
                                     &py_comm, &root, &file_name,
                                     &pop_name, &namespace))
        return NULL;

    assert(py_comm != NULL);
    comm_ptr = PyMPIComm_Get(py_comm);
    assert(comm_ptr != NULL);
    assert(*comm_ptr != MPI_COMM_NULL);

    vector<pair <pop_t, string> > pop_labels;
    status = cell::read_population_labels(*comm_ptr, string(file_name), pop_labels);
    assert (status >= 0);
    
    // Determine index of population to be read
    size_t pop_idx=0; bool pop_idx_set=false;
    for (size_t i=0; i<pop_labels.size(); i++)
      {
        if (get<1>(pop_labels[i]) == pop_name)
          {
            pop_idx = get<0>(pop_labels[i]);
            pop_idx_set = true;
          }
      }
    if (!pop_idx_set)
      {
        throw_err("Population not found");
      }

    size_t n_nodes;
    map<CELL_IDX_T, pair<uint32_t,pop_t> > pop_ranges;
    vector<pop_range_t> pop_vector;
    assert(cell::read_population_ranges(*comm_ptr,
                                        string(file_name),
                                        pop_ranges, pop_vector,
                                        n_nodes) >= 0);


    cell::bcast_cell_attributes (*comm_ptr, (int)root,
                                 string(file_name), string(namespace),
                                 string(pop_name), pop_vector[pop_idx].start,
                                 attr_values);
    vector<vector<string>> attr_names;
    attr_values.attr_names(attr_names);

    
    PyObject *py_idx_dict = PyDict_New();
    for (auto it = attr_values.index_set.begin(); it != attr_values.index_set.end(); ++it)
      {
        CELL_IDX_T idx = *it;

        PyObject *py_attr_dict = PyDict_New();

        py_attr_values<float> (idx,
                               attr_names[AttrMap::attr_index_float],
                               attr_values.attr_maps<float>(),
                               NPY_FLOAT,
                               py_attr_dict);
        py_attr_values<uint8_t> (idx,
                                 attr_names[AttrMap::attr_index_uint8],
                                 attr_values.attr_maps<uint8_t>(),
                                 NPY_UINT8,
                                 py_attr_dict);
        py_attr_values<int8_t> (idx,
                                attr_names[AttrMap::attr_index_int8],
                                attr_values.attr_maps<int8_t>(),
                                NPY_INT8,
                                py_attr_dict);
        py_attr_values<uint16_t> (idx,
                                  attr_names[AttrMap::attr_index_uint16],
                                  attr_values.attr_maps<uint16_t>(),
                                  NPY_UINT16,
                                  py_attr_dict);
        py_attr_values<int16_t> (idx,
                                 attr_names[AttrMap::attr_index_int16],
                                 attr_values.attr_maps<int16_t>(),
                                 NPY_INT16,
                                 py_attr_dict);
        py_attr_values<uint32_t> (idx,
                                  attr_names[AttrMap::attr_index_uint32],
                                  attr_values.attr_maps<uint32_t>(),
                                  NPY_UINT32,
                                  py_attr_dict);
        py_attr_values<int32_t> (idx,
                                 attr_names[AttrMap::attr_index_int32],
                                 attr_values.attr_maps<int32_t>(),
                                 NPY_INT32,
                                 py_attr_dict);

        PyDict_SetItem(py_idx_dict, PyLong_FromUnsignedLong(idx), py_attr_dict);

      }
    
    return py_idx_dict;
  }

  
  static PyObject *py_write_cell_attributes (PyObject *self, PyObject *args, PyObject *kwds)
  {
    PyObject *idx_values;
    PyObject *py_comm = NULL;
    MPI_Comm *comm_ptr  = NULL;
    const string default_namespace = "Attributes";
    char *file_name_arg, *pop_name_arg, *namespace_arg = (char *)default_namespace.c_str();
    
    static const char *kwlist[] = {"comm",
                                   "file_name",
                                   "pop_name",
                                   "values",
                                   "namespace",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OssO|s", (char **)kwlist,
                                     &py_comm, &file_name_arg, &pop_name_arg, &idx_values,
                                     &namespace_arg))
        return NULL;

    assert(py_comm != NULL);
    comm_ptr = PyMPIComm_Get(py_comm);
    assert(comm_ptr != NULL);
    assert(*comm_ptr != MPI_COMM_NULL);

    string file_name = string(file_name_arg);
    string pop_name = string(pop_name_arg);
    string attr_namespace = string(namespace_arg);
    
    int npy_type=0;
    
    vector<string> attr_names;
    vector<int> attr_types;
        
    vector < map <CELL_IDX_T, vector<uint32_t> > > all_attr_values_uint32;
    vector < map <CELL_IDX_T, vector<int32_t> > >  all_attr_values_int32;
    vector < map <CELL_IDX_T, vector<uint16_t> > > all_attr_values_uint16;
    vector < map <CELL_IDX_T, vector<int16_t> > >  all_attr_values_int16;
    vector < map <CELL_IDX_T, vector<uint8_t> > >  all_attr_values_uint8;
    vector < map <CELL_IDX_T, vector<int8_t> > >   all_attr_values_int8;
    vector < map <CELL_IDX_T, vector<float> > >    all_attr_values_float;

    build_cell_attr_value_maps(idx_values,
                               attr_names,
                               attr_types,
                               all_attr_values_uint32,
                               all_attr_values_uint16,
                               all_attr_values_uint8,
                               all_attr_values_int32,
                               all_attr_values_int16,
                               all_attr_values_int8,
                               all_attr_values_float);
    
    const data::optional_hid dflt_data_type;
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
              cell::write_cell_attribute_map<uint32_t> (*comm_ptr, file_name, attr_namespace, pop_name, 
                                                        attr_name, all_attr_values_uint32[attr_type_idx[AttrMap::attr_index_uint32]],
                                                        dflt_data_type);
              attr_type_idx[AttrMap::attr_index_uint32]++;
              break;
            }
          case NPY_UINT16:
            {
              cell::write_cell_attribute_map<uint16_t> (*comm_ptr, file_name, attr_namespace, pop_name, 
                                                        attr_name, all_attr_values_uint16[attr_type_idx[AttrMap::attr_index_uint16]],
                                                        dflt_data_type);
              attr_type_idx[AttrMap::attr_index_uint16]++;
              break;
            }
          case NPY_UINT8:
            {
              cell::write_cell_attribute_map<uint8_t> (*comm_ptr, file_name, attr_namespace, pop_name, 
                                                       attr_name, all_attr_values_uint8[attr_type_idx[AttrMap::attr_index_uint8]],
                                                       dflt_data_type);
              attr_type_idx[AttrMap::attr_index_uint8]++;
              break;
            }
          case NPY_INT32:
            {
              cell::write_cell_attribute_map<uint32_t> (*comm_ptr, file_name, attr_namespace, pop_name, 
                                                        attr_name, all_attr_values_int32[attr_type_idx[AttrMap::attr_index_int32]],
                                                        dflt_data_type);
              attr_type_idx[AttrMap::attr_index_int32]++;
              break;
            }
          case NPY_INT16:
            {
              cell::write_cell_attribute_map<int16_t> (*comm_ptr, file_name, attr_namespace, pop_name, 
                                                        attr_name, all_attr_values_int16[attr_type_idx[AttrMap::attr_index_int16]],
                                                        dflt_data_type);
              attr_type_idx[AttrMap::attr_index_int16]++;
              break;
            }
          case NPY_INT8:
            {
              cell::write_cell_attribute_map<int8_t> (*comm_ptr, file_name, attr_namespace, pop_name, 
                                                      attr_name, all_attr_values_uint8[attr_type_idx[AttrMap::attr_index_int8]],
                                                      dflt_data_type);
              attr_type_idx[AttrMap::attr_index_int8]++;
              break;
            }
          case NPY_FLOAT:
            {
              cell::write_cell_attribute_map<float> (*comm_ptr, file_name, attr_namespace, pop_name, 
                                                     attr_name, all_attr_values_float[attr_type_idx[AttrMap::attr_index_float]],
                                                     dflt_data_type);
              attr_type_idx[AttrMap::attr_index_float]++;
              break;
            }
          default:
            throw runtime_error("Unsupported attribute type");
            break;
          }
      }
    
    Py_INCREF(Py_None);
    return Py_None;
  }
  
  
  static PyObject *py_append_cell_attributes (PyObject *self, PyObject *args, PyObject *kwds)
  {
    MPI_Comm data_comm;
    PyObject *idx_values;
    const unsigned long default_cache_size = 4*1024*1024;
    const unsigned long default_chunk_size = 4000;
    const unsigned long default_value_chunk_size = 4000;
    const string default_namespace = "Attributes";
    PyObject *py_comm = NULL;
    MPI_Comm *comm_ptr  = NULL;
    unsigned long io_size = 0;
    unsigned long chunk_size = default_chunk_size;
    unsigned long value_chunk_size = default_value_chunk_size;
    unsigned long cache_size = default_cache_size;
    char *file_name_arg, *pop_name_arg, *namespace_arg = (char *)default_namespace.c_str();
    herr_t status;
    
    static const char *kwlist[] = {"comm",
                                   "file_name",
                                   "pop_name",
                                   "values",
                                   "namespace",
                                   "io_size",
                                   "chunk_size",
                                   "value_chunk_size",
                                   "cache_size",
                                   NULL};


    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OssO|skkkkk", (char **)kwlist,
                                     &py_comm, &file_name_arg, &pop_name_arg, &idx_values,
                                     &namespace_arg,
                                     &io_size, &chunk_size, &value_chunk_size, &cache_size))
        return NULL;

    assert(py_comm != NULL);
    comm_ptr = PyMPIComm_Get(py_comm);
    assert(comm_ptr != NULL);
    assert(*comm_ptr != MPI_COMM_NULL);

    Py_ssize_t dict_size = PyDict_Size(idx_values);
    int data_color = 2;

    // In cases where some ranks do not have any data to write, split
    // the communicator, so that collective operations can be executed
    // only on the ranks that do have data.
    if (dict_size > 0)
      {
        MPI_Comm_split(*comm_ptr,data_color,0,&data_comm);
      }
    else
      {
        MPI_Comm_split(*comm_ptr,0,0,&data_comm);
      }
    MPI_Comm_set_errhandler(data_comm, MPI_ERRORS_RETURN);
    
    
    int srank, ssize; size_t size;
    assert(MPI_Comm_size(data_comm, &ssize) >= 0);
    assert(MPI_Comm_rank(data_comm, &srank) >= 0);
    assert(ssize > 0);
    assert(srank >= 0);
    size = ssize;
    
    if ((io_size == 0) || (io_size > size))
      {
        io_size = size;
      }
    assert(io_size <= size);
    
    string file_name      = string(file_name_arg);
    string pop_name       = string(pop_name_arg);
    string attr_namespace = string(namespace_arg);
    
    int npy_type=0;
    
    vector<string> attr_names;
    vector<int> attr_types;
        
    vector< map<CELL_IDX_T, vector<uint32_t> >> all_attr_values_uint32;
    vector< map<CELL_IDX_T, vector<int32_t> >> all_attr_values_int32;
    vector< map<CELL_IDX_T, vector<uint16_t> >> all_attr_values_uint16;
    vector< map<CELL_IDX_T, vector<int16_t> >> all_attr_values_int16;
    vector< map<CELL_IDX_T, vector<uint8_t> >>  all_attr_values_uint8;
    vector< map<CELL_IDX_T, vector<int8_t> >>  all_attr_values_int8;
    vector< map<CELL_IDX_T, vector<float> >>  all_attr_values_float;

    build_cell_attr_value_maps(idx_values,
                               attr_names,
                               attr_types,
                               all_attr_values_uint32,
                               all_attr_values_uint16,
                               all_attr_values_uint8,
                               all_attr_values_int32,
                               all_attr_values_int16,
                               all_attr_values_int8,
                               all_attr_values_float);

    if (access( file_name.c_str(), F_OK ) != 0)
      {
        vector <string> groups;
        groups.push_back (hdf5::POPULATIONS);
        status = hdf5::create_file_toplevel (data_comm, file_name, groups);
      }
    else
      {
        status = 0;
      }
    assert(status == 0);
    MPI_Barrier(data_comm);

    const data::optional_hid dflt_data_type;
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
              cell::append_cell_attribute_map<uint32_t> (data_comm, file_name, attr_namespace, pop_name, attr_name,
                                                         all_attr_values_uint32[attr_type_idx[AttrMap::attr_index_uint32]],
                                                         io_size, dflt_data_type);
              attr_type_idx[AttrMap::attr_index_uint32]++;
              break;
            }
          case NPY_INT32:
            {
              cell::append_cell_attribute_map<int32_t> (data_comm, file_name, attr_namespace, pop_name, attr_name,
                                                         all_attr_values_int32[attr_type_idx[AttrMap::attr_index_int32]],
                                                         io_size, dflt_data_type);
              attr_type_idx[AttrMap::attr_index_int32]++;
              break;
            }
          case NPY_UINT16:
            {
              cell::append_cell_attribute_map<uint16_t> (data_comm, file_name, attr_namespace, pop_name, attr_name,
                                                         all_attr_values_uint16[attr_type_idx[AttrMap::attr_index_uint16]],
                                                         io_size, dflt_data_type);
              attr_type_idx[AttrMap::attr_index_uint16]++;
              break;
            }
          case NPY_INT16:
            {
              cell::append_cell_attribute_map<int16_t> (data_comm, file_name, attr_namespace, pop_name, attr_name,
                                                        all_attr_values_int16[attr_type_idx[AttrMap::attr_index_int16]],
                                                        io_size, dflt_data_type);
              attr_type_idx[AttrMap::attr_index_int16]++;
              break;
            }
          case NPY_UINT8:
            {
              cell::append_cell_attribute_map<uint8_t> (data_comm, file_name, attr_namespace, pop_name, attr_name,
                                                        all_attr_values_uint8[attr_type_idx[AttrMap::attr_index_uint8]],
                                                        io_size, dflt_data_type);
              attr_type_idx[AttrMap::attr_index_uint8]++;
              break;
            }
          case NPY_INT8:
            {
              cell::append_cell_attribute_map<int8_t> (data_comm, file_name, attr_namespace, pop_name, attr_name,
                                                       all_attr_values_int8[attr_type_idx[AttrMap::attr_index_int8]],
                                                       io_size, dflt_data_type);
              attr_type_idx[AttrMap::attr_index_int8]++;
              break;
            }
          case NPY_FLOAT:
            {
              cell::append_cell_attribute_map<float> (data_comm, file_name, attr_namespace, pop_name, attr_name,
                                                      all_attr_values_float[attr_type_idx[AttrMap::attr_index_float]],
                                                      io_size, dflt_data_type);
              attr_type_idx[AttrMap::attr_index_float]++;
              break;
            }
          default:
            throw runtime_error("Unsupported attribute type");
            break;
          }
      }

    assert(MPI_Comm_free(&data_comm) == MPI_SUCCESS);
    Py_INCREF(Py_None);
    return Py_None;
  }


  static PyObject *py_append_cell_trees (PyObject *self, PyObject *args, PyObject *kwds)
  {
    MPI_Comm data_comm;
    PyObject *idx_values;
    const unsigned long default_cache_size = 4*1024*1024;
    const unsigned long default_chunk_size = 4000;
    const unsigned long default_value_chunk_size = 4000;
    PyObject *py_comm = NULL;
    MPI_Comm *comm_ptr  = NULL;

    unsigned long create_index = 0, io_size = 0;
    unsigned long chunk_size = default_chunk_size;
    unsigned long value_chunk_size = default_value_chunk_size;
    unsigned long cache_size = default_cache_size;
    char *file_name_arg, *pop_name_arg;
    
    static const char *kwlist[] = {"comm",
                                   "file_name",
                                   "pop_name",
                                   "values",
                                   "create_index",
                                   "io_size",
                                   "chunk_size",
                                   "value_chunk_size",
                                   "cache_size",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OssO|kkkkkk", (char **)kwlist,
                                     &py_comm, &file_name_arg, &pop_name_arg, &idx_values,
                                     &create_index, &io_size, &chunk_size, &value_chunk_size, &cache_size))
        return NULL;

    assert(py_comm != NULL);
    comm_ptr = PyMPIComm_Get(py_comm);
    assert(comm_ptr != NULL);
    assert(*comm_ptr != MPI_COMM_NULL);

    Py_ssize_t dict_size = PyDict_Size(idx_values);
    int data_color = 2;

    // In cases where some ranks do not have any data to write, split
    // the communicator, so that collective operations can be executed
    // only on the ranks that do have data.
    if (dict_size > 0)
      {
        MPI_Comm_split(*comm_ptr,data_color,0,&data_comm);
      }
    else
      {
        MPI_Comm_split(*comm_ptr,0,0,&data_comm);
      }
    MPI_Comm_set_errhandler(data_comm, MPI_ERRORS_RETURN);
    
    
    int srank, ssize; size_t size;
    assert(MPI_Comm_size(data_comm, &ssize) >= 0);
    assert(MPI_Comm_rank(data_comm, &srank) >= 0);
    assert(ssize > 0);
    assert(srank >= 0);
    size = ssize;
    
    if ((io_size == 0) || (io_size > size))
      {
        io_size = size;
      }
    assert(io_size <= size);

    
    string file_name      = string(file_name_arg);
    string pop_name       = string(pop_name_arg);
    
    vector<string> attr_names;
    vector<int> attr_types;
        
    vector< map<CELL_IDX_T, vector<uint32_t> >> all_attr_values_uint32;
    vector< map<CELL_IDX_T, vector<int32_t> >> all_attr_values_int32;
    vector< map<CELL_IDX_T, vector<uint16_t> >> all_attr_values_uint16;
    vector< map<CELL_IDX_T, vector<int16_t> >> all_attr_values_int16;
    vector< map<CELL_IDX_T, vector<uint8_t> >>  all_attr_values_uint8;
    vector< map<CELL_IDX_T, vector<int8_t> >>  all_attr_values_int8;
    vector< map<CELL_IDX_T, vector<float> >>  all_attr_values_float;

    vector<neurotree_t> tree_list;
    
    build_cell_attr_value_maps(idx_values,
                               attr_names,
                               attr_types,
                               all_attr_values_uint32,
                               all_attr_values_uint16,
                               all_attr_values_uint8,
                               all_attr_values_int32,
                               all_attr_values_int16,
                               all_attr_values_int8,
                               all_attr_values_float);

    assert(cell::append_trees (data_comm, file_name, pop_name, tree_list) >= 0);
    
    assert(MPI_Comm_free(&data_comm) == MPI_SUCCESS);
    Py_INCREF(Py_None);
    return Py_None;
  }
  
  enum seq_pos {seq_next, seq_last, seq_empty, seq_done};
  
  /* NeuroH5ProjectionGenState - neurograph generator instance.
   *
   * file_name: input file name
   * src_pop: source population name
   * dst_pop: destination population name
   * namespace: attribute namespace
   * node_rank_map: used to assign edges to MPI ranks
   * seq_index: index of the next edge in the sequence to yield
   * start_index: starting index of the next batch of edges to read from file
   * cache_size: how many edge blocks to read from file at at time
   *
   */
  typedef struct {
    Py_ssize_t edge_index, edge_count, block_index, block_count, cache_index, cache_size, io_size, comm_size;

    string file_name;
    MPI_Comm *comm_ptr;

    seq_pos pos;
    graph::EdgeMapType edge_map_type;
    map<NODE_IDX_T, rank_t> node_rank_map;
    vector<pop_range_t> pop_vector;
    map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
    set< pair<pop_t, pop_t> > pop_pairs;
    vector<pair <pop_t, string> > pop_labels;
    vector<pair<string,string> > prj_names;
    edge_map_t edge_map;
    edge_map_iter_t edge_map_iter;
    vector<NODE_IDX_T>::const_iterator edge_iter;
    vector< pair<string,hid_t> >  edge_attr_info;
    vector<size_t> edge_attr_num;
    vector< vector<string> >  edge_attr_names;
    vector<string> edge_attr_namespaces;
    string src_pop_name, dst_pop_name;
    size_t total_num_nodes, local_num_nodes, total_num_edges, local_num_edges;


  } NeuroH5ProjectionGenState;

  typedef struct {
    PyObject_HEAD
    NeuroH5ProjectionGenState *state;
  } PyNeuroH5ProjectionGenState;
  
  /* NeuroH5TreeGenState - tree generator instance.
   *
   * file_name: input file name
   * pop_name: population name
   * namespace: attribute namespace
   * node_rank_map: used to assign trees to MPI ranks
   * seq_index: index of the next tree in the sequence to yield
   * start_index: starting index of the next batch of trees to read from file
   * cache_size: how many trees to read from file at at time
   *
   */
  typedef struct {
    Py_ssize_t seq_index, cache_index, cache_size, io_size, comm_size, count;
    seq_pos pos;
    string pop_name;
    size_t pop_idx;
    string file_name;
    MPI_Comm *comm_ptr;
    vector<pop_range_t> pop_vector;
    map<CELL_IDX_T, neurotree_t> tree_map;
    vector<string> attr_namespaces;
    map <string, NamedAttrMap> attr_maps;
    map <string, vector< vector <string> > > attr_names;
    map<CELL_IDX_T, neurotree_t>::const_iterator it_tree;
    map<CELL_IDX_T, rank_t> node_rank_map; 
  } NeuroH5TreeGenState;

  typedef struct {
    PyObject_HEAD
    NeuroH5TreeGenState *state;
  } PyNeuroH5TreeGenState;

  
  /* NeuroH5CellAttrGenState - cell attribute generator instance.
   *
   * file_name: input file name
   * pop_name: population name
   * namespace: attribute namespace
   * node_rank_map: used to assign trees to MPI ranks
   * seq_index: index of the next tree in the sequence to yield
   * start_index: starting index of the next batch of trees to read from file
   * cache_size: how many trees to read from file at at time
   *
   */
  typedef struct {
    Py_ssize_t seq_index, cache_index, cache_size, io_size, comm_size, count;
    seq_pos pos;
    string pop_name;
    size_t pop_idx;
    string file_name;
    string namespace;
    MPI_Comm* comm_ptr;
    vector<pop_range_t> pop_vector;
    string attr_namespace;
    NamedAttrMap attr_map;
    vector< vector <string> > attr_names;
    set<CELL_IDX_T>::const_iterator it_idx;
    map <CELL_IDX_T, rank_t> node_rank_map;
  } NeuroH5CellAttrGenState;
  
  typedef struct {
    PyObject_HEAD
    NeuroH5CellAttrGenState *state;
  } PyNeuroH5CellAttrGenState;

  
  static PyObject *
  neuroh5_prj_gen_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
  {
    int status, opt_edge_map_type=0;
    graph::EdgeMapType edge_map_type = graph::EdgeMapDst;
    PyObject *py_comm = NULL;
    MPI_Comm *comm_ptr  = NULL;
    unsigned int io_size=0, cache_size=1;
    char *file_name, *src_pop_name, *dst_pop_name;
    PyObject* py_attr_namespaces = NULL;
    vector<string> attr_namespaces;
    vector<pop_range_t> pop_vector;
    map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
    set< pair<pop_t, pop_t> > pop_pairs;
    vector<pair <pop_t, string> > pop_labels;
    vector<pair<string,string> > prj_names;
    vector< pair<string,hid_t> >  edge_attr_info;
    vector<size_t> edge_attr_num;
    size_t total_num_nodes;

    edge_attr_num.resize(data::AttrVal::num_attr_types, 0);
    
    static const char *kwlist[] = {"comm",
                                   "file_name",
                                   "src_pop_name",
                                   "dst_pop_name",
                                   "namespaces",
                                   "edge_map_type",
                                   "io_size",
                                   "cache_size",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Osss|Oikk", (char **)kwlist,
                                     &py_comm, &file_name, &src_pop_name, &dst_pop_name, 
                                     &py_attr_namespaces, &opt_edge_map_type, 
                                     &io_size, &cache_size))
      return NULL;
    
    assert(py_comm != NULL);
    comm_ptr = PyMPIComm_Get(py_comm);
    assert(comm_ptr != NULL);
    assert(*comm_ptr != MPI_COMM_NULL);
    
    if (opt_edge_map_type == 1)
      {
        edge_map_type = graph::EdgeMapSrc;
      }
    
    int size, rank;
    assert(MPI_Comm_size(*comm_ptr, &size) >= 0);
    assert(MPI_Comm_rank(*comm_ptr, &rank) >= 0);

    if (io_size == 0)
      {
        io_size = size;
      }

    vector <string> attr_namespaces;
    // Create C++ vector of namespace strings:
    if (py_attr_namespaces != NULL)
      {
        for (size_t i = 0; (Py_ssize_t)i < PyList_Size(py_attr_namespaces); i++)
          {
            PyObject *pyval = PyList_GetItem(py_attr_namespaces, (Py_ssize_t)i);
            char *str = PyBytes_AsString (pyval);
            attr_namespaces.push_back(string(str));
          }
      }
    
    assert(graph::read_projection_names(*comm_ptr, string(file_name), prj_names) >= 0);

    // Read population info to determine total_num_nodes
    assert(cell::read_population_ranges(*comm_ptr, string(file_name),
                                        pop_ranges, pop_vector, total_num_nodes) >= 0);
    assert(cell::read_population_labels(*comm_ptr, file_name, pop_labels) >= 0);
    assert(cell::read_population_combos(*comm_ptr, string(file_name), pop_pairs) >= 0);

    if (attr_namespaces.size()>0)
      {

        assert(graph::get_edge_attributes(file_name, src_pop_name, dst_pop_name, attr_namespace,
                                          edge_attr_info) >= 0);
        assert(graph::num_edge_attributes(edge_attr_info, edge_attr_num) >= 0);
        
        assert(MPI_Bcast(&edge_attr_num[0], edge_attr_num.size(), MPI_UINT32_T, 0, *comm_ptr) == MPI_SUCCESS);
      }
    
    hsize_t num_blocks = hdf5::num_projection_blocks(*comm_ptr, string(file_name),
                                                     src_pop_name, dst_pop_name);


    /* Create a new generator state and initialize it */
    PyNeuroH5ProjectionGenState *py_ngg = (PyNeuroH5ProjectionGenState *)type->tp_alloc(type, 0);
    if (!py_ngg) return NULL;
    py_ngg->state = new NeuroH5ProjectionGenState();

    map<CELL_IDX_T, rank_t> node_rank_map;
    // Create C++ map for node_rank_map:
    // round-robin node to rank assignment from file
    rank_t r=0; 
    for (size_t i = 0; i < total_num_nodes; i++)
      {
        py_ngg->state->node_rank_map.insert(make_pair(i, r++));
        if ((unsigned int)size <= r) r=0;
      }
    

    py_ngg->state->pos             = seq_next;
    py_ngg->state->edge_index      = 0;
    py_ngg->state->edge_count      = 0;
    py_ngg->state->block_index     = 0;
    py_ngg->state->block_count     = num_blocks;
    py_ngg->state->cache_size      = cache_size;
    py_ngg->state->io_size         = io_size;
    py_ngg->state->comm_ptr        = comm_ptr;
    py_ngg->state->file_name       = string(file_name);
    py_ngg->state->src_pop_name    = string(src_pop_name);
    py_ngg->state->dst_pop_name    = string(dst_pop_name);
    py_ngg->state->pop_vector      = pop_vector;
    py_ngg->state->pop_ranges      = pop_ranges;
    py_ngg->state->pop_pairs       = pop_pairs;
    py_ngg->state->pop_labels      = pop_labels;
    py_ngg->state->edge_map_iter   = py_ngg->state->edge_map.cbegin();
    py_ngg->state->edge_map_type   = edge_map_type;
    py_ngg->state->edge_attr_info  = edge_attr_info;
    py_ngg->state->edge_attr_num   = edge_attr_num;
    py_ngg->state->edge_attr_namespaces = attr_namespaces;
    py_ngg->state->total_num_nodes = total_num_nodes;
    py_ngg->state->local_num_nodes = 0;
    py_ngg->state->total_num_edges = 0;
    py_ngg->state->local_num_edges = 0;
    
    return (PyObject *)py_ngg;
    
  }
  
  static PyObject *
  neuroh5_tree_gen_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
  {
    int status;
    PyObject *py_comm = NULL;
    MPI_Comm *comm_ptr  = NULL;
    unsigned int io_size, cache_size=100;
    char *file_name, *pop_name;
    PyObject* py_attr_namespaces = NULL;
    vector<string> attr_namespaces;

    static const char *kwlist[] = {"comm",
                                   "file_name",
                                   "pop_name",
                                   "namespaces",
                                   "io_size",
                                   "cache_size",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OssI|iOi", (char **)kwlist,
                                     &py_comm, &file_name, &pop_name, 
                                     &py_attr_namespaces,
                                     &io_size, &cache_size))
      return NULL;

    assert(py_comm != NULL);
    comm_ptr = PyMPIComm_Get(py_comm);
    assert(comm_ptr != NULL);
    assert(*comm_ptr != MPI_COMM_NULL);

    int rank, size;
    assert(MPI_Comm_size(*comm_ptr, &size) >= 0);
    assert(MPI_Comm_rank(*comm_ptr, &rank) >= 0);

    assert(size > 0);
    
    if ((size > 0) && (io_size < (unsigned int)size))
      io_size = size;
    
    if ((size > 0) && (cache_size < (unsigned int)size))
      cache_size = size;

    // Create C++ vector of namespace strings:
    if (py_attr_namespaces != NULL)
      {
        for (size_t i = 0; (Py_ssize_t)i < PyList_Size(py_attr_namespaces); i++)
          {
            PyObject *pyval = PyList_GetItem(py_attr_namespaces, (Py_ssize_t)i);
            char *str = PyBytes_AsString (pyval);
            attr_namespaces.push_back(string(str));
          }
      }
    
    vector<pair <pop_t, string> > pop_labels;
    status = cell::read_population_labels(*comm_ptr, string(file_name), pop_labels);
    assert (status >= 0);
    
    // Determine index of population to be read
    size_t pop_idx=0; bool pop_idx_set=false;
    for (size_t i=0; i<pop_labels.size(); i++)
      {
        if (get<1>(pop_labels[i]) == pop_name)
          {
            pop_idx = get<0>(pop_labels[i]);
            pop_idx_set = true;
          }
      }
    if (!pop_idx_set)
      {
        throw_err("Population not found");
      }

    size_t n_nodes;
    map<CELL_IDX_T, pair<uint32_t,pop_t> > pop_ranges;
    vector<pop_range_t> pop_vector;
    assert(cell::read_population_ranges(*comm_ptr,
                                        string(file_name),
                                        pop_ranges, pop_vector,
                                        n_nodes) >= 0);
    
    vector<CELL_IDX_T> tree_index;
    assert(cell::read_cell_index(*comm_ptr,
                                 string(file_name),
                                 get<1>(pop_labels[pop_idx]),
                                 hdf5::TREES,
                                 tree_index) >= 0);

    for (size_t i=0; i<tree_index.size(); i++)
      {
        tree_index[i] += pop_vector[pop_idx].start;
      }

    /* Create a new generator state and initialize it */
    PyNeuroH5TreeGenState *py_ntrg = (PyNeuroH5TreeGenState *)type->tp_alloc(type, 0);
    if (!py_ntrg) return NULL;
    py_ntrg->state = new NeuroH5TreeGenState();

    map<CELL_IDX_T, rank_t> node_rank_map;
    // Create C++ map for node_rank_map:
    // round-robin node to rank assignment from file
    rank_t r=0; size_t count=0;
    for (size_t i = 0; i < tree_index.size(); i++)
      {
        if ((unsigned int)rank == r) count++;
        py_ntrg->state->node_rank_map.insert(make_pair(tree_index[i], r++));
        if ((unsigned int)size <= r) r=0;
      }

    size_t m = tree_index.size() % size;
    if (m == 0)
      {
        py_ntrg->state->pos = seq_last;
      }
    else
      {
        if ((unsigned int)rank < m)
          {
            py_ntrg->state->pos = seq_last;
          }
        else
          {
            py_ntrg->state->pos = seq_next;
          }
      }

    py_ntrg->state->count       = count;
    py_ntrg->state->seq_index   = 0;
    py_ntrg->state->cache_index = 0;
    py_ntrg->state->comm_ptr   = comm_ptr;
    py_ntrg->state->file_name  = string(file_name);
    py_ntrg->state->pop_name   = string(pop_name);
    py_ntrg->state->pop_idx    = pop_idx;
    py_ntrg->state->pop_vector = pop_vector;
    py_ntrg->state->io_size    = io_size;
    py_ntrg->state->comm_size  = size;
    py_ntrg->state->cache_size = cache_size;
    py_ntrg->state->attr_namespaces  = attr_namespaces;

    map<CELL_IDX_T, neurotree_t> tree_map;
    py_ntrg->state->tree_map  = tree_map;
    py_ntrg->state->it_tree  = py_ntrg->state->tree_map.cbegin();

    return (PyObject *)py_ntrg;
  }


  static PyObject *
  neuroh5_cell_attr_gen_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
  {
    int status;
    PyObject *py_comm = NULL;
    MPI_Comm *comm_ptr  = NULL;
    unsigned long io_size=1, cache_size=100;
    const string default_namespace = "Attributes";
    char *file_name, *pop_name, *attr_namespace = (char *)default_namespace.c_str();

    static const char *kwlist[] = {"comm",
                                   "file_name",
                                   "pop_name",
                                   "io_size",
                                   "namespace",
                                   "cache_size",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oss|ksi", (char **)kwlist,
                                     &py_comm, &file_name, &pop_name, 
                                     &io_size, &attr_namespace, &cache_size))
      return NULL;

    assert(py_comm != NULL);
    comm_ptr = PyMPIComm_Get(py_comm);
    assert(comm_ptr != NULL);
    assert(*comm_ptr != MPI_COMM_NULL);

    int rank, size;
    assert(MPI_Comm_size(*comm_ptr, &size) >= 0);
    assert(MPI_Comm_rank(*comm_ptr, &rank) >= 0);

    assert(size > 0);
    
    if (io_size > (unsigned int)size)
      io_size = size;

    if ((size > 0) && (cache_size < (unsigned int)size))
      cache_size = size;

    vector<pair <pop_t, string> > pop_labels;
    status = cell::read_population_labels(*comm_ptr, string(file_name), pop_labels);
    assert (status >= 0);

    // Determine index of population to be read
    size_t pop_idx=0; bool pop_idx_set=false;
    for (size_t i=0; i<pop_labels.size(); i++)
      {
        if (get<1>(pop_labels[i]) == pop_name)
          {
            pop_idx = get<0>(pop_labels[i]);
            pop_idx_set = true;
          }
      }
    if (!pop_idx_set)
      {
        throw_err("Population not found");
      }

    size_t n_nodes;
    map<CELL_IDX_T, pair<uint32_t,pop_t> > pop_ranges;
    vector<pop_range_t> pop_vector;
    assert(cell::read_population_ranges(*comm_ptr,
                                        string(file_name),
                                        pop_ranges, pop_vector,
                                        n_nodes) >= 0);

    vector< pair<string,hid_t> > attr_info;
    assert(cell::get_cell_attributes (string(file_name), string(attr_namespace),
                                      get<1>(pop_labels[pop_idx]), attr_info) >= 0);
    
    vector<CELL_IDX_T> cell_index;
    assert(cell::read_cell_index(*comm_ptr,
                                 string(file_name),
                                 get<1>(pop_labels[pop_idx]),
                                 string(attr_namespace) + "/" + attr_info[0].first,
                                 cell_index) >= 0);

    for (size_t i=0; i<cell_index.size(); i++)
      {
        cell_index[i] += pop_vector[pop_idx].start;
      }
    
    /* Create a new generator state and initialize its state - pointing to the last
     * index in the sequence.
    */
    PyNeuroH5CellAttrGenState *py_ntrg = (PyNeuroH5CellAttrGenState *)type->tp_alloc(type, 0);
    if (!py_ntrg) return NULL;

    py_ntrg->state = new NeuroH5CellAttrGenState();

    // Create C++ map for node_rank_map:
    // round-robin node to rank assignment from file
    rank_t r=0; size_t count=0;
    for (size_t i = 0; i < cell_index.size(); i++)
      {
        if ((unsigned int)rank == r) count++;
        py_ntrg->state->node_rank_map.insert(make_pair(cell_index[i], r++));
        if ((unsigned int)size <= r) r=0;
      }

    size_t m = cell_index.size() % size;
    if (m == 0)
      {
        py_ntrg->state->pos = seq_last;
      }
    else
      {
        if ((unsigned int)rank < m)
          {
            py_ntrg->state->pos = seq_last;
          }
        else
          {
            py_ntrg->state->pos = seq_next;
          }
      }

    py_ntrg->state->count       = count;
    py_ntrg->state->seq_index   = 0;
    py_ntrg->state->cache_index = 0;
    py_ntrg->state->comm_ptr   = comm_ptr;
    py_ntrg->state->file_name  = string(file_name);
    py_ntrg->state->pop_name   = string(pop_name);
    py_ntrg->state->pop_idx    = pop_idx;
    py_ntrg->state->pop_vector = pop_vector;
    py_ntrg->state->io_size    = io_size;
    py_ntrg->state->comm_size  = size;
    py_ntrg->state->cache_size = cache_size;
    py_ntrg->state->attr_namespace  = string(attr_namespace);

    NamedAttrMap attr_map;
    py_ntrg->state->attr_map  = attr_map;
    py_ntrg->state->it_idx = py_ntrg->state->attr_map.index_set.cbegin();

    return (PyObject *)py_ntrg;
  }

  static void
  neuroh5_tree_gen_dealloc(PyNeuroH5TreeGenState *py_ntrg)
  {
    delete py_ntrg->state;
    Py_TYPE(py_ntrg)->tp_free(py_ntrg);
  }

  static void
  neuroh5_cell_attr_gen_dealloc(PyNeuroH5CellAttrGenState *py_ntrg)
  {
    delete py_ntrg->state;
    Py_TYPE(py_ntrg)->tp_free(py_ntrg);
  }

  static void
  neuroh5_prj_gen_dealloc(PyNeuroH5ProjectionGenState *py_ngg)
  {
    delete py_ngg->state;
    Py_TYPE(py_ngg)->tp_free(py_ngg);
  }

  static PyObject *
  neuroh5_tree_gen_next(PyNeuroH5TreeGenState *py_ntrg)
  {
    int size, rank;
    assert(MPI_Comm_size(*py_ntrg->state->comm_ptr, &size) == MPI_SUCCESS);
    assert(MPI_Comm_rank(*py_ntrg->state->comm_ptr, &rank) == MPI_SUCCESS);

    /* 
     * Returning NULL in this case is enough. The next() builtin will raise the
     * StopIteration error for us.
    */
    if (py_ntrg->state->pos != seq_done)
      {
        
        // If the end of the current cache block has been reached,
        // and the iterator has not exceed its locally assigned elements,
        // read the next block
        if ((py_ntrg->state->it_tree == py_ntrg->state->tree_map.cend()) &&
            (py_ntrg->state->seq_index < py_ntrg->state->count))
          {
            int status;
            py_ntrg->state->tree_map.clear();
            py_ntrg->state->attr_maps.clear();
            status = cell::scatter_read_trees (*py_ntrg->state->comm_ptr, py_ntrg->state->file_name,
                                               py_ntrg->state->io_size, py_ntrg->state->attr_namespaces,
                                               py_ntrg->state->node_rank_map, py_ntrg->state->pop_name,
                                               py_ntrg->state->pop_vector[py_ntrg->state->pop_idx].start,
                                               py_ntrg->state->tree_map, py_ntrg->state->attr_maps,
                                               py_ntrg->state->cache_index, py_ntrg->state->cache_size);
            assert (status >= 0);

            py_ntrg->state->cache_index += py_ntrg->state->io_size * py_ntrg->state->cache_size;
            py_ntrg->state->it_tree = py_ntrg->state->tree_map.cbegin();
          }

        if (py_ntrg->state->it_tree != py_ntrg->state->tree_map.cend())
          {
            CELL_IDX_T key = py_ntrg->state->it_tree->first;
            const neurotree_t &tree = py_ntrg->state->it_tree->second;
            PyObject *elem = py_build_tree_value(key, tree, py_ntrg->state->attr_maps);
            assert(elem != NULL);
        
            /* Exceptions from PySequence_GetItem are propagated to the caller
             * (elem will be NULL so we also return NULL).
             */
            PyObject *result = Py_BuildValue("lO", key, elem);
            py_ntrg->state->it_tree++;
            py_ntrg->state->seq_index++;
            return result;
          }
        else
          {
            PyObject *result = NULL;
            switch (py_ntrg->state->pos)
              {
              case seq_next:
                {
                  py_ntrg->state->pos = seq_last;
                  result = PyTuple_Pack(2, (Py_INCREF(Py_None), Py_None),
                                        (Py_INCREF(Py_None), Py_None));
                  break;
                }
              case seq_last:
                {
                  py_ntrg->state->pos = seq_done;
                  result = NULL;
                  break;
                }
              case seq_done:
                {
                  result = NULL;
                  break;
                }
              case seq_empty:
                {
                  result = NULL;
                  break;
                }
              }
            return result;
          }
      }

    return NULL;
  }

  
  static PyObject *
  neuroh5_cell_attr_gen_next(PyNeuroH5CellAttrGenState *py_ntrg)
  {
    int size, rank;
    assert(MPI_Comm_size(*py_ntrg->state->comm_ptr, &size) == MPI_SUCCESS);
    assert(MPI_Comm_rank(*py_ntrg->state->comm_ptr, &rank) == MPI_SUCCESS);

    if (py_ntrg->state->pos != seq_done)
      {
        /* seq_index = count-1 means that the generator is exhausted.
         * Returning NULL in this case is enough. The next() builtin will raise the
         * StopIteration error for us.
         */
        if ((py_ntrg->state->it_idx == py_ntrg->state->attr_map.index_set.cend()) &&
            (py_ntrg->state->seq_index < py_ntrg->state->count))
          {
            // If the end of the current cache block has been reached,
            // read the next block
            py_ntrg->state->attr_map.clear();
            
            int status;
            status = cell::scatter_read_cell_attributes (*py_ntrg->state->comm_ptr, py_ntrg->state->file_name,
                                                         py_ntrg->state->io_size, py_ntrg->state->attr_namespace,
                                                         py_ntrg->state->node_rank_map, py_ntrg->state->pop_name,
                                                         py_ntrg->state->pop_vector[py_ntrg->state->pop_idx].start,
                                                         py_ntrg->state->attr_map,
                                                         py_ntrg->state->cache_index, py_ntrg->state->cache_size);
            assert (status >= 0);
            py_ntrg->state->attr_map.attr_names(py_ntrg->state->attr_names);
            py_ntrg->state->cache_index += py_ntrg->state->io_size * py_ntrg->state->cache_size;
            py_ntrg->state->it_idx = py_ntrg->state->attr_map.index_set.cbegin();
          }

        PyObject *result = NULL;

        if (py_ntrg->state->it_idx != py_ntrg->state->attr_map.index_set.cend())
          {
            const CELL_IDX_T key = *(py_ntrg->state->it_idx);
            PyObject *elem = py_build_cell_attr_value(key, py_ntrg->state->attr_map,
                                                      py_ntrg->state->attr_namespace,
                                                      py_ntrg->state->attr_names);
            assert(elem != NULL);
            py_ntrg->state->it_idx++;
            py_ntrg->state->seq_index++;
            result = Py_BuildValue("lO", key, elem);
          }
        else
          {
            switch (py_ntrg->state->pos)
              {
              case seq_next:
                {
                  py_ntrg->state->pos = seq_last;
                  result = PyTuple_Pack(2, (Py_INCREF(Py_None), Py_None),
                                        (Py_INCREF(Py_None), Py_None));
                  break;
                }
              case seq_last:
                {
                  py_ntrg->state->pos = seq_done;
                  result = NULL;
                  break;
                }
              case seq_done:
                {
                  result = NULL;
                  break;
                }
              case seq_empty:
                {
                  result = NULL;
                  break;
                }
              }
          }
        
        /* Exceptions from PySequence_GetItem are propagated to the caller
         * (elem will be NULL so we also return NULL).
        */
        return result;
      }
    
    return NULL;
  }
  
  static int neuroh5_prj_gen_next_block(PyNeuroH5ProjectionGenState *py_ngg)
  {
    int status = 0;
    int size, rank;
    assert(MPI_Comm_size(*py_ngg->state->comm_ptr, &size) == MPI_SUCCESS);
    assert(MPI_Comm_rank(*py_ngg->state->comm_ptr, &rank) == MPI_SUCCESS);
    
    // If the end of the current edge map has been reached,
    // read the next block
    py_ngg->state->edge_map.clear();

    vector < vector < vector<string> > > edge_attr_name_vector;
    vector <edge_map_t> prj_vector;
    size_t max_local_num_edges=0;
    
    status = graph::scatter_projection(*py_ngg->state->comm_ptr,
                                       py_ngg->state->io_size, py_ngg->state->edge_map_type, 
                                       py_ngg->state->file_name,
                                       py_ngg->state->src_pop_name,
                                       py_ngg->state->dst_pop_name,
                                       py_ngg->state->attr_namespaces, py_ngg->state->node_rank_map,
                                       py_ngg->state->pop_vector, py_ngg->state->pop_ranges,
                                       py_ngg->state->pop_labels, py_ngg->state->pop_pairs,
                                       py_ngg->state->edge_attr_info, py_ngg->state->edge_attr_num,
                                       prj_vector, edge_attr_name_vector,
                                       py_ngg->state->local_num_nodes,
                                       py_ngg->state->local_num_edges,
                                       py_ngg->state->total_num_edges,
                                       py_ngg->state->block_index,
                                       py_ngg->state->cache_size);
    assert (status >= 0);
    assert(prj_vector.size() > 0);
    if (edge_attr_name_vector.size() > 0)
      {
        py_ngg->state->edge_attr_names = edge_attr_name_vector[0];
      }
    
    py_ngg->state->edge_map = prj_vector[0];
    //assert(py_ngg->state->edge_map.size() > 0);
    py_ngg->state->edge_map_iter = py_ngg->state->edge_map.cbegin();
    py_ngg->state->edge_iter = get<0>(py_ngg->state->edge_map_iter->second).cbegin();
    
    py_ngg->state->block_index += py_ngg->state->io_size * py_ngg->state->cache_size;
    status = MPI_Allreduce(&(py_ngg->state->local_num_edges), &max_local_num_edges, 1,
                           MPI_SIZE_T, MPI_MAX, *py_ngg->state->comm_ptr);
    assert(status == MPI_SUCCESS);
    py_ngg->state->edge_count += max_local_num_edges;

    return status;
  }

  
  static PyObject *
  neuroh5_prj_gen_next(PyNeuroH5ProjectionGenState *py_ngg)
  {
    PyObject *result = NULL; 
    int status = 0;
    int size, rank;
    assert(MPI_Comm_size(*py_ngg->state->comm_ptr, &size) == MPI_SUCCESS);
    assert(MPI_Comm_rank(*py_ngg->state->comm_ptr, &rank) == MPI_SUCCESS);

    switch (py_ngg->state->pos)
      {
      case seq_next:
        {
          if ((py_ngg->state->edge_map_iter == py_ngg->state->edge_map.cend()) &&
              (py_ngg->state->block_index < py_ngg->state->block_count))
            {
              neuroh5_prj_gen_next_block(py_ngg);
            }
          
          if (py_ngg->state->edge_map_iter == py_ngg->state->edge_map.cend())
            {
              if (py_ngg->state->edge_index == py_ngg->state->edge_count)
                {
                  py_ngg->state->pos = seq_last;
                }
              else
                {
                  py_ngg->state->edge_index++;
                }
              result = PyTuple_Pack(3,
                                    (Py_INCREF(Py_None), Py_None),
                                    (Py_INCREF(Py_None), Py_None),
                                    (Py_INCREF(Py_None), Py_None));
              break;
            }

          const vector<NODE_IDX_T>& adj_vector = get<0>(py_ngg->state->edge_map_iter->second);
          if (py_ngg->state->edge_iter != adj_vector.cend())
            {
              const NODE_IDX_T key = py_ngg->state->edge_map_iter->first;
              result = py_build_edge_value(key, *(py_ngg->state->edge_iter),
                                           distance(py_ngg->state->edge_iter, adj_vector.cbegin()),
                                           get<1>(py_ngg->state->edge_map_iter->second),
                                           py_ngg->state->edge_attr_names);
              py_ngg->state->edge_iter = next(py_ngg->state->edge_iter);
            }
          else
            {
              py_ngg->state->edge_map_iter = next(py_ngg->state->edge_map_iter);
              if ((py_ngg->state->edge_map_iter == py_ngg->state->edge_map.cend()) &&
                  (py_ngg->state->block_index < py_ngg->state->block_count))
                {
                  neuroh5_prj_gen_next_block(py_ngg);
                }
              if (py_ngg->state->edge_map_iter != py_ngg->state->edge_map.cend())
                {
                  const vector<NODE_IDX_T>& adj_vector1 = get<0>(py_ngg->state->edge_map_iter->second);
                  py_ngg->state->edge_iter = adj_vector1.cbegin();
                  assert(py_ngg->state->edge_iter != adj_vector1.cend());
                  const NODE_IDX_T key = py_ngg->state->edge_map_iter->first;
                  const NODE_IDX_T adj = *(py_ngg->state->edge_iter);
                  result = py_build_edge_value(key, adj,
                                               distance(py_ngg->state->edge_iter, adj_vector1.cbegin()),
                                               get<1>(py_ngg->state->edge_map_iter->second),
                                               py_ngg->state->edge_attr_names);
                  py_ngg->state->edge_iter = next(py_ngg->state->edge_iter);
                 }
              else
                {
                  if (py_ngg->state->edge_index == py_ngg->state->edge_count)
                    {
                      py_ngg->state->pos = seq_last;
                    }
                  else
                    if (py_ngg->state->block_index >= py_ngg->state->block_count)
                      {
                        py_ngg->state->pos = seq_empty;
                      }
                  result = PyTuple_Pack(3,
                                        (Py_INCREF(Py_None), Py_None),
                                        (Py_INCREF(Py_None), Py_None),
                                        (Py_INCREF(Py_None), Py_None));
                }
            }
          py_ngg->state->edge_index++;

          break;
        }
      case seq_empty:
        {
          if (py_ngg->state->edge_index == py_ngg->state->edge_count)
            {
              py_ngg->state->pos = seq_last;
            }
          else
            {
              py_ngg->state->edge_index++;
            }
          result = PyTuple_Pack(3,
                                (Py_INCREF(Py_None), Py_None),
                                (Py_INCREF(Py_None), Py_None),
                                (Py_INCREF(Py_None), Py_None));
          break;
        }
      case seq_last:
        {
          py_ngg->state->pos = seq_done;
          result = NULL;
          break;
        }
      case seq_done:
        {
          result = NULL;
          break;
        }
      }
      

    /* Exceptions from PySequence_GetItem are propagated to the caller
     * (elem will be NULL so we also return NULL).
     */
    
    return result;
  }

  
  // NeuroH5 read iterator
  PyTypeObject PyNeuroH5TreeGen_Type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    "NeuroH5TreeGen",                 /* tp_name */
    sizeof(PyNeuroH5TreeGenState),      /* tp_basicsize */
    0,                              /* tp_itemsize */
    (destructor)neuroh5_tree_gen_dealloc, /* tp_dealloc */
    0,                              /* tp_print */
    0,                              /* tp_getattr */
    0,                              /* tp_setattr */
    0,                              /* tp_reserved */
    0,                              /* tp_repr */
    0,                              /* tp_as_number */
    0,                              /* tp_as_sequence */
    0,                              /* tp_as_mapping */
    0,                              /* tp_hash */
    0,                              /* tp_call */
    0,                              /* tp_str */
    0,                              /* tp_getattro */
    0,                              /* tp_setattro */
    0,                              /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,             /* tp_flags */
    0,                              /* tp_doc */
    0,                              /* tp_traverse */
    0,                              /* tp_clear */
    0,                              /* tp_richcompare */
    0,                              /* tp_weaklistoffset */
    PyObject_SelfIter,              /* tp_iter */
    (iternextfunc)neuroh5_tree_gen_next, /* tp_iternext */
    0,                              /* tp_methods */
    0,                              /* tp_members */
    0,                              /* tp_getset */
    0,                              /* tp_base */
    0,                              /* tp_dict */
    0,                              /* tp_descr_get */
    0,                              /* tp_descr_set */
    0,                              /* tp_dictoffset */
    0,                              /* tp_init */
    PyType_GenericAlloc,            /* tp_alloc */
    neuroh5_tree_gen_new,           /* tp_new */
  };

  
  // NeuroH5 attribute read iterator
  PyTypeObject PyNeuroH5CellAttrGen_Type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    "NeuroH5CellAttrGen",                 /* tp_name */
    sizeof(PyNeuroH5TreeGenState),      /* tp_basicsize */
    0,                              /* tp_itemsize */
    (destructor)neuroh5_cell_attr_gen_dealloc, /* tp_dealloc */
    0,                              /* tp_print */
    0,                              /* tp_getattr */
    0,                              /* tp_setattr */
    0,                              /* tp_reserved */
    0,                              /* tp_repr */
    0,                              /* tp_as_number */
    0,                              /* tp_as_sequence */
    0,                              /* tp_as_mapping */
    0,                              /* tp_hash */
    0,                              /* tp_call */
    0,                              /* tp_str */
    0,                              /* tp_getattro */
    0,                              /* tp_setattro */
    0,                              /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,             /* tp_flags */
    0,                              /* tp_doc */
    0,                              /* tp_traverse */
    0,                              /* tp_clear */
    0,                              /* tp_richcompare */
    0,                              /* tp_weaklistoffset */
    PyObject_SelfIter,              /* tp_iter */
    (iternextfunc)neuroh5_cell_attr_gen_next, /* tp_iternext */
    0,                              /* tp_methods */
    0,                              /* tp_members */
    0,                              /* tp_getset */
    0,                              /* tp_base */
    0,                              /* tp_dict */
    0,                              /* tp_descr_get */
    0,                              /* tp_descr_set */
    0,                              /* tp_dictoffset */
    0,                              /* tp_init */
    PyType_GenericAlloc,            /* tp_alloc */
    neuroh5_cell_attr_gen_new,      /* tp_new */
  };

  // NeuroH5 graph read iterator
  PyTypeObject PyNeuroH5ProjectionGen_Type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    "NeuroH5ProjectionGen",                 /* tp_name */
    sizeof(PyNeuroH5ProjectionGenState),      /* tp_basicsize */
    0,                              /* tp_itemsize */
    (destructor)neuroh5_prj_gen_dealloc, /* tp_dealloc */
    0,                              /* tp_print */
    0,                              /* tp_getattr */
    0,                              /* tp_setattr */
    0,                              /* tp_reserved */
    0,                              /* tp_repr */
    0,                              /* tp_as_number */
    0,                              /* tp_as_sequence */
    0,                              /* tp_as_mapping */
    0,                              /* tp_hash */
    0,                              /* tp_call */
    0,                              /* tp_str */
    0,                              /* tp_getattro */
    0,                              /* tp_setattro */
    0,                              /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,             /* tp_flags */
    0,                              /* tp_doc */
    0,                              /* tp_traverse */
    0,                              /* tp_clear */
    0,                              /* tp_richcompare */
    0,                              /* tp_weaklistoffset */
    PyObject_SelfIter,              /* tp_iter */
    (iternextfunc)neuroh5_prj_gen_next, /* tp_iternext */
    0,                              /* tp_methods */
    0,                              /* tp_members */
    0,                              /* tp_getset */
    0,                              /* tp_base */
    0,                              /* tp_dict */
    0,                              /* tp_descr_get */
    0,                              /* tp_descr_set */
    0,                              /* tp_dictoffset */
    0,                              /* tp_init */
    PyType_GenericAlloc,            /* tp_alloc */
    neuroh5_prj_gen_new,           /* tp_new */
  };

  
  static PyMethodDef module_methods[] = {
    { "read_population_ranges", (PyCFunction)py_read_population_ranges, METH_VARARGS,
      "Returns population size and ranges." },
    { "read_population_names", (PyCFunction)py_read_population_names, METH_VARARGS,
      "Returns the names of the populations contained in the given file." },
    { "read_projection_names", (PyCFunction)py_read_projection_names, METH_VARARGS,
      "Returns the names of the projections contained in the given file." },
    { "read_trees", (PyCFunction)py_read_trees, METH_VARARGS,
      "Reads neuronal tree morphology." },
    { "read_tree_selection", (PyCFunction)py_read_tree_selection, METH_VARARGS,
      "Reads the selected neuronal tree morphologies." },
    { "scatter_read_trees", (PyCFunction)py_scatter_read_trees, METH_VARARGS | METH_KEYWORDS,
      "Reads neuronal tree morphology using scalable parallel read/scatter." },
    { "read_cell_attribute_selection", (PyCFunction)py_read_cell_attribute_selection, METH_VARARGS | METH_KEYWORDS,
      "Reads attributes for a selection of cells." },
    { "read_cell_attributes", (PyCFunction)py_read_cell_attributes, METH_VARARGS | METH_KEYWORDS,
      "Reads additional attributes for the given range of cells." },
    { "bcast_cell_attributes", (PyCFunction)py_bcast_cell_attributes, METH_VARARGS | METH_KEYWORDS,
      "Reads attributes for the given range of cells and broadcasts to all ranks." },
    { "write_cell_attributes", (PyCFunction)py_write_cell_attributes, METH_VARARGS | METH_KEYWORDS,
      "Writes attributes for the given range of cells." },
    { "append_cell_attributes", (PyCFunction)py_append_cell_attributes, METH_VARARGS | METH_KEYWORDS,
      "Appends additional attributes for the given range of cells." },
    { "append_cell_trees", (PyCFunction)py_append_cell_trees, METH_VARARGS | METH_KEYWORDS,
      "Appends tree morphologies." },
    { "read_graph", (PyCFunction)py_read_graph, METH_VARARGS,
      "Reads graph connectivity in Destination Block Sparse format." },
    { "scatter_read_graph", (PyCFunction)py_scatter_read_graph, METH_VARARGS | METH_KEYWORDS,
      "Reads and scatters graph connectivity in Destination Block Sparse format." },
    { "bcast_graph", (PyCFunction)py_bcast_graph, METH_VARARGS | METH_KEYWORDS,
      "Reads and broadcasts graph connectivity in Destination Block Sparse format." },
    { "read_graph_serial", (PyCFunction)py_read_graph_serial, METH_VARARGS,
      "Reads graph connectivity in Destination Block Sparse format." },
    { "write_graph", (PyCFunction)py_write_graph, METH_VARARGS,
      "Writes graph connectivity in Destination Block Sparse format." },
    { "append_graph", (PyCFunction)py_append_graph, METH_VARARGS,
      "Appends graph connectivity in Destination Block Sparse format." },
    { NULL, NULL, 0, NULL }
  };
}

#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "io",
        NULL,
        NULL,
        module_methods,
        NULL,
        NULL,
        NULL,
        NULL
};

#endif

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC
PyInit_io(void)
#else
PyMODINIT_FUNC
initio(void)
#endif
{
  assert(import_mpi4py() >= 0);
  import_array();
  
#if PY_MAJOR_VERSION >= 3
  PyObject *module = PyModule_Create(&moduledef);
#else
  PyObject *module = Py_InitModule3("io", module_methods, "NeuroH5 I/O module");
#endif
  
  if (PyType_Ready(&PyNeuroH5TreeGen_Type) < 0)
    {
      printf("NeuroH5TreeGen type cannot be added\n");
#if PY_MAJOR_VERSION >= 3
      return NULL;
#else      
      return;
#endif
    }

  Py_INCREF((PyObject *)&PyNeuroH5TreeGen_Type);
  PyModule_AddObject(module, "NeuroH5TreeGen", (PyObject *)&PyNeuroH5TreeGen_Type);

  if (PyType_Ready(&PyNeuroH5CellAttrGen_Type) < 0)
    {
      printf("NeuroH5CellAttrGen type cannot be added\n");
#if PY_MAJOR_VERSION >= 3
      return NULL;
#else      
      return;
#endif
    }

  Py_INCREF((PyObject *)&PyNeuroH5CellAttrGen_Type);
  PyModule_AddObject(module, "NeuroH5CellAttrGen", (PyObject *)&PyNeuroH5CellAttrGen_Type);

  if (PyType_Ready(&PyNeuroH5ProjectionGen_Type) < 0)
    {
      printf("NeuroH5ProjectionGen type cannot be added\n");
#if PY_MAJOR_VERSION >= 3
      return NULL;
#else      
      return;
#endif
    }

  Py_INCREF((PyObject *)&PyNeuroH5ProjectionGen_Type);
  PyModule_AddObject(module, "NeuroH5ProjectionGen", (PyObject *)&PyNeuroH5ProjectionGen_Type);
  
#if PY_MAJOR_VERSION >= 3
  return module;
#else
  return;
#endif
}

  

