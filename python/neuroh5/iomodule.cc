// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file iomodule.cc
///
///  Python module for reading and writing neuronal connectivity and morphological information.
///
///  Copyright (C) 2016-2018 Project NeuroH5.
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
#include <iterator>

#include "throw_assert.hh"

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
#include "mpe_seq.hh"
#include "read_projection.hh"
#include "read_graph.hh"
#include "read_graph_selection.hh"
#include "scatter_read_graph.hh"
#include "scatter_read_projection.hh"
#include "bcast_graph.hh"
#include "write_graph.hh"
#include "append_graph.hh"
#include "projection_names.hh"
#include "edge_attributes.hh"
#include "serialize_data.hh"

#if PY_MAJOR_VERSION >= 3
#define Py_TPFLAGS_HAVE_ITER ((Py_ssize_t)0)
#endif

using namespace std;
using namespace neuroh5;
using namespace neuroh5::data;

struct Intern {

  map<string, char *> string_map;
  
  char *add(const std::string &s)
  {
    char *str;
    auto sm_it = string_map.find(s);
    if (sm_it != string_map.end())
      {
        str = sm_it->second;
      }
    else
      {
        size_t n = s.size();
        str = (char *)malloc(n+1);
        throw_assert_nomsg(str != NULL);
        s.copy(str, n);
        str[n] = 0;
        string_map.insert(make_pair(s, str));
      }
    return str;
  }
};

Intern attr_name_intern;

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

PyObject* PyStr_FromCString(const char *string)
{
#if PY_MAJOR_VERSION >= 3
  return PyUnicode_FromString(string);
#else
  return PyBytes_FromString(string);
#endif
  
}

const char* PyStr_ToCString(PyObject *string)
{
#if PY_MAJOR_VERSION >= 3
  return PyUnicode_AsUTF8AndSize(string, NULL);
#else
  return PyBytes_AsString(string);
#endif
  
}

bool PyStr_Check(PyObject *string)
{
#if PY_MAJOR_VERSION >= 3
  return PyUnicode_Check(string);
#else
  return PyBytes_Check(string);
#endif
  
}

template<class T>
void py_array_to_vector (PyObject *pyval,
                         vector<T>& value_vector)
{
  npy_intp *dims, ind = 0;
  throw_assert(PyArray_Check(pyval), "py_array_to_vector: argument is not an array");
  PyArrayObject* pyarr = (PyArrayObject*)PyArray_FROM_OTF(pyval, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
  T *pyarr_ptr = (T *)PyArray_GetPtr(pyarr, &ind);
  dims = PyArray_DIMS(pyarr);
  throw_assert(dims != NULL, "py_array_to_vector: argument has no dimensions");
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
                       map<CELL_IDX_T, vector<T> >& all_attr_values)
{
  npy_intp *dims, ind = 0;
  throw_assert(PyArray_Check(pyval), "append_value_map: argument is not an array");
  PyArrayObject* pyarr = (PyArrayObject*)PyArray_FROM_OTF(pyval, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
  dims = PyArray_DIMS(pyarr);
  if (dims != NULL)
    {
      size_t value_size = dims[0];
      T *pyarr_ptr = (T *)PyArray_GetPtr(pyarr, &ind);
      vector<T> attr_values(value_size);
      for (size_t j=0; j<value_size; j++)
        {
          attr_values[j] = pyarr_ptr[j];
        }
      all_attr_values.insert(make_pair(idx, attr_values));
    }
  Py_DECREF(pyarr);
}


void build_cell_attr_value_maps (PyObject *idx_values,
                                 map<string, map<CELL_IDX_T, vector<uint32_t>>>& all_attr_values_uint32,
                                 map<string, map<CELL_IDX_T, vector<uint16_t>>>& all_attr_values_uint16,
                                 map<string, map<CELL_IDX_T, vector<uint8_t>>>& all_attr_values_uint8,
                                 map<string, map<CELL_IDX_T, vector<int32_t>>>& all_attr_values_int32,
                                 map<string, map<CELL_IDX_T, vector<int16_t>>>& all_attr_values_int16,
                                 map<string, map<CELL_IDX_T, vector<int8_t>>>& all_attr_values_int8,
                                 map<string, map<CELL_IDX_T, vector<float>>>& all_attr_values_float)
{
  PyObject *idx_key, *idx_value;
  Py_ssize_t idx_pos = 0;
  int npy_type=0;
                           
  throw_assert(PyDict_Check(idx_values),
               "build_cell_attr_value_maps: invalid idx_values dictionary");

  while (PyDict_Next(idx_values, &idx_pos, &idx_key, &idx_value))
    {
      throw_assert((idx_key != Py_None) && (idx_value != Py_None),
                   "build_cell_attr_value_maps: invalid idx_values dictionary");

      CELL_IDX_T idx = PyLong_AsLong(idx_key);

      PyObject *attr_key, *attr_values;
      Py_ssize_t attr_pos = 0;
                        
      while (PyDict_Next(idx_value, &attr_pos, &attr_key, &attr_values))
        {
          if (!PyArray_Check(attr_values))
            {
              continue;
            }

          npy_type = PyArray_TYPE((PyArrayObject *)attr_values);
          string attr_name = string(PyStr_ToCString(attr_key));
                                     
          switch (npy_type)
            {
            case NPY_UINT32:
              {
                append_value_map<uint32_t> (idx, attr_values,
                                            all_attr_values_uint32[attr_name]);
                break;
              }
            case NPY_INT32:
              {
                append_value_map<int32_t> (idx, attr_values, 
                                           all_attr_values_int32[attr_name]);
                break;
              }
            case NPY_UINT16:
              {
                append_value_map<uint16_t> (idx, attr_values, 
                                            all_attr_values_uint16[attr_name]);
                break;
              }
            case NPY_INT16:
              {
                append_value_map<int16_t> (idx, attr_values, 
                                           all_attr_values_int16[attr_name]);
                break;
              }
            case NPY_UINT8:
              {
                append_value_map<uint8_t> (idx, attr_values, 
                                           all_attr_values_uint8[attr_name]);
                break;
              }
            case NPY_INT8:
              {
                append_value_map<int8_t> (idx, attr_values, 
                                          all_attr_values_int8[attr_name]);
                break;
              }
            case NPY_FLOAT:
              {
                append_value_map<float> (idx, attr_values, 
                                         all_attr_values_float[attr_name]);
                break;
              }
            default:
              throw runtime_error("Unsupported attribute type");
              break;
            }
        }
    }
}


PyObject *py_build_edge_attribute_info (const vector< pair<string,string> >& prj_names,
                                        const vector <string>& edge_attr_name_spaces,
                                        const vector < map <string, vector < vector <string> > > >& edge_attr_name_vector)
{
    PyObject *py_attribute_info = PyDict_New();
    for (size_t p = 0; p<edge_attr_name_vector.size(); p++)
      {
        PyObject *py_prj_attr_info  = PyDict_New();
        for (const string& attr_namespace : edge_attr_name_spaces) 
          {
            PyObject *py_prj_ns_attr_info  = PyDict_New();
            int attr_index=0;
            const vector <vector <string> > ns_edge_attr_names = edge_attr_name_vector[p].at(attr_namespace);
            for (size_t n = 0; n<ns_edge_attr_names.size(); n++)
              {
                for (size_t t = 0; t<ns_edge_attr_names[n].size(); t++)
                  {
                    PyObject *py_attr_index = PyLong_FromLong(attr_index);
                    
                    PyDict_SetItemString(py_prj_ns_attr_info, ns_edge_attr_names[n][t].c_str(), py_attr_index);
                    Py_DECREF(py_attr_index);

                    attr_index++;
                  }
              }
            PyDict_SetItemString(py_prj_attr_info, attr_namespace.c_str(), py_prj_ns_attr_info);
            Py_DECREF(py_prj_ns_attr_info);
          }

        PyObject *py_prj_attr_info_dict = PyDict_GetItemString(py_attribute_info, prj_names[p].second.c_str());
        if (py_prj_attr_info_dict == NULL)
          {
            py_prj_attr_info_dict = PyDict_New();
            PyDict_SetItemString(py_attribute_info, prj_names[p].second.c_str(),
                                 py_prj_attr_info_dict);
            Py_DECREF(py_prj_attr_info_dict);
          }
        PyDict_SetItemString(py_prj_attr_info_dict,
                             prj_names[p].first.c_str(),
                             py_prj_attr_info);
        Py_DECREF(py_prj_attr_info);
      }

    return py_attribute_info;
}


void get_edge_attr_index (PyObject *py_edge_values,
                          map <string, pair <size_t, AttrIndex > >& edge_attr_index)
{
  PyObject *py_edge_key, *py_edge_value;
  Py_ssize_t edge_pos = 0;
  int npy_type=0;
  map <string, AttrSet> attr_set_map;
  
  // Iterate through attributes of the first edge in order to build a map of attribute names to indices  
  while (PyDict_Next(py_edge_values, &edge_pos, &py_edge_key, &py_edge_value))
    {
      throw_assert((py_edge_key != Py_None) && (py_edge_value != Py_None),
                   "build_edge_map: invalid edge map");

      PyObject *py_attr_name_spaces = PyTuple_GetItem(py_edge_value, 1);

      Py_ssize_t attr_namespace_pos = 0;
      PyObject *py_attr_namespace, *py_attr_namespace_value;
      
      while (PyDict_Next(py_attr_name_spaces, &attr_namespace_pos, &py_attr_namespace, &py_attr_namespace_value))
        {
          PyObject *py_attr_key, *py_attr_values;
          Py_ssize_t attr_pos = 0;

          throw_assert(PyStr_Check(py_attr_namespace),
                       "build_edge_map: namespace is not a string");

          const char *str = PyStr_ToCString (py_attr_namespace);
          string attr_namespace = string(str);
          
          while (PyDict_Next(py_attr_namespace_value, &attr_pos, &py_attr_key, &py_attr_values))
            {
              string attr_name = string(PyStr_ToCString(py_attr_key));
              npy_type = PyArray_TYPE((PyArrayObject *)py_attr_values);

              switch (npy_type)
                {
                case NPY_UINT32:
                  {
                    attr_set_map[attr_namespace].add<uint32_t>(attr_name);
                    break;
                  }
                case NPY_UINT16:
                  {
                    attr_set_map[attr_namespace].add<uint16_t>(attr_name);
                    break;
                  }
                case NPY_UINT8:
                  {
                    attr_set_map[attr_namespace].add<uint8_t>(attr_name);
                    break;
                  }
                case NPY_INT32:
                  {
                    attr_set_map[attr_namespace].add<int32_t>(attr_name);
                    break;
                  }
                case NPY_INT16:
                  {
                    attr_set_map[attr_namespace].add<int16_t>(attr_name);
                    break;
                  }
                case NPY_INT8:
                  {
                    attr_set_map[attr_namespace].add<int8_t>(attr_name);
                    break;
                  }
                case NPY_FLOAT:
                  {
                    attr_set_map[attr_namespace].add<float>(attr_name);
                    break;
                  }
                default:
                  throw runtime_error("Unsupported attribute type");
                  break;
                }
            }
          
        }
      
      break;
    }
  
  map <string, AttrSet>::const_iterator ns_first = attr_set_map.cbegin();
  for (auto ns_it=attr_set_map.cbegin(); ns_it != attr_set_map.cend(); ++ns_it)
    {
      const string& attr_namespace = ns_it->first;
      const AttrSet& attr_set = ns_it->second;
      size_t attr_ns_index = distance(ns_first, ns_it);
      AttrIndex attr_index(attr_set);
      edge_attr_index[attr_namespace] = make_pair(attr_ns_index, attr_index);
    }
  
}


void build_edge_map (PyObject *py_edge_values,
                     const map <string, pair <size_t, AttrIndex > >& edge_attr_index,
                     edge_map_t& edge_map)
{
  PyObject *py_edge_key, *py_edge_value;
  Py_ssize_t edge_pos = 0;
  int npy_type=0;

        
  while (PyDict_Next(py_edge_values, &edge_pos, &py_edge_key, &py_edge_value))
    {
      throw_assert((py_edge_key != Py_None) && (py_edge_value != Py_None),
                   "build_edge_map: invalid edge map");


      NODE_IDX_T node_idx = PyLong_AsLong(py_edge_key);

      PyObject *py_attr_name_spaces = PyTuple_GetItem(py_edge_value, 1);
      PyObject *py_adj_values = PyTuple_GetItem(py_edge_value, 0);

      Py_ssize_t attr_namespace_pos = 0;
      PyObject *py_attr_namespace, *py_attr_namespace_value;
      vector <data::AttrVal> edge_attr_vector(edge_attr_index.size());
      vector<NODE_IDX_T>  adj_values;

      npy_type = PyArray_TYPE((PyArrayObject *)py_adj_values);
      size_t num_edges=0;
      
      switch (npy_type)
        {
        case NPY_UINT32:
          {
            py_array_to_vector<NODE_IDX_T> (py_adj_values, adj_values);
            num_edges = adj_values.size();
            break;
          }
        default:
          throw runtime_error("Unsupported source vertex type");
          break;
        }
      
      while (PyDict_Next(py_attr_name_spaces, &attr_namespace_pos, &py_attr_namespace, &py_attr_namespace_value))
        {
          PyObject *py_attr_key, *py_attr_values;
          Py_ssize_t attr_pos = 0;
          size_t attr_idx = 0;

          throw_assert(PyStr_Check(py_attr_namespace),
                       "build_edge_map: namespace is not a string");

          const char *str = PyStr_ToCString (py_attr_namespace);
          string attr_namespace = string(str);
                                   
          data::AttrVal edge_attr_values;

          auto ns_it = edge_attr_index.find(attr_namespace);
          throw_assert(ns_it != edge_attr_index.end(),
                       "build_edge_map: namespace mismatch");

          const size_t ns_index = ns_it->second.first;
          const AttrIndex& attr_index = ns_it->second.second;

          while (PyDict_Next(py_attr_namespace_value, &attr_pos, &py_attr_key, &py_attr_values))
            {
              throw_assert((py_attr_key != Py_None) && (py_attr_values != Py_None),
                           "build_edge_map: invalid attribute dictionary");

              string attr_name = string(PyStr_ToCString(py_attr_key));

              npy_type = PyArray_TYPE((PyArrayObject *)py_attr_values);

              vector<uint32_t>    attr_values_uint32;
              vector<uint16_t>    attr_values_uint16;
              vector<uint8_t>     attr_values_uint8;
              vector<int32_t>     attr_values_int32;
              vector<int16_t>     attr_values_int16;
              vector<int8_t>      attr_values_int8;
              vector<float>       attr_values_float;
                                       
              switch (npy_type)
                {
                case NPY_UINT32:
                  {
                    size_t idx = attr_index.attr_index<uint32_t>(attr_name);
                    py_array_to_vector<uint32_t>(py_attr_values, attr_values_uint32);
                    throw_assert(attr_values_uint32.size() == num_edges,
                                 "build_edge_map: mismatch in number of edges and number of uint32 attributes");
                    edge_attr_values.resize<uint32_t>(attr_index.size_attr_index<uint32_t>());
                    edge_attr_values.insert(attr_values_uint32, idx);
                    break;
                  }
                case NPY_UINT16:
                  {
                    size_t idx = attr_index.attr_index<uint16_t>(attr_name);
                    py_array_to_vector<uint16_t>(py_attr_values, attr_values_uint16);
                    throw_assert(attr_values_uint16.size() == num_edges,
                                 "build_edge_map: mismatch in number of edges and number of uint16 attributes");
                    edge_attr_values.resize<uint16_t>(attr_index.size_attr_index<uint16_t>());
                    edge_attr_values.insert(attr_values_uint16, idx);
                    break;
                  }
                case NPY_UINT8:
                  {
                    size_t idx = attr_index.attr_index<uint8_t>(attr_name);
                    py_array_to_vector<uint8_t>(py_attr_values, attr_values_uint8);
                    throw_assert(attr_values_uint8.size() == num_edges,
                                 "build_edge_map: mismatch in number of edges and number of uint8 attributes");
                    edge_attr_values.resize<uint8_t>(attr_index.size_attr_index<uint8_t>());
                    edge_attr_values.insert(attr_values_uint8, idx);
                    break;
                  }
                case NPY_INT32:
                  {
                    size_t idx = attr_index.attr_index<int32_t>(attr_name);
                    py_array_to_vector<int32_t>(py_attr_values, attr_values_int32);
                    throw_assert(attr_values_int32.size() == num_edges,
                                 "build_edge_map: mismatch in number of edges and number of int32 attributes");
                    edge_attr_values.resize<int32_t>(attr_index.size_attr_index<int32_t>());
                    edge_attr_values.insert(attr_values_int32, idx);
                    break;
                  }
                case NPY_INT16:
                  {
                    size_t idx = attr_index.attr_index<int16_t>(attr_name);
                    py_array_to_vector<int16_t>(py_attr_values, attr_values_int16);
                    throw_assert(attr_values_int16.size() == num_edges,
                                 "build_edge_map: mismatch in number of edges and number of int16 attributes");
                    edge_attr_values.resize<int16_t>(attr_index.size_attr_index<int16_t>());
                    edge_attr_values.insert(attr_values_int16, idx);
                    break;
                  }
                case NPY_INT8:
                  {
                    size_t idx = attr_index.attr_index<int8_t>(attr_name);
                    py_array_to_vector<int8_t>(py_attr_values, attr_values_int8);
                    throw_assert(attr_values_int8.size() == num_edges,
                                 "build_edge_map: mismatch in number of edges and number of int8 attributes");
                    edge_attr_values.resize<int8_t>(attr_index.size_attr_index<int8_t>());
                    edge_attr_values.insert(attr_values_int8, idx);
                    break;
                  }
                case NPY_FLOAT:
                  {
                    size_t idx = attr_index.attr_index<float>(attr_name);
                    py_array_to_vector<float>(py_attr_values, attr_values_float);
                    throw_assert(attr_values_float.size() == num_edges,
                                 "build_edge_map: mismatch in number of edges and number of float attributes");
                    edge_attr_values.resize<float>(attr_index.size_attr_index<float>());
                    edge_attr_values.insert(attr_values_float, idx);
                    break;
                  }
                default:
                  throw runtime_error("Unsupported attribute type");
                  break;
                }
              attr_idx = attr_idx+1;
            }
                                   
          edge_attr_vector[ns_index] = edge_attr_values;
        }
      edge_map.insert(make_pair(node_idx, make_tuple (adj_values, edge_attr_vector)));
    }
}


void build_edge_attr_indexes (PyObject *py_edge_dict, map <string, pair <size_t, AttrIndex > >& edge_attr_index)
{ 
  PyObject *py_dst_dict_key, *py_dst_dict_value;
  Py_ssize_t dst_dict_pos = 0;

  while (PyDict_Next(py_edge_dict, &dst_dict_pos, &py_dst_dict_key, &py_dst_dict_value))
    {
      throw_assert(py_dst_dict_key != Py_None,
                   "build_edge_indexes: invalid key in edge dictionary");
      throw_assert(py_dst_dict_value != Py_None,
                   "build_edge_indexes: invalid value in edge dictionary");
      throw_assert(PyStr_Check(py_dst_dict_key),
                   "build_edge_indexes: non-string key in edge dictionary");
      PyObject *py_src_dict_key, *py_src_dict_value;
      Py_ssize_t src_dict_pos = 0;
      
      while (PyDict_Next(py_dst_dict_value, &src_dict_pos, &py_src_dict_key, &py_src_dict_value))
        {
          throw_assert(py_src_dict_key != Py_None,
                       "build_edge_maps: invalid key in edge dictionary");
          throw_assert(PyStr_Check(py_src_dict_key),
                       "build_edge_maps: non-string key in edge dictionary");
          get_edge_attr_index (py_src_dict_value, edge_attr_index);
          break;
        }
          
      break;
    }
  
}

void build_edge_maps (int rank, PyObject *py_edge_dict,
                      map <string, map <string, pair <map <string, pair <size_t, AttrIndex > >,
                                                      edge_map_t> > >& edge_maps)
{
  PyObject *py_dst_dict_key, *py_dst_dict_value;
  Py_ssize_t dst_dict_pos = 0;

  map <string, pair <size_t, AttrIndex > > edge_attr_index;

  build_edge_attr_indexes(py_edge_dict, edge_attr_index);

  
  while (PyDict_Next(py_edge_dict, &dst_dict_pos, &py_dst_dict_key, &py_dst_dict_value))
    {
      throw_assert(py_dst_dict_key != Py_None,
                   "build_edge_maps: invalid key in edge dictionary");
      throw_assert(py_dst_dict_value != Py_None,
                   "build_edge_maps: invalid value in edge dictionary");
      throw_assert(PyStr_Check(py_dst_dict_key),
                   "build_edge_maps: non-string key in edge dictionary");

      string dst_pop_name = string(PyStr_ToCString (py_dst_dict_key));
      PyObject *py_src_dict_key, *py_src_dict_value;
      Py_ssize_t src_dict_pos = 0;

      while (PyDict_Next(py_dst_dict_value, &src_dict_pos, &py_src_dict_key, &py_src_dict_value))
        {
          throw_assert(py_src_dict_key != Py_None,
                       "build_edge_maps: invalid key in edge dictionary");
          throw_assert(PyStr_Check(py_src_dict_key),
                       "build_edge_maps: non-string key in edge dictionary");
          
          string src_pop_name = string(PyStr_ToCString (py_src_dict_key));

          edge_map_t edge_map;

          build_edge_map (py_src_dict_value, edge_attr_index, edge_map);

          edge_maps[dst_pop_name][src_pop_name] = make_pair(edge_attr_index, edge_map);
        }
    }
}


PyObject* py_build_tree_value(const CELL_IDX_T key, const neurotree_t &tree,
                              const map <string, NamedAttrMap>& attr_maps,
                              const bool topology)
{
  const CELL_IDX_T idx = get<0>(tree);
  throw_assert(idx == key,
               "py_build_tree_value: tree index mismatch");

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
  npy_intp ind = 0;

  PyObject *py_section_topology = NULL;
  PyObject *py_section_vector = NULL;
  PyObject *py_section_src = NULL;
  PyObject *py_section_dst = NULL;
  PyObject *py_section_loc = NULL;
  PyObject *py_sections = NULL;
  
  if (topology)
    {
      npy_intp dims[1];
      dims[0] = num_nodes;
      py_section_topology = PyDict_New();
      py_section_vector = (PyObject *)PyArray_SimpleNew(1, dims, NPY_UINT16);
      SECTION_IDX_T *section_vector_ptr = (SECTION_IDX_T *)PyArray_GetPtr((PyArrayObject *)py_section_vector, &ind);
      size_t sections_ptr=0;
      SECTION_IDX_T section_idx = 0;
      PyObject *py_section_node_map = PyDict_New();
      PyObject *py_section_key; PyObject *py_section_nodes;
      set<NODE_IDX_T> marked_nodes;
      map <SECTION_IDX_T, vector<NODE_IDX_T> > section_node_map;
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
          NODE_IDX_T *py_section_nodes_ptr = (NODE_IDX_T *)PyArray_GetPtr((PyArrayObject *)py_section_nodes, &nodes_ind);
          sections_ptr++;
          for (size_t p = 0; p < num_section_nodes; p++)
            {
              NODE_IDX_T node_idx = sections[sections_ptr];
              throw_assert(node_idx <= num_nodes,
                           "py_build_tree_value: invalid node index in tree");

              py_section_nodes_ptr[p] = node_idx;
              section_nodes.push_back(node_idx);
              if (marked_nodes.find(node_idx) == marked_nodes.end())
                {
                  section_vector_ptr[node_idx] = section_idx;
                  marked_nodes.insert(node_idx);
                }
              sections_ptr++;
            }
          section_node_map.insert(make_pair(section_idx, section_nodes));
          PyDict_SetItem(py_section_node_map, py_section_key, py_section_nodes);
          Py_DECREF(py_section_nodes);
          Py_DECREF(py_section_key);
          section_idx++;
        }
      throw_assert(section_idx == num_sections,
                   "py_build_tree_value: invalid section index in tree");

      npy_intp topology_dims[1];
      topology_dims[0] = src_vector.size();
      py_section_src = (PyObject *)PyArray_SimpleNew(1, topology_dims, NPY_UINT16);
      SECTION_IDX_T *section_src_ptr = (SECTION_IDX_T *)PyArray_GetPtr((PyArrayObject *)py_section_src, &ind);
      py_section_dst = (PyObject *)PyArray_SimpleNew(1, topology_dims, NPY_UINT16);
      SECTION_IDX_T *section_dst_ptr = (SECTION_IDX_T *)PyArray_GetPtr((PyArrayObject *)py_section_dst, &ind);
      py_section_loc = (PyObject *)PyArray_SimpleNew(1, topology_dims, NPY_UINT32);
      NODE_IDX_T *section_loc_ptr = (NODE_IDX_T *)PyArray_GetPtr((PyArrayObject *)py_section_loc, &ind);
      for (size_t s = 0; s < src_vector.size(); s++)
        {
          section_src_ptr[s] = src_vector[s];
          section_dst_ptr[s] = dst_vector[s];
          auto node_map_it = section_node_map.find(src_vector[s]);
          throw_assert (node_map_it != section_node_map.end(),
                        "py_build_tree_value: invalid section index in tree source vector");
          // find parent point in destination section and determine location where the two sections are located
          vector<NODE_IDX_T>& src_section_nodes = node_map_it->second;
          node_map_it = section_node_map.find(dst_vector[s]);
          throw_assert (node_map_it != section_node_map.end(),
                        "py_build_tree_value: invalid section index in tree source vector");
          vector<NODE_IDX_T>& dst_section_nodes = node_map_it->second;
          const NODE_IDX_T dst_start = dst_section_nodes[0];
          const PARENT_NODE_IDX_T dst_start_parent = parents[dst_start];

          auto node_it = std::find(src_section_nodes.begin(), src_section_nodes.end(), dst_start);
          if (node_it != src_section_nodes.end())
            {
              section_loc_ptr[s] = *node_it;
            }
          else 
            {
              if (dst_start_parent != -1)
                {
                  node_it = std::find(src_section_nodes.begin(), src_section_nodes.end(), dst_start_parent);
                  if (node_it != src_section_nodes.end())
                    {
                      section_loc_ptr[s] = *node_it;
                    }
                  else
                    {
                      throw_err("py_build_tree: unable to determine connection point");
                    }
                }
              else
                {
                  throw_err("py_build_tree: unable to determine connection point");
                }
            }
        }

      PyObject *py_num_sections = PyLong_FromUnsignedLong(num_sections);
      PyDict_SetItemString(py_section_topology, "num_sections", py_num_sections);
      Py_DECREF(py_num_sections);
      PyDict_SetItemString(py_section_topology, "nodes", py_section_node_map);
      Py_DECREF(py_section_node_map);
      PyDict_SetItemString(py_section_topology, "src", py_section_src);
      Py_DECREF(py_section_src);
      PyDict_SetItemString(py_section_topology, "dst", py_section_dst);
      Py_DECREF(py_section_dst);
      PyDict_SetItemString(py_section_topology, "loc", py_section_loc);
      Py_DECREF(py_section_loc);
    }
  else
    {
      npy_intp topology_dims[1];
      topology_dims[0] = src_vector.size();
      py_section_src = (PyObject *)PyArray_SimpleNew(1, topology_dims, NPY_UINT16);
      SECTION_IDX_T *section_src_ptr = (SECTION_IDX_T *)PyArray_GetPtr((PyArrayObject *)py_section_src, &ind);
      py_section_dst = (PyObject *)PyArray_SimpleNew(1, topology_dims, NPY_UINT16);
      SECTION_IDX_T *section_dst_ptr = (SECTION_IDX_T *)PyArray_GetPtr((PyArrayObject *)py_section_dst, &ind);
      for (size_t s = 0; s < src_vector.size(); s++)
        {
          section_src_ptr[s] = src_vector[s];
          section_dst_ptr[s] = dst_vector[s];
        }

      
      npy_intp sections_dims[1];
      sections_dims[0] = sections.size();
      py_sections = (PyObject *)PyArray_SimpleNew(1, sections_dims, NPY_UINT16);
      SECTION_IDX_T *sections_ptr = (SECTION_IDX_T *)PyArray_GetPtr((PyArrayObject *)py_sections, &ind);
      for (size_t p = 0; p < sections.size(); p++)
        {
          sections_ptr[p] = sections[p];
        }
    }
  
                           

  npy_intp dims[1];
  dims[0] = num_nodes;

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
  Py_DECREF(py_xcoords);

  PyDict_SetItemString(py_treeval, "y", py_ycoords);
  Py_DECREF(py_ycoords);
                           
  PyDict_SetItemString(py_treeval, "z", py_zcoords);
  Py_DECREF(py_zcoords);

  PyDict_SetItemString(py_treeval, "radius", py_radiuses);
  Py_DECREF(py_radiuses);
                           
  PyDict_SetItemString(py_treeval, "layer", py_layers);
  Py_DECREF(py_layers);
                           
  PyDict_SetItemString(py_treeval, "parent", py_parents);
  Py_DECREF(py_parents);
                           
  PyDict_SetItemString(py_treeval, "swc_type", py_swc_types);
  Py_DECREF(py_swc_types);

  if (topology)
    {
      throw_assert(py_section_vector != NULL,
                   "py_build_tree_value: invalid section vector in tree");
      throw_assert(py_section_topology != NULL,
                   "py_build_tree_value: invalid section topology in tree");
      
      PyDict_SetItemString(py_treeval, "section", py_section_vector);
      Py_DECREF(py_section_vector);

      PyDict_SetItemString(py_treeval, "section_topology", py_section_topology);
      Py_DECREF(py_section_topology);
    }
  else
    {
      throw_assert(py_sections != NULL,
                   "py_build_tree_value: invalid sections in tree");
      throw_assert(py_section_src != NULL,
                   "py_build_tree_value: invalid source section vector in tree");
      throw_assert(py_section_dst != NULL,
                   "py_build_tree_value: invalid destination section vector in tree");

      PyDict_SetItemString(py_treeval, "sections", py_sections);
      Py_DECREF(py_sections);
      PyDict_SetItemString(py_treeval, "src", py_section_src);
      Py_DECREF(py_section_src);
      PyDict_SetItemString(py_treeval, "dst", py_section_dst);
      Py_DECREF(py_section_dst);
    }
                           
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
          Py_DECREF(py_value);

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
          Py_DECREF(py_value);
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
          Py_DECREF(py_value);
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
          Py_DECREF(py_value);
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
          Py_DECREF(py_value);
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
          Py_DECREF(py_value);
        }

      PyDict_SetItemString(py_treeval,
                           attr_namespace.c_str(),
                           py_namespace_dict);
      Py_DECREF(py_namespace_dict);
    }
                           

  return py_treeval;
}

/* NeuroH5TreeIterState - in-memory tree iterator instance.
 *
 * seq_index: index of the next id in the sequence to yield
 *
 */
typedef struct {
  Py_ssize_t seq_index, count;
                           
  vector <neurotree_t> tree_vector;
  vector<string> attr_name_spaces;
  map <string, NamedAttrMap> attr_maps;
  vector<neurotree_t>::const_iterator it_tree;
  bool topology_flag;
  
} NeuroH5TreeIterState;

typedef struct {
  PyObject_HEAD
  NeuroH5TreeIterState *state;
} PyNeuroH5TreeIterState;


PyObject* NeuroH5TreeIter_iter(PyObject *self)
{
  Py_INCREF(self);
  return self;
}

static void NeuroH5TreeIter_dealloc(PyNeuroH5TreeIterState *py_state)
{
  delete py_state->state;
  Py_TYPE(py_state)->tp_free(py_state);
}


PyObject* NeuroH5TreeIter_iternext(PyObject *self)
{
  PyNeuroH5TreeIterState *py_state = (PyNeuroH5TreeIterState *)self;
  if (py_state->state->it_tree != py_state->state->tree_vector.cend())
    {
      const neurotree_t &tree = *(py_state->state->it_tree);
      const CELL_IDX_T key = get<0>(tree);
      const map <string, NamedAttrMap>& attr_maps = py_state->state->attr_maps;

      PyObject *treeval = py_build_tree_value(key, tree, attr_maps, py_state->state->topology_flag);
      throw_assert(treeval != NULL,
                   "NeuroH5TreeIter: invalid tree value");
      
      py_state->state->it_tree++;
      py_state->state->seq_index++;

      PyObject *result = Py_BuildValue("lN", key, treeval);
                               
      return result;
    }
  else
    {
      /* Raising of standard StopIteration exception with empty value. */
      PyErr_SetNone(PyExc_StopIteration);
      return NULL;
    }
}


static PyTypeObject PyNeuroH5TreeIter_Type = {
  PyVarObject_HEAD_INIT(&PyType_Type, 0)
  "NeuroH5TreeIter",         /*tp_name*/
  sizeof(PyNeuroH5TreeIterState), /*tp_basicsize*/
  0,                         /*tp_itemsize*/
  (destructor)NeuroH5TreeIter_dealloc, /* tp_dealloc */
  0,                         /*tp_print*/
  0,                         /*tp_getattr*/
  0,                         /*tp_setattr*/
  0,                         /*tp_compare*/
  0,                         /*tp_repr*/
  0,                         /*tp_as_number*/
  0,                         /*tp_as_sequence*/
  0,                         /*tp_as_mapping*/
  0,                         /*tp_hash */
  0,                         /*tp_call*/
  0,                         /*tp_str*/
  0,                         /*tp_getattro*/
  0,                         /*tp_setattro*/
  0,                         /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_ITER,
  /* tp_flags: Py_TPFLAGS_HAVE_ITER tells python to
     use tp_iter and tp_iternext fields. */
  "In-memory tree iterator instance.",           /* tp_doc */
  0,  /* tp_traverse */
  0,  /* tp_clear */
  0,  /* tp_richcompare */
  0,  /* tp_weaklistoffset */
  NeuroH5TreeIter_iter,  /* tp_iter: __iter__() method */
  NeuroH5TreeIter_iternext  /* tp_iternext: next() method */
};



static PyObject *
NeuroH5TreeIter_FromVector(const vector <neurotree_t>& tree_vector,
                           const vector<string>& attr_name_spaces,
                           const map <string, NamedAttrMap>& attr_maps,
                           const bool topology_flag)
{

  PyNeuroH5TreeIterState *p = PyObject_New(PyNeuroH5TreeIterState, &PyNeuroH5TreeIter_Type);
  if (!p) return NULL;

  if (!PyObject_Init((PyObject *)p, &PyNeuroH5TreeIter_Type))
    {
      Py_DECREF(p);
      return NULL;
    }

  p->state = new NeuroH5TreeIterState();

  p->state->seq_index     = 0;
  p->state->count         = tree_vector.size();
  p->state->tree_vector   = tree_vector;
  p->state->attr_name_spaces = attr_name_spaces;
  p->state->attr_maps  = attr_maps;
  p->state->it_tree    = p->state->tree_vector.cbegin();
  p->state->topology_flag = topology_flag;

                           
  return (PyObject *)p;
}



static PyObject *
NeuroH5TreeIter_FromMap(const map<CELL_IDX_T, neurotree_t>& tree_map,
                        const vector<string>& attr_name_spaces,
                        const map <string, NamedAttrMap>& attr_maps,
                        const bool topology_flag)

{
  vector <neurotree_t> tree_vector;

  std::transform (tree_map.cbegin(), tree_map.cend(),
                  std::back_inserter(tree_vector),
                  [] (const map<CELL_IDX_T, neurotree_t>::value_type& kv)
                  { return kv.second; });

                           
  return (PyObject *)NeuroH5TreeIter_FromVector(tree_vector,
                                                attr_name_spaces,
                                                attr_maps,
                                                topology_flag);
}




PyObject* py_build_cell_attr_values_dict(const CELL_IDX_T key, 
                                         const NamedAttrMap& attr_map,
                                         const vector <vector<string> >& attr_names)
{
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
      Py_DECREF(py_value);
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
      Py_DECREF(py_value);
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
      Py_DECREF(py_value);

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
      Py_DECREF(py_value);

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
      Py_DECREF(py_value);

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
      Py_DECREF(py_value);

    }


  return py_attrval;
}

PyTypeObject* py_build_cell_attr_struct_type(const NamedAttrMap& attr_map,
                                             const vector <vector<string> >& attr_names,
                                             vector<PyStructSequence_Field> struct_descr_fields)
{
    PyStructSequence_Desc descr;
    
    PyTypeObject* structseq_type = NULL;

    if (attr_map.index_set.size() > 0)
      {
        CELL_IDX_T key = *attr_map.index_set.begin();
        
        const vector < vector <float>> &float_attrs      = attr_map.find<float>(key);
        const vector < vector <uint8_t> > &uint8_attrs   = attr_map.find<uint8_t>(key);
        const vector < vector <int8_t> > &int8_attrs     = attr_map.find<int8_t>(key);
        const vector < vector <uint16_t> > &uint16_attrs = attr_map.find<uint16_t>(key);
        const vector < vector <int16_t> > &int16_attrs   = attr_map.find<int16_t>(key);
        const vector < vector <uint32_t> > &uint32_attrs = attr_map.find<uint32_t>(key);
        const vector < vector <int32_t> > &int32_attrs   = attr_map.find<int32_t>(key);

        for (size_t i=0; i<float_attrs.size(); i++)
          {
            char *attr_name = attr_name_intern.add(attr_names[AttrMap::attr_index_float][i]);
            char *attr_type = attr_name_intern.add(string("float"));
            struct_descr_fields.push_back((PyStructSequence_Field){attr_name, attr_type});
          }
        for (size_t i=0; i<uint8_attrs.size(); i++)
          {
            char *attr_name = attr_name_intern.add(attr_names[AttrMap::attr_index_uint8][i].c_str());
            char *attr_type = attr_name_intern.add(string("uint8"));
            struct_descr_fields.push_back((PyStructSequence_Field){attr_name, attr_type});
          }
        for (size_t i=0; i<int8_attrs.size(); i++)
          {
            char *attr_name = attr_name_intern.add(attr_names[AttrMap::attr_index_int8][i].c_str());
            char *attr_type = attr_name_intern.add(string("int8"));
            struct_descr_fields.push_back((PyStructSequence_Field){attr_name, attr_type});
          }
        for (size_t i=0; i<uint16_attrs.size(); i++)
          {
            char *attr_name = attr_name_intern.add(attr_names[AttrMap::attr_index_uint16][i].c_str());
            char *attr_type = attr_name_intern.add(string("uint16"));
            struct_descr_fields.push_back((PyStructSequence_Field){attr_name, attr_type});
          }
        for (size_t i=0; i<int16_attrs.size(); i++)
          {
            char *attr_name = attr_name_intern.add(attr_names[AttrMap::attr_index_int16][i].c_str());
            char *attr_type = attr_name_intern.add(string("int16"));
            struct_descr_fields.push_back((PyStructSequence_Field){attr_name, attr_type});
          }
        for (size_t i=0; i<uint32_attrs.size(); i++)
          {
            char *attr_name = attr_name_intern.add(attr_names[AttrMap::attr_index_uint32][i].c_str());
            char *attr_type = attr_name_intern.add(string("uint32"));
            struct_descr_fields.push_back((PyStructSequence_Field){attr_name, attr_type});
          }
        for (size_t i=0; i<int32_attrs.size(); i++)
          {
            char *attr_name = attr_name_intern.add(attr_names[AttrMap::attr_index_int32][i].c_str());
            char *attr_type = attr_name_intern.add(string("int32"));
            struct_descr_fields.push_back((PyStructSequence_Field){attr_name, attr_type});
          }

        struct_descr_fields.push_back((PyStructSequence_Field){NULL, NULL});
        
        descr.name = attr_name_intern.add("neuroh5_cell_attributes");
        descr.doc = "NeuroH5 cell attributes";
        descr.fields = &struct_descr_fields[0];
        descr.n_in_sequence = struct_descr_fields.size()-1;
        
        structseq_type = PyStructSequence_NewType(&descr);
        throw_assert_nomsg(structseq_type != NULL);
        throw_assert_nomsg(PyType_Check(structseq_type));
      }
    
    return structseq_type;
}


PyObject* py_build_cell_attr_values_struct(const CELL_IDX_T key, 
                                           const NamedAttrMap& attr_map,
                                           const vector <vector<string> >& attr_names,
                                           PyTypeObject* struct_type)
{
  npy_intp dims[1];
  npy_intp ind = 0;

  

  PyObject* py_attrval = PyStructSequence_New(struct_type);
  throw_assert_nomsg(py_attrval != NULL);
  
  const vector < vector <float>> &float_attrs      = attr_map.find<float>(key);
  const vector < vector <uint8_t> > &uint8_attrs   = attr_map.find<uint8_t>(key);
  const vector < vector <int8_t> > &int8_attrs     = attr_map.find<int8_t>(key);
  const vector < vector <uint16_t> > &uint16_attrs = attr_map.find<uint16_t>(key);
  const vector < vector <int16_t> > &int16_attrs   = attr_map.find<int16_t>(key);
  const vector < vector <uint32_t> > &uint32_attrs = attr_map.find<uint32_t>(key);
  const vector < vector <int32_t> > &int32_attrs   = attr_map.find<int32_t>(key);

  size_t attr_pos = 0;
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

      PyStructSequence_SetItem(py_attrval, attr_pos++, py_value);
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
                               
      PyStructSequence_SetItem(py_attrval, attr_pos++, py_value);
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
                               
      PyStructSequence_SetItem(py_attrval, attr_pos++, py_value);
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
                               
      PyStructSequence_SetItem(py_attrval, attr_pos++, py_value);
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
                               
      PyStructSequence_SetItem(py_attrval, attr_pos++, py_value);
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
                               
      PyStructSequence_SetItem(py_attrval, attr_pos++, py_value);
    }

  //Py_DECREF(py_attrval);
  
  return py_attrval;
}


/* NeuroH5CellAttrIterState - in-memory cell attribute iterator instance.
 *
 * seq_index: index of the next id in the sequence to yield
 *
 */
typedef struct {
  Py_ssize_t seq_index, count;
                           
  string attr_namespace;
  vector< vector <string> > attr_names;
  NamedAttrMap attr_map;
  set<CELL_IDX_T>::const_iterator it_idx;
  bool return_struct;
  PyTypeObject* struct_type;
  vector<PyStructSequence_Field> struct_descr_fields;
                           
} NeuroH5CellAttrIterState;

typedef struct {
  PyObject_HEAD
  NeuroH5CellAttrIterState *state;
} PyNeuroH5CellAttrIterState;


PyObject* NeuroH5CellAttrIter_iter(PyObject *self)
{
  Py_INCREF(self);
  return self;
}

static void NeuroH5CellAttrIter_dealloc(PyNeuroH5CellAttrIterState *py_state)
{
  Py_DECREF(py_state->state->struct_type);
  delete py_state->state;
  Py_TYPE(py_state)->tp_free(py_state);
}


PyObject* NeuroH5CellAttrIter_iternext(PyObject *self)
{
  PyNeuroH5CellAttrIterState *py_state = (PyNeuroH5CellAttrIterState *)self;
  if (py_state->state->it_idx != py_state->state->attr_map.index_set.cend())
    {
      const CELL_IDX_T key = *(py_state->state->it_idx);
      PyObject *attrval;
      if (py_state->state->return_struct)
        {
          attrval = py_build_cell_attr_values_struct(key, py_state->state->attr_map,
                                                     py_state->state->attr_names,
                                                     py_state->state->struct_type);
        }
      else
        {
          attrval = py_build_cell_attr_values_dict(key, py_state->state->attr_map,
                                                   py_state->state->attr_names);
        }
      
      throw_assert(attrval != NULL,
                   "NeuroH5CellAttrIter: invalid attribute value");

      PyObject *result = Py_BuildValue("lN", key, attrval);

      py_state->state->it_idx++;
      py_state->state->seq_index++;
                               
      return result;
    }
  else
    {
      /* Raising of standard StopIteration exception with empty value. */
      PyErr_SetNone(PyExc_StopIteration);
      return NULL;
    }
}


static PyTypeObject PyNeuroH5CellAttrIter_Type = {
  PyVarObject_HEAD_INIT(&PyType_Type, 0)
  "NeuroH5CellAttrIter",         /*tp_name*/
  sizeof(PyNeuroH5CellAttrIterState), /*tp_basicsize*/
  0,                         /*tp_itemsize*/
  (destructor)NeuroH5CellAttrIter_dealloc, /* tp_dealloc */
  0,                         /*tp_print*/
  0,                         /*tp_getattr*/
  0,                         /*tp_setattr*/
  0,                         /*tp_compare*/
  0,                         /*tp_repr*/
  0,                         /*tp_as_number*/
  0,                         /*tp_as_sequence*/
  0,                         /*tp_as_mapping*/
  0,                         /*tp_hash */
  0,                         /*tp_call*/
  0,                         /*tp_str*/
  0,                         /*tp_getattro*/
  0,                         /*tp_setattro*/
  0,                         /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_ITER,
  /* tp_flags: Py_TPFLAGS_HAVE_ITER tells python to
     use tp_iter and tp_iternext fields. */
  "In-memory cell attribute iterator instance.",           /* tp_doc */
  0,  /* tp_traverse */
  0,  /* tp_clear */
  0,  /* tp_richcompare */
  0,  /* tp_weaklistoffset */
  NeuroH5CellAttrIter_iter,  /* tp_iter: __iter__() method */
  NeuroH5CellAttrIter_iternext  /* tp_iternext: next() method */
};



static PyObject *
NeuroH5CellAttrIter_FromMap(const string& attr_namespace,
                            const vector< vector <string> >& attr_names,
                            const NamedAttrMap& attr_map,
                            const bool return_struct)

{

  PyNeuroH5CellAttrIterState *p = PyObject_New(PyNeuroH5CellAttrIterState, &PyNeuroH5CellAttrIter_Type);
  if (!p) return NULL;

  if (!PyObject_Init((PyObject *)p, &PyNeuroH5CellAttrIter_Type))
    {
      Py_DECREF(p);
      return NULL;
    }

  p->state = new NeuroH5CellAttrIterState();
  p->state->seq_index     = 0;
  p->state->count         = attr_map.index_set.size();
  p->state->attr_map      = attr_map;
  p->state->attr_namespace = attr_namespace;
  p->state->attr_names    = attr_names;
  p->state->it_idx        = p->state->attr_map.index_set.cbegin();
  p->state->struct_type   = py_build_cell_attr_struct_type(attr_map, attr_names, p->state->struct_descr_fields);
  p->state->return_struct = return_struct;
  
  return (PyObject *)p;
}



template <class T>
void build_edge_attr_vec (const AttrVal& attr_val,
                          const size_t pos,
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
                               const vector <vector <string> >& attr_names,
                               PyObject *py_attrval)
{
  for (auto iter : attr_names)
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
          Py_DECREF(py_value);
        }
    }
}


PyObject* py_build_edge_value(const NODE_IDX_T key,
                              const NODE_IDX_T adj, 
                              const size_t pos, 
                              const map <string, NamedAttrVal >& attr_val_map,
                              const map <string, vector <vector<string> > >& attr_names)
{
  PyObject *py_attrval = Py_None;
  PyObject *py_attrmap = Py_None;

  for (auto const &it : attr_val_map)
    {
      const string& attr_namespace = it.first;

      const NamedAttrVal& attr_val = it.second;
      
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
                                attr_names.at(attr_namespace), py_attrval);
      py_build_edge_attr_value (uint8_attrs, NPY_UINT8, AttrMap::attr_index_uint8,
                                attr_names.at(attr_namespace), py_attrval);
      py_build_edge_attr_value (uint16_attrs, NPY_UINT16, AttrMap::attr_index_uint16,
                                attr_names.at(attr_namespace), py_attrval);
      py_build_edge_attr_value (uint32_attrs, NPY_UINT32, AttrMap::attr_index_uint32,
                                attr_names.at(attr_namespace), py_attrval);
      py_build_edge_attr_value (int8_attrs,   NPY_INT8, AttrMap::attr_index_int8,
                                attr_names.at(attr_namespace), py_attrval);
      py_build_edge_attr_value (int16_attrs,  NPY_INT16, AttrMap::attr_index_int16,
                                attr_names.at(attr_namespace), py_attrval);
      py_build_edge_attr_value (int32_attrs,  NPY_INT32, AttrMap::attr_index_int32,
                                attr_names.at(attr_namespace), py_attrval);
      
      PyDict_SetItemString(py_attrmap,
                           attr_namespace.c_str(),
                           py_attrval);
      Py_DECREF(py_attrval);

    }
  
  PyObject *py_result = PyTuple_New(3);

  PyObject *py_key = PyLong_FromLong(key);
  PyObject *py_adj = PyLong_FromLong(adj);
  
  PyTuple_SetItem(py_result, 0, py_key);
  PyTuple_SetItem(py_result, 1, py_adj);
  PyTuple_SetItem(py_result, 2, (attr_names.size()>0) ? py_attrmap : (Py_INCREF(Py_None), Py_None));
  
  return py_result;
}


PyObject* py_build_edge_tuple_value (const NODE_IDX_T key,
                                     const edge_tuple_t& et,
                                     const vector<string>& edge_attr_name_spaces)
{
  int status;
  const vector<NODE_IDX_T>& adj_vector = get<0>(et);
  const vector<AttrVal>& edge_attr_vector = get<1>(et);

  npy_intp dims[1], ind = 0;
  dims[0] = adj_vector.size();
                
  PyObject *adj_arr = PyArray_SimpleNew(1, dims, NPY_UINT32);
  uint32_t *adj_ptr = (uint32_t *)PyArray_GetPtr((PyArrayObject *)adj_arr, &ind);
  
  for (size_t j = 0; j < adj_vector.size(); j++)
    {
      adj_ptr[j] = adj_vector[j];
    }
  
  PyObject *py_attrmap  = PyDict_New();
  size_t namespace_index=0;
  for (auto const & edge_attr_values : edge_attr_vector)
    {
      PyObject *py_attrval  = PyList_New(0);
          
      for (size_t i = 0; i < edge_attr_values.size_attr_vec<float>(); i++)
        {
          PyObject *py_arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_FLOAT);
          float *ptr = (float *)PyArray_GetPtr((PyArrayObject *)py_arr, &ind);
          for (size_t j = 0; j < adj_vector.size(); j++)
            {
              ptr[j] = edge_attr_values.at<float>(i,j); 
            }
          status = PyList_Append(py_attrval, py_arr);
          throw_assert(status == 0,
                       "py_build_edge_tuple_value: unable to append to list");
          
          Py_DECREF(py_arr);
        }
      for (size_t i = 0; i < edge_attr_values.size_attr_vec<uint8_t>(); i++)
        {
          PyObject *py_arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_UINT8);
          uint8_t *ptr = (uint8_t *)PyArray_GetPtr((PyArrayObject *)py_arr, &ind);
          for (size_t j = 0; j < adj_vector.size(); j++)
            {
              ptr[j] = edge_attr_values.at<uint8_t>(i,j); 
            }
          status = PyList_Append(py_attrval, py_arr);
          throw_assert(status == 0,
                   "py_build_edge_tuple_value: unable to append to list");
          Py_DECREF(py_arr);
        }
      for (size_t i = 0; i < edge_attr_values.size_attr_vec<uint16_t>(); i++)
        {
          PyObject *py_arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_UINT16);
          uint16_t *ptr = (uint16_t *)PyArray_GetPtr((PyArrayObject *)py_arr, &ind);
          for (size_t j = 0; j < adj_vector.size(); j++)
            {
              ptr[j] = edge_attr_values.at<uint16_t>(i,j); 
            }
          status = PyList_Append(py_attrval, py_arr);
          throw_assert(status == 0,
                   "py_build_edge_tuple_value: unable to append to list");
          Py_DECREF(py_arr);
        }
      for (size_t i = 0; i < edge_attr_values.size_attr_vec<uint32_t>(); i++)
        {
          PyObject *py_arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_UINT32);
          uint32_t *ptr = (uint32_t *)PyArray_GetPtr((PyArrayObject *)py_arr, &ind);
          for (size_t j = 0; j < adj_vector.size(); j++)
            {
              ptr[j] = edge_attr_values.at<uint32_t>(i,j); 
            }
          status = PyList_Append(py_attrval, py_arr);
          throw_assert(status == 0,
                   "py_build_edge_tuple_value: unable to append to list");
          Py_DECREF(py_arr);
        }
      for (size_t i = 0; i < edge_attr_values.size_attr_vec<int8_t>(); i++)
        {
          PyObject *py_arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_INT8);
          int8_t *ptr = (int8_t *)PyArray_GetPtr((PyArrayObject *)py_arr, &ind);
          for (size_t j = 0; j < adj_vector.size(); j++)
            {
              ptr[j] = edge_attr_values.at<int8_t>(i,j); 
            }
          status = PyList_Append(py_attrval, py_arr);
          throw_assert(status == 0,
                   "py_build_edge_tuple_value: unable to append to list");
          Py_DECREF(py_arr);
        }
      for (size_t i = 0; i < edge_attr_values.size_attr_vec<int16_t>(); i++)
        {
          PyObject *py_arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_INT16);
          int16_t *ptr = (int16_t *)PyArray_GetPtr((PyArrayObject *)py_arr, &ind);
          for (size_t j = 0; j < adj_vector.size(); j++)
            {
              ptr[j] = edge_attr_values.at<int16_t>(i,j); 
            }
          status = PyList_Append(py_attrval, py_arr);
          throw_assert(status == 0,
                   "py_build_edge_tuple_value: unable to append to list");
          Py_DECREF(py_arr);
        }
      for (size_t i = 0; i < edge_attr_values.size_attr_vec<int32_t>(); i++)
        {
          PyObject *py_arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_INT32);
          int32_t *ptr = (int32_t *)PyArray_GetPtr((PyArrayObject *)py_arr, &ind);
          for (size_t j = 0; j < adj_vector.size(); j++)
            {
              ptr[j] = edge_attr_values.at<int32_t>(i,j); 
            }
          status = PyList_Append(py_attrval, py_arr);
          throw_assert(status == 0,
                   "py_build_edge_tuple_value: unable to append to list");
          Py_DECREF(py_arr);
        }
          
      PyDict_SetItemString(py_attrmap, edge_attr_name_spaces[namespace_index].c_str(), py_attrval);
      Py_DECREF(py_attrval);

      namespace_index++;
    }
                
  PyObject *py_edgeval  = PyTuple_New(2);
  PyTuple_SetItem(py_edgeval, 0, adj_arr);
  PyTuple_SetItem(py_edgeval, 1, py_attrmap);

  return py_edgeval;

}


PyObject* py_build_edge_array_dict_value (const NODE_IDX_T key,
                                          const edge_tuple_t& et,
                                          const vector<string>& edge_attr_name_spaces,
                                          const map <string, vector <vector<string> > >& attr_names)
{
  int status;
  const vector<NODE_IDX_T>& adj_vector = get<0>(et);
  const vector<AttrVal>& edge_attr_vector = get<1>(et);

  npy_intp dims[1], ind = 0;
  dims[0] = adj_vector.size();
                
  PyObject *adj_arr = PyArray_SimpleNew(1, dims, NPY_UINT32);
  uint32_t *adj_ptr = (uint32_t *)PyArray_GetPtr((PyArrayObject *)adj_arr, &ind);
  
  for (size_t j = 0; j < adj_vector.size(); j++)
    {
      adj_ptr[j] = adj_vector[j];
    }
  
  PyObject *py_attrmap  = PyDict_New();
  size_t namespace_index=0;
  for (auto const & edge_attr_values : edge_attr_vector)
    {
      PyObject *py_attrvalmap  = PyDict_New();
      const string& attr_namespace = edge_attr_name_spaces[namespace_index];
      
      for (size_t i = 0; i < edge_attr_values.size_attr_vec<float>(); i++)
        {
          PyObject *py_arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_FLOAT);
          float *ptr = (float *)PyArray_GetPtr((PyArrayObject *)py_arr, &ind);
          for (size_t j = 0; j < adj_vector.size(); j++)
            {
              ptr[j] = edge_attr_values.at<float>(i,j); 
            }
          status = PyDict_SetItemString(py_attrvalmap, attr_names.at(attr_namespace)[AttrVal::attr_index_float][i].c_str(), py_arr);
          throw_assert(status == 0,
                   "py_build_edge_array_dict_value: unable to set dictionary item");
          Py_DECREF(py_arr);
        }
      for (size_t i = 0; i < edge_attr_values.size_attr_vec<uint8_t>(); i++)
        {
          PyObject *py_arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_UINT8);
          uint8_t *ptr = (uint8_t *)PyArray_GetPtr((PyArrayObject *)py_arr, &ind);
          for (size_t j = 0; j < adj_vector.size(); j++)
            {
              ptr[j] = edge_attr_values.at<uint8_t>(i,j); 
            }
          status = PyDict_SetItemString(py_attrvalmap, attr_names.at(attr_namespace)[AttrVal::attr_index_uint8][i].c_str(), py_arr);
          throw_assert(status == 0,
                   "py_build_edge_array_dict_value: unable to set dictionary item");
          Py_DECREF(py_arr);
        }
      for (size_t i = 0; i < edge_attr_values.size_attr_vec<uint16_t>(); i++)
        {
          PyObject *py_arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_UINT16);
          uint16_t *ptr = (uint16_t *)PyArray_GetPtr((PyArrayObject *)py_arr, &ind);
          for (size_t j = 0; j < adj_vector.size(); j++)
            {
              ptr[j] = edge_attr_values.at<uint16_t>(i,j); 
            }
          status = PyDict_SetItemString(py_attrvalmap, attr_names.at(attr_namespace)[AttrVal::attr_index_uint16][i].c_str(), py_arr);
          throw_assert(status == 0,
                   "py_build_edge_array_dict_value: unable to set dictionary item");
          Py_DECREF(py_arr);
        }
      for (size_t i = 0; i < edge_attr_values.size_attr_vec<uint32_t>(); i++)
        {
          PyObject *py_arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_UINT32);
          uint32_t *ptr = (uint32_t *)PyArray_GetPtr((PyArrayObject *)py_arr, &ind);
          for (size_t j = 0; j < adj_vector.size(); j++)
            {
              ptr[j] = edge_attr_values.at<uint32_t>(i,j); 
            }
          status = PyDict_SetItemString(py_attrvalmap, attr_names.at(attr_namespace)[AttrVal::attr_index_uint32][i].c_str(), py_arr);
          throw_assert(status == 0,
                   "py_build_edge_array_dict_value: unable to set dictionary item");
          Py_DECREF(py_arr);
        }
      for (size_t i = 0; i < edge_attr_values.size_attr_vec<int8_t>(); i++)
        {
          PyObject *py_arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_INT8);
          int8_t *ptr = (int8_t *)PyArray_GetPtr((PyArrayObject *)py_arr, &ind);
          for (size_t j = 0; j < adj_vector.size(); j++)
            {
              ptr[j] = edge_attr_values.at<int8_t>(i,j); 
            }
          status = PyDict_SetItemString(py_attrvalmap, attr_names.at(attr_namespace)[AttrVal::attr_index_int8][i].c_str(), py_arr);
          throw_assert(status == 0,
                   "py_build_edge_array_dict_value: unable to set dictionary item");
          Py_DECREF(py_arr);
        }
      for (size_t i = 0; i < edge_attr_values.size_attr_vec<int16_t>(); i++)
        {
          PyObject *py_arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_INT16);
          int16_t *ptr = (int16_t *)PyArray_GetPtr((PyArrayObject *)py_arr, &ind);
          for (size_t j = 0; j < adj_vector.size(); j++)
            {
              ptr[j] = edge_attr_values.at<int16_t>(i,j); 
            }
          status = PyDict_SetItemString(py_attrvalmap, attr_names.at(attr_namespace)[AttrVal::attr_index_int16][i].c_str(), py_arr);
          throw_assert(status == 0,
                   "py_build_edge_array_dict_value: unable to set dictionary item");
          Py_DECREF(py_arr);
        }
      for (size_t i = 0; i < edge_attr_values.size_attr_vec<int32_t>(); i++)
        {
          PyObject *py_arr = (PyObject *)PyArray_SimpleNew(1, dims, NPY_INT32);
          int32_t *ptr = (int32_t *)PyArray_GetPtr((PyArrayObject *)py_arr, &ind);
          for (size_t j = 0; j < adj_vector.size(); j++)
            {
              ptr[j] = edge_attr_values.at<int32_t>(i,j); 
            }
          status = PyDict_SetItemString(py_attrvalmap, attr_names.at(attr_namespace)[AttrVal::attr_index_int32][i].c_str(), py_arr);
          throw_assert(status == 0,
                   "py_build_edge_array_dict_value: unable to set dictionary item");
          Py_DECREF(py_arr);
        }
          
      PyDict_SetItemString(py_attrmap, attr_namespace.c_str(), py_attrvalmap);
      Py_DECREF(py_attrvalmap);

      namespace_index++;
    }
                
  PyObject *py_edgeval  = PyTuple_New(2);
  PyTuple_SetItem(py_edgeval, 0, adj_arr);
  PyTuple_SetItem(py_edgeval, 1, py_attrmap);

  return py_edgeval;

}

/* NeuroH5EdgeIterState - in-memory edge iterator instance.
 *
 * seq_index: index of the next id in the sequence to yield
 *
 */
typedef struct {
  Py_ssize_t seq_index, count;
  
  edge_map_t edge_map;
  vector<string> edge_attr_name_spaces;
  
  edge_map_iter_t it_edge;
  
} NeuroH5EdgeIterState;

typedef struct {
  PyObject_HEAD
  NeuroH5EdgeIterState *state;
} PyNeuroH5EdgeIterState;


PyObject* NeuroH5EdgeIter_iter(PyObject *self)
{
  Py_INCREF(self);
  return self;
}

static void NeuroH5EdgeIter_dealloc(PyNeuroH5EdgeIterState *py_state)
{
  delete py_state->state;
  Py_TYPE(py_state)->tp_free(py_state);
}


PyObject* NeuroH5EdgeIter_iternext(PyObject *self)
{
  PyNeuroH5EdgeIterState *py_state = (PyNeuroH5EdgeIterState *)self;
  if (py_state->state->it_edge != py_state->state->edge_map.cend())
    {
      const NODE_IDX_T key      = py_state->state->it_edge->first;
      const edge_tuple_t& et    = py_state->state->it_edge->second;

      PyObject* py_edge_tuple_value = py_build_edge_tuple_value (key, et, py_state->state->edge_attr_name_spaces);
      throw_assert(py_edge_tuple_value != NULL,
                   "NeuroH5EdgeIter: invalid edge tuple value");

      py_state->state->it_edge++;
      py_state->state->seq_index++;

      PyObject *result = Py_BuildValue("lN", key, py_edge_tuple_value);
      return result;
    }
  else
    {
      /* Raising of standard StopIteration exception with empty value. */
      PyErr_SetNone(PyExc_StopIteration);
      return NULL;
    }
}


static PyTypeObject PyNeuroH5EdgeIter_Type = {
  PyVarObject_HEAD_INIT(&PyType_Type, 0)
  "NeuroH5EdgeIter",         /*tp_name*/
  sizeof(PyNeuroH5EdgeIterState), /*tp_basicsize*/
  0,                         /*tp_itemsize*/
  (destructor)NeuroH5EdgeIter_dealloc, /* tp_dealloc */
  0,                         /*tp_print*/
  0,                         /*tp_getattr*/
  0,                         /*tp_setattr*/
  0,                         /*tp_compare*/
  0,                         /*tp_repr*/
  0,                         /*tp_as_number*/
  0,                         /*tp_as_sequence*/
  0,                         /*tp_as_mapping*/
  0,                         /*tp_hash */
  0,                         /*tp_call*/
  0,                         /*tp_str*/
  0,                         /*tp_getattro*/
  0,                         /*tp_setattro*/
  0,                         /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_ITER,
  /* tp_flags: Py_TPFLAGS_HAVE_ITER tells python to
     use tp_iter and tp_iternext fields. */
  "In-memory edge iterator instance.",           /* tp_doc */
  0,  /* tp_traverse */
  0,  /* tp_clear */
  0,  /* tp_richcompare */
  0,  /* tp_weaklistoffset */
  NeuroH5EdgeIter_iter,  /* tp_iter: __iter__() method */
  NeuroH5EdgeIter_iternext  /* tp_iternext: next() method */
};


static PyObject *
NeuroH5EdgeIter_FromMap(const edge_map_t& prj_edge_map,
                        const vector <string>& edge_attr_name_spaces)
{

  PyNeuroH5EdgeIterState *p = PyObject_New(PyNeuroH5EdgeIterState,
                                           &PyNeuroH5EdgeIter_Type);
  if (!p) return NULL;

  if (!PyObject_Init((PyObject *)p, &PyNeuroH5EdgeIter_Type))
    {
      Py_DECREF(p);
      return NULL;
    }

  p->state = new NeuroH5EdgeIterState();

  p->state->seq_index     = 0;
  p->state->count         = prj_edge_map.size();
  p->state->edge_map      = prj_edge_map;
  p->state->edge_attr_name_spaces = edge_attr_name_spaces;
  p->state->it_edge       = p->state->edge_map.cbegin();
  
  return (PyObject *)p;
}

extern "C"
{

  
  static PyObject *py_read_graph (PyObject *self, PyObject *args, PyObject *kwds)
  {
    int status;
    vector < map <string, vector < vector <string> > > > edge_attr_name_vector;
    vector<edge_map_t> prj_vector;
    vector< pair<string,string> > prj_names;
    char *input_file_name;
    PyObject *py_attr_name_spaces=NULL;
    PyObject *py_comm = NULL;
    MPI_Comm *comm_ptr  = NULL;
    size_t total_num_nodes, total_num_edges = 0, local_num_edges = 0;

    static const char *kwlist[] = {
                                   "file_name",
                                   "namespaces",
                                   "comm",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|OO", (char **)kwlist,
                                     &input_file_name,
                                     &py_attr_name_spaces,
                                     &py_comm))
      return NULL;

    PyObject *py_prj_dict = PyDict_New();
    MPI_Comm comm;

    if (py_comm != NULL)
      {
        comm_ptr = PyMPIComm_Get(py_comm);
        throw_assert(comm_ptr != NULL,
                     "py_read_graph: invalid MPI communicator");
        throw_assert(*comm_ptr != MPI_COMM_NULL,
                     "py_read_graph: invalid MPI communicator");
        status = MPI_Comm_dup(*comm_ptr, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_read_graph: unable to duplicate MPI communicator");
      }
    else
      {
        status = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_read_graph: unable to duplicate MPI communicator");

      }

    vector <string> edge_attr_name_spaces;
    // Create C++ vector of namespace strings:
    if (py_attr_name_spaces != NULL)
      {
        for (size_t i = 0; (Py_ssize_t)i < PyList_Size(py_attr_name_spaces); i++)
          {
            PyObject *pyval = PyList_GetItem(py_attr_name_spaces, (Py_ssize_t)i);
            const char *str = PyStr_ToCString (pyval);
            edge_attr_name_spaces.push_back(string(str));
          }
      }

    status = graph::read_projection_names(comm, input_file_name, prj_names) >= 0;
    throw_assert(status >= 0,
                 "py_read_graph: unable to read projection names");

    graph::read_graph(comm, std::string(input_file_name), edge_attr_name_spaces,
                      prj_names, prj_vector, edge_attr_name_vector,
                      total_num_nodes, local_num_edges, total_num_edges);
    status = MPI_Comm_free(&comm);
    throw_assert(status == MPI_SUCCESS,
                 "py_read_graph: unable to free MPI communicator");
    
    PyObject *py_attribute_info = py_build_edge_attribute_info (prj_names,
                                                                edge_attr_name_spaces,
                                                                edge_attr_name_vector);
    
    for (size_t i = 0; i < prj_vector.size(); i++)
      {
        edge_map_t prj_edge_map = prj_vector[i];

        PyObject *py_edge_iter = NeuroH5EdgeIter_FromMap(prj_edge_map, edge_attr_name_spaces);

        PyObject *py_src_dict = PyDict_GetItemString(py_prj_dict, prj_names[i].second.c_str());
        if (py_src_dict == NULL)
          {
            py_src_dict = PyDict_New();
            PyDict_SetItemString(py_src_dict, prj_names[i].first.c_str(), py_edge_iter);
            PyDict_SetItemString(py_prj_dict, prj_names[i].second.c_str(), py_src_dict);
            Py_DECREF(py_edge_iter);
            Py_DECREF(py_src_dict);
          }
        else
          {
            PyDict_SetItemString(py_src_dict, prj_names[i].first.c_str(), py_edge_iter);
            Py_DECREF(py_edge_iter);
          }
        
      }

    PyObject *py_prj_tuple = PyTuple_New(2);
    PyTuple_SetItem(py_prj_tuple, 0, py_prj_dict);
    PyTuple_SetItem(py_prj_tuple, 1, py_attribute_info);

    return py_prj_tuple;
  }

  

  
  static PyObject *py_scatter_read_graph (PyObject *self, PyObject *args, PyObject *kwds)
  {
    int status;
    int opt_edge_map_type=0;
    EdgeMapType edge_map_type = EdgeMapDst;
    // A vector that maps nodes to compute ranks
    PyObject *py_node_rank_map=NULL;
    PyObject *py_attr_name_spaces=NULL;
    PyObject *py_prj_names=NULL;
    map<NODE_IDX_T, rank_t> node_rank_map;
    vector < edge_map_t > prj_vector;
    vector < map <string, vector < vector<string> > > > edge_attr_name_vector;
    vector<pop_range_t> pop_vector;
    map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
    vector<pair<string,string> > prj_names;
    PyObject *py_prj_dict = PyDict_New();
    unsigned long io_size; int size;
    PyObject *py_comm = NULL;
    MPI_Comm *comm_ptr = NULL;
    
    char *input_file_name;
    size_t local_num_nodes = 0, total_num_nodes = 0, total_num_edges = 0, local_num_edges = 0;
    
    static const char *kwlist[] = {
                                   "file_name",
                                   "comm",
                                   "node_rank_map",
                                   "projections",
                                   "namespaces",
                                   "map_type",
                                   "io_size",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|OOOOik", (char **)kwlist,
                                     &input_file_name, &py_comm, 
                                     &py_node_rank_map, &py_prj_names,
                                     &py_attr_name_spaces,
                                     &opt_edge_map_type, &io_size))
      return NULL;

    MPI_Comm comm;

    if (py_comm != NULL)
      {
        comm_ptr = PyMPIComm_Get(py_comm);
        throw_assert(comm_ptr != NULL,
                     "py_scatter_read_graph: invalid MPI communicator");
        throw_assert(*comm_ptr != MPI_COMM_NULL,
                     "py_scatter_read_graph: invalid MPI communicator");
        status = MPI_Comm_dup(*comm_ptr, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_scatter_read_graph: unable to duplicate MPI communicator");
      }
    else
      {
        status = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_scatter_read_graph: unable to duplicate MPI communicator");
      }
    
    if (opt_edge_map_type == 1)
      {
        edge_map_type = EdgeMapSrc;
      }
    
    status = MPI_Comm_size(comm, &size);
    throw_assert(status == MPI_SUCCESS,
                 "py_scatter_read_graph: unable to obtain size of MPI communicator");
    
    if (io_size == 0)
      {
        io_size = size;
      }

    // Create C++ vector of projections to read
    if (py_prj_names != NULL)
      {
        for (size_t i = 0; (Py_ssize_t)i < PyList_Size(py_prj_names); i++)
          {
            PyObject *pyval = PyList_GetItem(py_prj_names, (Py_ssize_t)i);
            PyObject *p1    = PyTuple_GetItem(pyval, 0);
            PyObject *p2    = PyTuple_GetItem(pyval, 1);
            const char *s1        = PyStr_ToCString (p1);
            const char *s2        = PyStr_ToCString (p2);
            prj_names.push_back(make_pair(string(s1), string(s2)));
          }
      }
    else
      {
        status = graph::read_projection_names(comm, input_file_name, prj_names);
        throw_assert(status >= 0,
                     "py_scatter_read_graph: unable to read projection names");
      }
    
    vector <string> edge_attr_name_spaces;
    // Create C++ vector of namespace strings:
    if (py_attr_name_spaces != NULL)
      {
        for (size_t i = 0; (Py_ssize_t)i < PyList_Size(py_attr_name_spaces); i++)
          {
            PyObject *pyval = PyList_GetItem(py_attr_name_spaces, (Py_ssize_t)i);
            const char *str = PyStr_ToCString (pyval);
            edge_attr_name_spaces.push_back(string(str));
          }
        sort(edge_attr_name_spaces.begin(), edge_attr_name_spaces.end());
      }
    

    // Read population info to determine total_num_nodes
    status = cell::read_population_ranges(comm, input_file_name, pop_ranges, pop_vector, total_num_nodes);
    throw_assert(status >= 0,
                 "py_scatter_read_graph: unable to read population ranges");


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

    graph::scatter_read_graph(comm, edge_map_type, std::string(input_file_name),
                              io_size, edge_attr_name_spaces, prj_names, node_rank_map,
                              prj_vector, edge_attr_name_vector,
                              local_num_nodes, total_num_nodes,
                              local_num_edges, total_num_edges);
    status = MPI_Comm_free(&comm);
    throw_assert(status == MPI_SUCCESS,
                 "py_read_graph: unable to free MPI communicator");

    PyObject *py_attribute_info = py_build_edge_attribute_info (prj_names,
                                                                edge_attr_name_spaces,
                                                                edge_attr_name_vector);
    
    for (size_t i = 0; i < prj_vector.size(); i++)
      {
        edge_map_t prj_edge_map = prj_vector[i];

        PyObject *py_edge_iter = NeuroH5EdgeIter_FromMap(prj_edge_map, edge_attr_name_spaces);

        PyObject *py_src_dict = PyDict_GetItemString(py_prj_dict, prj_names[i].second.c_str());
        if (py_src_dict == NULL)
          {
            py_src_dict = PyDict_New();
            PyDict_SetItemString(py_src_dict, prj_names[i].first.c_str(), py_edge_iter);
            PyDict_SetItemString(py_prj_dict, prj_names[i].second.c_str(), py_src_dict);
            Py_DECREF(py_edge_iter);
            Py_DECREF(py_src_dict);
          }
        else
          {
            PyDict_SetItemString(py_src_dict, prj_names[i].first.c_str(), py_edge_iter);
            Py_DECREF(py_edge_iter);
          }
        
      }

    PyObject *py_prj_tuple = PyTuple_New(2);
    PyTuple_SetItem(py_prj_tuple, 0, py_prj_dict);
    PyTuple_SetItem(py_prj_tuple, 1, py_attribute_info);

    return py_prj_tuple;

  }

  
  static PyObject *py_bcast_graph (PyObject *self, PyObject *args, PyObject *kwds)
  {
    int status;
    int opt_edge_map_type=0;
    EdgeMapType edge_map_type = EdgeMapDst;
    vector < edge_map_t > prj_vector;
    vector < map < string, vector <vector<string> > > > edge_attr_name_vector;
    vector<pop_range_t> pop_vector;
    map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
    vector< pair<string,string> > prj_names;
    PyObject *py_prj_dict = PyDict_New();
    int size;
    char *input_file_name;
    PyObject *py_comm = NULL;
    MPI_Comm *comm_ptr  = NULL;
    PyObject *py_attr_name_spaces=NULL;
    size_t total_num_nodes, total_num_edges = 0, local_num_edges = 0;
    
    static const char *kwlist[] = {
                                   "file_name",
                                   "comm",
                                   "namespaces",
                                   "map_type",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|OOi", (char **)kwlist,
                                     &input_file_name, &py_comm, 
                                     &py_attr_name_spaces, &opt_edge_map_type))
      return NULL;

    MPI_Comm comm;

    if (py_comm != NULL)
      {
        comm_ptr = PyMPIComm_Get(py_comm);
        throw_assert(comm_ptr != NULL,
                     "py_bcast_graph: invalid MPI communicator");
        throw_assert(*comm_ptr != MPI_COMM_NULL,
                     "py_bcast_graph: invalid MPI communicator");
        status = MPI_Comm_dup(*comm_ptr, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_bcast_graph: unable to duplicate MPI communicator");
      }
    else
      {
        status = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_bcast_graph: unable to duplicate MPI communicator");
      }

    if (opt_edge_map_type == 1)
      {
        edge_map_type = EdgeMapSrc;
      }
    
    assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS);
    throw_assert(status == MPI_SUCCESS,
                 "py_bcast_graph: unable to obtain size of MPI communicator");

    vector <string> edge_attr_name_spaces;
    // Create C++ vector of namespace strings:
    if (py_attr_name_spaces != NULL)
      {
        for (size_t i = 0; (Py_ssize_t)i < PyList_Size(py_attr_name_spaces); i++)
          {
            PyObject *pyval = PyList_GetItem(py_attr_name_spaces, (Py_ssize_t)i);
            const char *str = PyStr_ToCString (pyval);
            edge_attr_name_spaces.push_back(string(str));
          }
      }

    status = graph::read_projection_names(comm, input_file_name, prj_names);
    throw_assert(status >= 0,
                 "py_bcast_graph: unable to read projection names");


    // Read population info to determine total_num_nodes
    status = cell::read_population_ranges(comm, input_file_name, pop_ranges, pop_vector, total_num_nodes);
    throw_assert(status >= 0,
                 "py_bcast_graph: unable to read population ranges");

    graph::bcast_graph(comm, edge_map_type, std::string(input_file_name),
                       edge_attr_name_spaces, prj_names, prj_vector, edge_attr_name_vector, 
                       total_num_nodes, local_num_edges, total_num_edges);
    status = MPI_Comm_free(&comm);
    throw_assert(status == MPI_SUCCESS,
                 "py_bcast_graph: unable to free MPI communicator");
    

    PyObject *py_attribute_info = PyDict_New();
    for (size_t p = 0; p<edge_attr_name_vector.size(); p++)
      {
        PyObject *py_prj_attr_info  = PyDict_New();
        for (string& attr_namespace : edge_attr_name_spaces) 
          {
            PyObject *py_prj_ns_attr_info  = PyDict_New();
            int attr_index=0;
            const vector <vector <string> > ns_edge_attr_names = edge_attr_name_vector[p].at(attr_namespace);
            for (size_t n = 0; n<ns_edge_attr_names.size(); n++)
              {
                for (size_t t = 0; t<ns_edge_attr_names[n].size(); t++)
                  {
                    PyObject *py_attr_index = PyLong_FromLong(attr_index);
                    
                    PyDict_SetItemString(py_prj_ns_attr_info, ns_edge_attr_names[n][t].c_str(), py_attr_index);
                    Py_DECREF(py_attr_index);
                    attr_index++;
                  }
              }
            PyDict_SetItemString(py_prj_attr_info, attr_namespace.c_str(), py_prj_ns_attr_info);
            Py_DECREF(py_prj_ns_attr_info);

          }
        PyObject *py_prj_key = PyTuple_New(2);
        PyTuple_SetItem(py_prj_key, 0, PyStr_FromCString(prj_names[p].first.c_str()));
        PyTuple_SetItem(py_prj_key, 1, PyStr_FromCString(prj_names[p].second.c_str()));
        PyDict_SetItem(py_attribute_info, py_prj_key, py_prj_attr_info);
        Py_DECREF(py_prj_key);
        Py_DECREF(py_prj_attr_info);
      }

    for (size_t i = 0; i < prj_vector.size(); i++)
      {
        PyObject *py_edge_dict = PyDict_New();
        edge_map_t prj_edge_map = prj_vector[i];
        
        if (prj_edge_map.size() > 0)
          {
            for (auto const& it : prj_edge_map)
              {
                const NODE_IDX_T key_node = it.first;
                const edge_tuple_t& et    = it.second;

                PyObject* py_edge_tuple_value = py_build_edge_tuple_value (key_node, et, edge_attr_name_spaces);

                PyObject *key = PyLong_FromLong(key_node);
                PyDict_SetItem(py_edge_dict, key, py_edge_tuple_value);
                Py_DECREF(key);
                Py_DECREF(py_edge_tuple_value);

              }
          }
        
        PyObject *py_src_dict = PyDict_GetItemString(py_prj_dict, prj_names[i].second.c_str());
        if (py_src_dict == NULL)
          {
            py_src_dict = PyDict_New();
            PyDict_SetItemString(py_src_dict, prj_names[i].first.c_str(), py_edge_dict);
            PyDict_SetItemString(py_prj_dict, prj_names[i].second.c_str(), py_src_dict);
            Py_DECREF(py_edge_dict);
            Py_DECREF(py_src_dict);
          }
        else
          {
            PyDict_SetItemString(py_src_dict, prj_names[i].first.c_str(), py_edge_dict);
            Py_DECREF(py_edge_dict);
          }
        
      }

    PyObject *py_prj_tuple = PyTuple_New(2);
    PyTuple_SetItem(py_prj_tuple, 0, py_prj_dict);
    PyTuple_SetItem(py_prj_tuple, 1, py_attribute_info);
    return py_prj_tuple;
  }

  static PyObject *py_read_graph_selection (PyObject *self, PyObject *args, PyObject *kwds)
  {
    int status;
    vector < map <string, vector < vector <string> > > > edge_attr_name_vector;
    vector<edge_map_t> prj_vector;
    vector< pair<string,string> > prj_names;
    char *input_file_name;
    PyObject *py_selection=NULL;
    PyObject *py_attr_name_spaces=NULL;
    PyObject *py_comm = NULL;
    PyObject *py_prj_names = NULL;
    MPI_Comm *comm_ptr  = NULL;
    vector <NODE_IDX_T> selection;
    size_t total_num_nodes, total_num_edges = 0, local_num_edges = 0;

    static const char *kwlist[] = {
                                   "file_name",
                                   "selection",
                                   "namespaces",
                                   "comm",
                                   "projections",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO|OOO", (char **)kwlist,
                                     &input_file_name,
                                     &py_selection, 
                                     &py_attr_name_spaces,
                                     &py_comm,
                                     &py_prj_names))
      return NULL;

    PyObject *py_prj_dict = PyDict_New();
    MPI_Comm comm;

    if (py_comm != NULL)
      {
        comm_ptr = PyMPIComm_Get(py_comm);
        throw_assert(comm_ptr != NULL,
                     "py_read_graph_selection: invalid MPI communicator");
        throw_assert(*comm_ptr != MPI_COMM_NULL,
                     "py_read_graph_selection: invalid MPI communicator");
        status = MPI_Comm_dup(*comm_ptr, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_read_graph_selection: unable to duplicate MPI communicator");

      }
    else
      {
        status = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_read_graph_selection: unable to duplicate MPI communicator");
      }

    vector <string> edge_attr_name_spaces;
    // Create C++ vector of namespace strings:
    if (py_attr_name_spaces != NULL)
      {
        for (size_t i = 0; (Py_ssize_t)i < PyList_Size(py_attr_name_spaces); i++)
          {
            PyObject *pyval = PyList_GetItem(py_attr_name_spaces, (Py_ssize_t)i);
            const char *str = PyStr_ToCString (pyval);
            edge_attr_name_spaces.push_back(string(str));
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

    if (py_prj_names != NULL)
      {
        for (size_t i = 0; (Py_ssize_t)i < PyList_Size(py_prj_names); i++)
          {
            PyObject *pyval = PyList_GetItem(py_prj_names, (Py_ssize_t)i);
            PyObject *p1    = PyTuple_GetItem(pyval, 0);
            PyObject *p2    = PyTuple_GetItem(pyval, 1);
            const char *s1        = PyStr_ToCString (p1);
            const char *s2        = PyStr_ToCString (p2);
            prj_names.push_back(make_pair(string(s1), string(s2)));
          }
      }
    else
      {
        status = graph::read_projection_names(comm, input_file_name, prj_names);
        throw_assert(status >= 0,
                     "py_read_graph_selection: unable to read projection names");
      }

    graph::read_graph_selection(comm, std::string(input_file_name), edge_attr_name_spaces,
                                prj_names, selection, prj_vector, edge_attr_name_vector,
                                total_num_nodes, local_num_edges, total_num_edges);
    assert(MPI_Comm_free(&comm) == MPI_SUCCESS);
    throw_assert(status == MPI_SUCCESS,
                 "py_read_graph_selection: unable to free MPI communicator");
    
    PyObject *py_attribute_info = py_build_edge_attribute_info (prj_names,
                                                                edge_attr_name_spaces,
                                                                edge_attr_name_vector);
    
    for (size_t i = 0; i < prj_vector.size(); i++)
      {
        edge_map_t prj_edge_map = prj_vector[i];

        PyObject *py_edge_iter = NeuroH5EdgeIter_FromMap(prj_edge_map, edge_attr_name_spaces);

        PyObject *py_src_dict = PyDict_GetItemString(py_prj_dict, prj_names[i].second.c_str());
        if (py_src_dict == NULL)
          {
            py_src_dict = PyDict_New();
            PyDict_SetItemString(py_src_dict, prj_names[i].first.c_str(), py_edge_iter);
            PyDict_SetItemString(py_prj_dict, prj_names[i].second.c_str(), py_src_dict);
            Py_DECREF(py_edge_iter);
            Py_DECREF(py_src_dict);
          }
        else
          {
            PyDict_SetItemString(py_src_dict, prj_names[i].first.c_str(), py_edge_iter);
            Py_DECREF(py_edge_iter);
          }
        
      }

    PyObject *py_prj_tuple = PyTuple_New(2);
    PyTuple_SetItem(py_prj_tuple, 0, py_prj_dict);
    PyTuple_SetItem(py_prj_tuple, 1, py_attribute_info);

    return py_prj_tuple;
  }


  static PyObject *py_write_graph (PyObject *self, PyObject *args, PyObject *kwds)
  {
    int status;
    PyObject *edge_values = NULL;
    PyObject *py_comm  = NULL;
    MPI_Comm *comm_ptr = NULL;
    char *file_name_arg, *src_pop_name_arg, *dst_pop_name_arg;
    unsigned long io_size = 0;
    
    static const char *kwlist[] = {
                                   "file_name",
                                   "src_pop_name",
                                   "dst_pop_name",
                                   "edges",
                                   "comm",
                                   "io_size",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sssO|Ok", (char **)kwlist,
                                     &file_name_arg, &src_pop_name_arg, &dst_pop_name_arg,
                                     &edge_values, &py_comm, &io_size))
      return NULL;
    MPI_Comm comm;

    if (py_comm != NULL)
      {
        comm_ptr = PyMPIComm_Get(py_comm);
        throw_assert(comm_ptr != NULL,
                     "py_read_graph_selection: invalid MPI communicator");
        throw_assert(*comm_ptr != MPI_COMM_NULL,
                     "py_read_graph_selection: invalid MPI communicator");
        status = MPI_Comm_dup(*comm_ptr, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_read_graph_selection: unable to duplicate MPI communicator");
      }
    else
      {
        status = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_read_graph_selection: unable to duplicate MPI communicator");

      }

    Py_ssize_t dict_size = PyDict_Size(edge_values);
    int data_color = 2;
    
    MPI_Comm data_comm;
    // In cases where some ranks do not have any data to write, split
    // the communicator, so that collective operations can be executed
    // only on the ranks that do have data.
    if (dict_size > 0)
      {
        MPI_Comm_split(comm,data_color,0,&data_comm);
      }
    else
      {
        MPI_Comm_split(comm,0,0,&data_comm);
      }
    MPI_Comm_set_errhandler(data_comm, MPI_ERRORS_RETURN);

    if (dict_size > 0)
      {
        int rank, size;
        status = MPI_Comm_size(data_comm, &size);
        throw_assert(status == MPI_SUCCESS,
                     "py_write_graph: unable to obtain size of MPI communicator");
        status = MPI_Comm_rank(data_comm, &rank);
        throw_assert(status == MPI_SUCCESS,
                     "py_write_graph: unable to obtain rank of MPI communicator");
        
        if (io_size == 0)
          {
            io_size = size;
          }
    
        string file_name = string(file_name_arg);
        string src_pop_name = string(src_pop_name_arg);
        string dst_pop_name = string(dst_pop_name_arg);
        
        edge_map_t edge_map;
        map <string, pair <size_t, AttrIndex > > edge_attr_index;
        
        get_edge_attr_index (edge_values, edge_attr_index);
        
        build_edge_map(edge_values, edge_attr_index, edge_map);
        
        status = graph::write_graph(data_comm, io_size, file_name, src_pop_name, dst_pop_name,
                                    edge_attr_index, edge_map);
        throw_assert(status >= 0,
                     "py_write_graph: unable to write graph");
      }

    status = MPI_Barrier(data_comm);
    throw_assert(status == MPI_SUCCESS,
                 "py_write_graph: barrier error");
    status = MPI_Barrier(comm);
    throw_assert(status == MPI_SUCCESS,
                 "py_write_graph: barrier error");
    status = MPI_Comm_free(&data_comm);
    throw_assert(status == MPI_SUCCESS,
                 "py_write_graph: unable to free MPI communicator");
    status = MPI_Comm_free(&comm);
    throw_assert(status == MPI_SUCCESS,
                 "py_write_graph: unable to free MPI communicator");

    Py_INCREF(Py_None);
    return Py_None;
  }
  

  static PyObject *py_append_graph (PyObject *self, PyObject *args, PyObject *kwds)
  {
    int status;
    PyObject *py_edge_dict;
    PyObject *py_comm = NULL;
    MPI_Comm *comm_ptr = NULL;
    char *file_name_arg;
    unsigned long io_size = 0;
    
    static const char *kwlist[] = {
                                   "file_name",
                                   "edge_dict",
                                   "comm",
                                   "io_size",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO|Ok", (char **)kwlist,
                                     &file_name_arg, &py_edge_dict,
                                     &py_comm, &io_size))
      return NULL;

    MPI_Comm comm;

    if (py_comm != NULL)
      {
        comm_ptr = PyMPIComm_Get(py_comm);
        throw_assert(comm_ptr != NULL,
                     "py_append_graph: invalid MPI communicator");
        throw_assert(*comm_ptr != MPI_COMM_NULL,
                     "py_append_graph: invalid MPI communicator");
        status = MPI_Comm_dup(*comm_ptr, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_append_graph: unable to duplicate MPI communicator");
      }
    else
      {
        status = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_append_graph: unable to duplicate MPI communicator");
      }
    
    string file_name = string(file_name_arg);

    Py_ssize_t dict_size = PyDict_Size(py_edge_dict);
    int data_color = 2;
    
    MPI_Comm data_comm;
    // In cases where some ranks do not have any data to write, split
    // the communicator, so that collective operations can be executed
    // only on the ranks that do have data.
    if (dict_size > 0)
      {
        MPI_Comm_split(comm,data_color,0,&data_comm);
      }
    else
      {
        MPI_Comm_split(comm,0,0,&data_comm);
      }
    MPI_Comm_set_errhandler(data_comm, MPI_ERRORS_RETURN);

    if (dict_size > 0)
      {
        int rank, size;
        status = MPI_Comm_size(data_comm, &size);
        throw_assert(status == MPI_SUCCESS,
                     "py_append_graph: unable to obtain size of MPI communicator");
        status = MPI_Comm_rank(data_comm, &rank);
        throw_assert(status == MPI_SUCCESS,
                     "py_append_graph: unable to obtain rank of MPI communicator");

        if (io_size == 0)
          {
            io_size = size;
          }
        
        map <string, map <string, pair <map <string, pair <size_t, AttrIndex > >, edge_map_t> > > edge_maps;
        
        build_edge_maps (rank, py_edge_dict, edge_maps);

        for (auto const& dst_edge_map_item : edge_maps)
          {
            const string & dst_pop_name = dst_edge_map_item.first;

            for (auto const& edge_map_item : dst_edge_map_item.second)
              {
                const string & src_pop_name = edge_map_item.first;
                const map <string, pair <size_t, AttrIndex > >& edge_attr_index = edge_map_item.second.first; 
                const edge_map_t & edge_map = edge_map_item.second.second; 

                status = graph::append_graph(data_comm, io_size, file_name, src_pop_name, dst_pop_name,
                                             edge_attr_index, edge_map);
                throw_assert(status >= 0,
                             "py_append_graph: unable to append projection");
                
              }
          }

      }
    
    status = MPI_Barrier(data_comm);
    throw_assert(status == MPI_SUCCESS,
                 "py_append_graph: barrier error");
    status = MPI_Barrier(comm);
    throw_assert(status == MPI_SUCCESS,
                 "py_append_graph: barrier error");
    status = MPI_Comm_free(&data_comm);
    throw_assert(status == MPI_SUCCESS,
                 "py_append_graph: unable to free MPI communicator");
    status = MPI_Comm_free(&comm);
    throw_assert(status == MPI_SUCCESS,
                 "py_append_graph: unable to free MPI communicator");
    
    Py_INCREF(Py_None);
    return Py_None;
  }
  
  PyDoc_STRVAR(
    read_population_names_doc,
    "read_population_names(file_name, comm=None)\n"
    "--\n"
    "\n"
    "Returns the names of all populations for which attributes exist in the given file.\n"
    "Parameters\n"
    "----------\n"
    "file_name : string\n"
    "    The NeuroH5 file to read.\n"
    "    \n"
    "    .. warning::\n"
    "       The given file must be a valid HDF5 file that contains /H5Types and /Populations groups.\n"
    "\n"
    "comm : MPI communicator\n"
    "    Optional MPI communicator. If None, the world communicator will be used.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "populations : list\n"
    "    A list of strings with the populations names\n"
    "\n");

  
  
  static PyObject *py_read_population_names (PyObject *self, PyObject *args, PyObject *kwds)
  {
    int status; 
    char *input_file_name;
    PyObject *py_comm = NULL;
    MPI_Comm *comm_ptr  = NULL;

    static const char *kwlist[] = {
                                   "file_name",
                                   "comm",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|O", (char **)kwlist,
                                     &input_file_name, &py_comm))
      return NULL;

    MPI_Comm comm;

    if (py_comm != NULL)
      {
        comm_ptr = PyMPIComm_Get(py_comm);
        throw_assert(comm_ptr != NULL,
                     "py_read_population_names: invalid MPI communicator");
        throw_assert(*comm_ptr != MPI_COMM_NULL,
                     "py_read_population_names: invalid MPI communicator");
        status = MPI_Comm_dup(*comm_ptr, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_read_population_names: unable to duplicate MPI communicator");
      }
    else
      {
        status = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_read_population_names: unable to duplicate MPI communicator");
      }

    int rank, size;
    status = MPI_Comm_size(comm, &size);
    throw_assert(status == MPI_SUCCESS,
                 "py_read_population_names: unable to obtain size of MPI communicator");
    status = MPI_Comm_rank(comm, &rank);
    throw_assert(status == MPI_SUCCESS,
                 "py_read_population_names: unable to obtain rank of MPI communicator");

    vector <string> pop_names;
    status = cell::read_population_names(comm, input_file_name, pop_names);
    throw_assert(status == MPI_SUCCESS,
                 "py_read_population_names: unable to read population names");


    status = MPI_Comm_free(&comm);
    throw_assert(status == MPI_SUCCESS,
                 "py_read_population_names: unable to free MPI communicator");


    PyObject *py_population_names = PyList_New(0);
    for (size_t i=0; i<pop_names.size(); i++)
      {
        PyObject *name = PyStr_FromCString(pop_names[i].c_str());
        PyList_Append(py_population_names, name);
        Py_DECREF(name);
      }
    
    return py_population_names;
  }
  
  PyDoc_STRVAR(
    read_cell_attribute_info_doc,
    "read_cell_attribute_info(file_name, populations, read_cell_index=False, comm=None)\n"
    "--\n"
    "\n"
    "Returns information about the attributes which are defined for the given populations in the given file.\n"
    "Parameters\n"
    "----------\n"
    "file_name : string\n"
    "    The NeuroH5 file to read.\n"
    "    \n"
    "    .. warning::\n"
    "       The given file must be a valid HDF5 file that contains /H5Types and /Populations groups.\n"
    "populations : string list\n"
    "     The populations for which attribute information should be returned\n"
    "\n"
    "read_cell_index : bool\n"
    "     Optional flag that specifies whether to return the cell ids for which attributes are defined in the given file.\n"
    "\n"
    "comm : MPI communicator\n"
    "    Optional MPI communicator. If None, the world communicator will be used.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "{ population : { namespace : [ \"attribute name\", ... ] } }\n"
    "    A dictionary that maps population names to dictionaries that map attribute name spaces to lists of attribute names.\n"
    "\n");

  
  static PyObject *py_read_cell_attribute_info (PyObject *self, PyObject *args, PyObject *kwds)
  {
    int status; 
    char *input_file_name;
    bool read_cell_index = false;
    int read_cell_index_flag = 0;
    PyObject *py_comm = NULL, *py_pop_names = NULL;
    MPI_Comm *comm_ptr  = NULL;

    static const char *kwlist[] = {
                                   "file_name",
                                   "populations",
                                   "read_cell_index",
                                   "comm",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO|iO", (char **)kwlist,
                                     &input_file_name, &py_pop_names, &read_cell_index_flag, &py_comm))
      return NULL;

    status = PyList_Check(py_pop_names);
    throw_assert(status > 0,
                 "py_read_cell_attribute_info: invalid pop_names list");

    read_cell_index = read_cell_index_flag>0;

    MPI_Comm comm;

    if (py_comm != NULL)
      {
        comm_ptr = PyMPIComm_Get(py_comm);
        throw_assert(comm_ptr != NULL,
                     "py_read_cell_attribute_info: invalid MPI communicator");
        throw_assert(*comm_ptr != MPI_COMM_NULL,
                     "py_read_cell_attribute_info: invalid MPI communicator");
        status = MPI_Comm_dup(*comm_ptr, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_read_cell_attribute_info: unable to duplicate MPI communicator");
      }
    else
      {
        status = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_read_cell_attribute_info: unable to duplicate MPI communicator");
      }

    int srank, ssize; size_t rank, size;
    status = MPI_Comm_size(comm, &ssize);
    throw_assert(status == MPI_SUCCESS,
                 "py_read_cell_attribute_info: unable to obtain size of MPI communicator");
    status = MPI_Comm_rank(comm, &srank);
    throw_assert(status == MPI_SUCCESS,
                 "py_read_cell_attribute_info: unable to obtain rank of MPI communicator");

    size = ssize;
    rank = srank;
    
    vector <string> pop_names;
    if (py_pop_names != NULL)
      {
        for (size_t i = 0; (Py_ssize_t)i < PyList_Size(py_pop_names); i++)
          {
            PyObject *pyval = PyList_GetItem(py_pop_names, (Py_ssize_t)i);
            const char *str = PyStr_ToCString (pyval);
            pop_names.push_back(string(str));
          }
      }

    size_t root = 0;

    vector< pair<pop_t, string> > pop_labels;
    status = cell::read_population_labels(comm, input_file_name, pop_labels);
    throw_assert(status >= 0,
                 "py_read_cell_attribute_info: unable to read population labels");

    size_t n_nodes;
    map<CELL_IDX_T, pair<uint32_t,pop_t> > pop_ranges;
    vector<pop_range_t> pop_vector;
    status = cell::read_population_ranges(comm,
                                          string(input_file_name),
                                          pop_ranges, pop_vector,
                                          n_nodes);
    throw_assert(status >= 0,
                 "py_read_cell_attribute_info: unable to read population ranges");

    
    map<string, map<string, vector<string> > > pop_attribute_info;
    map<string, map<string, map <string, vector<CELL_IDX_T> > > > cell_index_info;
    if (rank == root)
      {

        for (const string& pop_name : pop_names)
          {
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


            vector <string> name_spaces;
            status = cell::get_cell_attribute_name_spaces (input_file_name, pop_name, name_spaces);
            throw_assert (status >= 0,
                          "py_read_cell_attribute_info: unable to read cell attributes namespaces");
            for(const string& name_space : name_spaces)
              {
                vector< pair<string, AttrKind> > ns_attributes;

                status = cell::get_cell_attributes (input_file_name, name_space, pop_name, ns_attributes);
                throw_assert (status >= 0,
                              "py_read_cell_attribute_info: unable to read cell attributes metadata");

                for (auto const& it : ns_attributes)
                  {
                    const string attr_name = it.first;

                    pop_attribute_info[pop_name][name_space].push_back(attr_name);

                    if (read_cell_index)
                      {
                        status = cell::read_cell_index(comm,
                                                       input_file_name,
                                                       pop_name,
                                                       name_space + "/" + attr_name,
                                                       cell_index_info[pop_name][name_space][attr_name]);
                        throw_assert(status >= 0,
                                     "py_read_cell_attribute_info: unable to read cell attribute index");
                        vector<CELL_IDX_T>& cell_index_vector = cell_index_info[pop_name][name_space][attr_name];
                        for (size_t i=0; i<cell_index_vector.size(); i++)
                          {
                            cell_index_vector[i] += pop_vector[pop_idx].start;
                          }
                      }
                  }
              }
          }
      }

    throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                 "py_read_cell_attribute_info: barrier error");
    
    {
      vector<char> sendbuf;
      size_t sendbuf_size=0;
      if ((rank == root) && (pop_attribute_info.size() > 0) )
        {
          data::serialize_data(pop_attribute_info, sendbuf);
          sendbuf_size = sendbuf.size();
        }

      status = MPI_Bcast(&sendbuf_size, 1, MPI_SIZE_T, root, comm);
      throw_assert(status == MPI_SUCCESS,
                   "py_read_cell_attribute_info: broadcast error");
      
      sendbuf.resize(sendbuf_size);
      status = MPI_Bcast(&sendbuf[0], sendbuf_size, MPI_CHAR, root, comm);
      throw_assert(status == MPI_SUCCESS,
                   "py_read_cell_attribute_info: broadcast error");
      
      if ((rank != root) && (sendbuf_size > 0))
        {
          data::deserialize_data(sendbuf, pop_attribute_info);
        }
    }

    {
      vector<char> sendbuf;
      size_t sendbuf_size=0;
      if ((rank == root) && (cell_index_info.size() > 0) )
        {
          data::serialize_data(cell_index_info, sendbuf);
          sendbuf_size = sendbuf.size();
        }
      
      status = MPI_Bcast(&sendbuf_size, 1, MPI_SIZE_T, root, comm);
      throw_assert(status == MPI_SUCCESS,
                   "py_read_cell_attribute_info: broadcast error");
      
      sendbuf.resize(sendbuf_size);
      status = MPI_Bcast(&sendbuf[0], sendbuf_size, MPI_CHAR, root, comm);
      throw_assert(status == MPI_SUCCESS,
                   "py_read_cell_attribute_info: broadcast error");
      
      if ((rank != root) && (sendbuf_size > 0))
        {
          data::deserialize_data(sendbuf, cell_index_info);
        }
    }

    PyObject *py_population_attribute_info = PyDict_New();

    for (auto const& it : pop_attribute_info)
      {
        PyObject *py_ns_attribute_info = PyDict_New();

        for (auto const& it_ns : it.second)
          {
            PyObject *py_attribute_infos  = PyList_New(0);

            if (cell_index_info.size() > 0)
              {
                for (const string& name : it_ns.second)
                  {
                    PyObject *py_name = PyStr_FromCString(name.c_str());
                    PyObject *py_cell_index = PyList_New(0);
                    PyObject *py_info_tuple = PyTuple_New(2);

                    for(auto const& value: cell_index_info[it.first][it_ns.first][name])
                      {
                        status = PyList_Append(py_cell_index, PyLong_FromLong((long)value));
                        throw_assert(status == 0,
                                     "py_read_cell_attribute_info: list append error");

                      }

                    PyTuple_SetItem(py_info_tuple, 0, py_name);
                    PyTuple_SetItem(py_info_tuple, 1, py_cell_index);
                    throw_assert(status == 0,
                                 "py_read_cell_attribute_info: list append error");
                    status = PyList_Append(py_attribute_infos, py_info_tuple);

                    Py_DECREF(py_info_tuple);
                  }
              }
            else
              {
                for (const string& name : it_ns.second)
                  {
                    PyObject *py_name = PyStr_FromCString(name.c_str());
                    status = PyList_Append(py_attribute_infos, py_name);
                    throw_assert(status == 0,
                                 "py_read_cell_attribute_info: list append error");

                    Py_DECREF(py_name);
                  }
              }
            
            PyDict_SetItemString(py_ns_attribute_info,
                                 it_ns.first.c_str(),
                                 py_attribute_infos);
            Py_DECREF(py_attribute_infos);        
          }

        PyDict_SetItemString(py_population_attribute_info,
                             it.first.c_str(),
                             py_ns_attribute_info);
        Py_DECREF(py_ns_attribute_info);        
      }

    status = MPI_Comm_free(&comm);
    throw_assert(status == MPI_SUCCESS,
                 "py_read_cell_attribute_info: unable to free MPI communicator");

                 
    return py_population_attribute_info;
  }


  PyDoc_STRVAR(
    read_population_ranges_doc,
    "read_population_ranges(file_name, comm=None)\n"
    "--\n"
    "\n"
    "Returns population size and range for each population defined in the input file.\n"
    "Parameters\n"
    "----------\n"
    "file_name : string\n"
    "    The NeuroH5 file to read.\n"
    "    \n"
    "    .. warning::\n"
    "       The given file must be a valid HDF5 file that contains an /H5Types group.\n"
    "\n"
    "comm : MPI communicator\n"
    "    Optional MPI communicator. If None, the world communicator will be used.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "(range_dict, n_nodes) : tuple\n"
    "    A tuple with the following elements:  \n"
    "    - range_dict: { population: (size, offset) }\n"
    "       a dictionary where the key is population name and the value is tuple (size, offset).\n"
    "    - n_nodes: int\n"
    "       the total number of cells.\n"
    "\n");


  static PyObject *py_read_population_ranges (PyObject *self, PyObject *args, PyObject *kwds)
  {
    int status; 
    vector< pair<pop_t, string> > pop_labels;
    PyObject *py_comm = NULL;
    MPI_Comm *comm_ptr  = NULL;
    
    char *input_file_name;

    static const char *kwlist[] = {
                                   "file_name",
                                   "comm",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|O", (char **)kwlist,
                                     &input_file_name, &py_comm))
      return NULL;

    MPI_Comm comm;

    if (py_comm != NULL)
      {
        comm_ptr = PyMPIComm_Get(py_comm);
        throw_assert(comm_ptr != NULL,
                     "py_read_population_ranges: invalid MPI communicator");
        throw_assert(*comm_ptr != MPI_COMM_NULL,
                     "py_read_population_ranges: invalid MPI communicator");
        status = MPI_Comm_dup(*comm_ptr, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_read_population_ranges: unable to duplicate MPI communicator");
      }
    else
      {
        status = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_read_population_ranges: unable to duplicate MPI communicator");
      }

    status = cell::read_population_labels(comm, input_file_name, pop_labels);
    throw_assert (status >= 0,
                  "py_read_population_ranges: unable to read population labels");

    size_t n_nodes;
    map<CELL_IDX_T, pair<uint32_t,pop_t> > pop_ranges;
    vector<pop_range_t> pop_vector;
    status = cell::read_population_ranges(comm,
                                          string(input_file_name),
                                          pop_ranges, pop_vector,
                                          n_nodes);
    throw_assert(status >= 0,
                 "py_read_population_ranges: unable to read population labels");
    status = MPI_Comm_free(&comm);
    throw_assert(status == MPI_SUCCESS,
                 "py_read_population_ranges: unable to free MPI communicator");
    
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
            Py_DECREF(py_range_tuple);

          }
      }

    PyObject *py_result_tuple = PyTuple_New(2);

    PyTuple_SetItem(py_result_tuple, 0, py_population_ranges_dict);
    PyTuple_SetItem(py_result_tuple, 1, PyLong_FromLong((long)n_nodes));
    
    return py_result_tuple;
  }

  PyDoc_STRVAR(
    read_projection_names_doc,
    "read_projection_names(file_name, comm=None)\n"
    "--\n"
    "\n"
    "Returns the names of the projections contained in the given file.\n"
    "Parameters\n"
    "----------\n"
    "file_name : string\n"
    "    The NeuroH5 file to read.\n"
    "    \n"
    "    .. warning::\n"
    "       The given file must be a valid HDF5 file that contains /H5Types and /Populations groups.\n"
    "\n"
    "comm : MPI communicator\n"
    "    Optional MPI communicator. If None, the world communicator will be used.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "projections : (source, destination) list\n"
    "    A list of tuples with (source, destination) population names corresponding to each projection.\n"
    "\n");

  static PyObject *py_read_projection_names (PyObject *self, PyObject *args, PyObject *kwds)
  {
    int status;
    vector< pair<string,string> > prj_names;
    char *input_file_name;
    PyObject *py_comm = NULL;
    MPI_Comm *comm_ptr  = NULL;

    static const char *kwlist[] = {
                                   "file_name",
                                   "comm",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|O", (char **)kwlist,
                                     &input_file_name, &py_comm))
      return NULL;

    MPI_Comm comm;

    if (py_comm != NULL)
      {
        comm_ptr = PyMPIComm_Get(py_comm);
        throw_assert(comm_ptr != NULL,
                     "py_read_projection_names: invalid MPI communicator");
        throw_assert(*comm_ptr != MPI_COMM_NULL,
                     "py_read_projection_names: invalid MPI communicator");
        status = MPI_Comm_dup(*comm_ptr, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_read_projection_names: unable to duplicate MPI communicator");
      }
    else
      {
        status = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_read_projection_names: unable to duplicate MPI communicator");
      }


    PyObject *py_result  = PyList_New(0);

    if (graph::read_projection_names(comm, string(input_file_name), prj_names) >= 0)
      {

        for (auto name_pair: prj_names)
          {
            PyObject *py_pairval = PyTuple_New(2);
            PyTuple_SetItem(py_pairval, 0, PyStr_FromCString(name_pair.first.c_str()));
            PyTuple_SetItem(py_pairval, 1, PyStr_FromCString(name_pair.second.c_str()));
            status = PyList_Append(py_result, py_pairval);
            throw_assert(status == 0,
                         "py_read_projection_names: unable to append to list");
            Py_DECREF(py_pairval);
          }
      }

    status = MPI_Comm_free(&comm);
    throw_assert(status == MPI_SUCCESS,
                 "py_read_projection_names: unable to free communicator");
    
    return py_result;
  }

  PyDoc_STRVAR(
    read_trees_doc,
    "read_trees(file_name, population_name, namespaces=[], topology=True, comm=None)\n"
    "--\n"
    "\n"
    "Reads neuronal tree morphologies contained in the given file. "
    "Each rank will be assigned an equal number of morphologies, with the exception of the last rank if the number of morphologies is not evenly divisible by the number of ranks. \n"
    "\n"
    "Parameters\n"
    "----------\n"
    "file_name : string\n"
    "    The NeuroH5 file to read.\n"
    "    \n"
    "    .. warning::\n"
    "       The given file must be a valid HDF5 file that contains /H5Types and /Populations/Trees groups.\n"
    "\n"
    "population_name : string\n"
    "    Name of population from which to read.\n"
    "\n"
    "namespaces : string list\n"
    "    An optional list of namespaces from which additional attributes for the trees will be read.\n"
    "\n"
    "topology : boolean\n"
    "    An optional flag that specifies whether section topology dictionary should be returned.\n"
    "\n"
    "comm : MPI communicator\n"
    "    Optional MPI communicator. If None, the world communicator will be used.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "(tree_iter, n_nodes) : tuple\n"
    "    A tuple with the following elements:  \n"
    "    - tree_iter: ( gid, tree_dict }\n"
    "       An iterator that returns pairs (gid, tree_dict) where gid is the cell id and tree_dict is a morphology dictionary with the following fields: \n"
    "          - x: X coordinates of tree morphology points (float ndarray)\n"
    "          - y: Y coordinates of tree morphology points (float ndarray)\n"
    "          - z: Z coordinates of tree morphology points (float ndarray)\n"
    "          - radius: radiuses of tree morphology points (float ndarray)\n"
    "          - layer: layer index of tree morphology points (-1 is undefined) (int8 ndarray)\n"
    "          - parent: parent index of tree morphology points (-1 is undefined) (int32 ndarray)\n"
    "          - swc_type: SWC type of tree morphology points (enumerated ndarray)\n"
    "          If topology is True :\n"
    "             - section: the section to which each point is assigned (uint16 ndarray)\n"
    "             - section_topology: a dictionary with the following fields: \n"
    "               - num_sections: number of sections in the morphology\n"
    "               - nodes: a dictionary that maps each section indices to the point indices contained in that section\n"
    "               - src: a vector of source section indices (uint16 ndarray)\n"
    "               - dst: a vector of destination section indices (uint16 ndarray)\n"
    "               - loc: a vector of source section connection point indices (uint32 ndarray)\n"
    "          If topology is False :\n"
    "             - sections: for each section, the number of points, followed by the corresponding point indices (uint16 ndarray)\n"
    "             - src: a vector of source section indices (uint16 ndarray)\n"
    "             - dst: a vector of destination section indices (uint16 ndarray)\n"
    "    - n_nodes: int\n"
    "    - n_nodes: int\n"
    "       the total number of cells.\n"
    "\n");
  
  static PyObject *py_read_trees (PyObject *self, PyObject *args, PyObject *kwds)
  {
    int status; int topology_flag=1; 
    PyObject *py_comm = NULL;
    PyObject *py_mask = NULL;
    MPI_Comm *comm_ptr  = NULL;
    char *file_name, *pop_name;
    PyObject *py_attr_name_spaces=NULL;
    
    static const char *kwlist[] = {
                                   "file_name",
                                   "population_name",
                                   "comm",
                                   "mask",
                                   "namespaces",
                                   "topology",
                                   "io_size",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ss|OOOik", (char **)kwlist,
                                     &file_name, &pop_name, &py_comm, &py_mask, 
                                     &py_attr_name_spaces, 
                                     &topology_flag))
      return NULL;

    set<string> attr_mask;

    if (py_mask != NULL)
      {
        throw_assert(PySet_Check(py_mask),
                     "py_read_trees: argument mask must be a set of strings");
        
        PyObject *py_iter = PyObject_GetIter(py_mask);
        if (py_iter != NULL)
          {
            PyObject *pyval;
            while((pyval = PyIter_Next(py_iter)))
              {
                const char* str = PyStr_ToCString (pyval);
                attr_mask.insert(string(str));
                Py_DECREF(pyval);
              }
          }

        Py_DECREF(py_iter);
      }

    MPI_Comm comm;

    if (py_comm != NULL)
      {
        comm_ptr = PyMPIComm_Get(py_comm);
        throw_assert(comm_ptr != NULL,
                     "py_read_trees: invalid MPI communicator");
        throw_assert(*comm_ptr != MPI_COMM_NULL,
                     "py_read_trees: invalid MPI communicator");
        status = MPI_Comm_dup(*comm_ptr, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_read_trees: unable to duplicate MPI communicator");
      }
    else
      {
        status = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_read_trees: unable to duplicate MPI communicator");
      }

    vector <string> attr_name_spaces;
    // Create C++ vector of namespace strings:
    if (py_attr_name_spaces != NULL)
      {
        for (size_t i = 0; (Py_ssize_t)i < PyList_Size(py_attr_name_spaces); i++)
          {
            PyObject *pyval = PyList_GetItem(py_attr_name_spaces, (Py_ssize_t)i);
            const char *str = PyStr_ToCString (pyval);
            attr_name_spaces.push_back(string(str));
          }
      }

    vector<pair <pop_t, string> > pop_labels;
    status = cell::read_population_labels(comm, string(file_name), pop_labels);
    throw_assert (status >= 0,
                  "py_read_trees: unable to read population labels");

    
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
    status = cell::read_population_ranges(comm, string(file_name),
                                          pop_ranges, pop_vector,
                                          n_nodes);
    throw_assert(status >= 0,
                 "py_read_trees: unable to read population ranges");


    vector<neurotree_t> tree_vector;

    status = cell::read_trees (comm, string(file_name),
                               string(pop_name), pop_vector[pop_idx].start,
                               tree_vector);
    throw_assert (status >= 0,
                 "py_read_trees: unable to read trees");


    map <string, NamedAttrMap> attr_maps;
    
    for (string attr_namespace : attr_name_spaces)
      {
        data::NamedAttrMap attr_map;
        cell::read_cell_attributes(comm, string(file_name), 
                                   attr_namespace, attr_mask, pop_name,
                                   pop_vector[pop_idx].start, attr_map);
        attr_maps.insert(make_pair(attr_namespace, attr_map));
      }
    status = MPI_Comm_free(&comm);
    throw_assert(status == MPI_SUCCESS,
                 "py_read_trees: unable to free MPI communicator");


    PyObject* py_tree_iter = NeuroH5TreeIter_FromVector(tree_vector,
                                                        attr_name_spaces,
                                                        attr_maps,
                                                        topology_flag>0);

    PyObject *py_result_tuple = PyTuple_New(2);
    PyTuple_SetItem(py_result_tuple, 0, py_tree_iter);
    PyTuple_SetItem(py_result_tuple, 1, PyLong_FromLong((long)n_nodes));

    return py_result_tuple;
  }



  PyDoc_STRVAR(
    scatter_read_trees_doc,
    "scatter_read_trees(file_name, population_name, namespaces=[], topology=True, node_rank_map=None, comm=None, io_size=0)\n"
    "--\n"
    "\n"
    "Reads neuronal tree morphologies contained in the given file and scatters them to their assigned ranks. "
    "Each rank will be assigned morphologies according to a given rank to gid dictionary, or round robin if no user-specified assignment is given. \n"
    "\n"
    "Parameters\n"
    "----------\n"
    "file_name : string\n"
    "    The NeuroH5 file to read.\n"
    "    \n"
    "    .. warning::\n"
    "       The given file must be a valid HDF5 file that contains /H5Types and /Populations/Trees groups.\n"
    "\n"
    "population_name : string\n"
    "    Name of population from which to read.\n"
    "\n"
    "namespaces : string list\n"
    "    An optional list of namespaces from which additional attributes for the trees will be read.\n"
    "\n"
    "node_rank_map : { gid: rank }\n"
    "    An optional dictionary that specifies the mapping of cell gids to MPI ranks.\n"
    "\n"
    "topology : boolean\n"
    "    An optional flag that specifies whether section topology dictionary should be returned.\n"
    "\n"
    "comm : MPI communicator\n"
    "    Optional MPI communicator. If None, the world communicator will be used.\n"
    "\n"
    "io_size : integer\n"
    "    Optional number of I/O ranks, i.e. how many ranks should perform I/O operations. If 0, all ranks with perform I/O operations.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "(tree_iter, n_nodes) : tuple\n"
    "    A tuple with the following elements:  \n"
    "    - tree_iter: ( gid, tree_dict }\n"
    "       An iterator that returns pairs (gid, tree_dict) where gid is the cell id and tree_dict is a morphology dictionary with the following fields: \n"
    "          - x: X coordinates of tree morphology points (float ndarray)\n"
    "          - y: Y coordinates of tree morphology points (float ndarray)\n"
    "          - z: Z coordinates of tree morphology points (float ndarray)\n"
    "          - radius: radiuses of tree morphology points (float ndarray)\n"
    "          - layer: layer index of tree morphology points (-1 is undefined) (int8 ndarray)\n"
    "          - parent: parent index of tree morphology points (-1 is undefined) (int32 ndarray)\n"
    "          - swc_type: SWC type of tree morphology points (enumerated ndarray)\n"
    "          If topology is True :\n"
    "             - section: the section to which each point is assigned (uint16 ndarray)\n"
    "             - section_topology: a dictionary with the following fields: \n"
    "               - num_sections: number of sections in the morphology\n"
    "               - nodes: a dictionary that maps each section indices to the point indices contained in that section\n"
    "               - src: a vector of source section indices (uint16 ndarray)\n"
    "               - dst: a vector of destination section indices (uint16 ndarray)\n"
    "               - loc: a vector of source section connection indices (uint32 ndarray)\n"
    "          If topology is False :\n"
    "             - sections: for each section, the number of points, followed by the corresponding point indices (uint16 ndarray)\n"
    "             - src: a vector of source section indices (uint16 ndarray)\n"
    "             - dst: a vector of destination section indices (uint16 ndarray)\n"
    "    - n_nodes: int\n"
    "    - n_nodes: int\n"
    "       the total number of cells.\n"
    "\n");
  
  static PyObject *py_scatter_read_trees (PyObject *self, PyObject *args, PyObject *kwds)
  {
    int status; int topology_flag = 1;
    PyObject *py_comm = NULL;
    MPI_Comm *comm_ptr  = NULL;
    unsigned long io_size = 0;
    char *file_name, *pop_name;
    PyObject *py_node_rank_map=NULL;
    PyObject *py_attr_name_spaces=NULL;
    map<CELL_IDX_T, rank_t> node_rank_map;
    static const char *kwlist[] = {
                                   "file_name",
                                   "population_name",
                                   "comm",
                                   "node_rank_map",
                                   "namespaces",
                                   "topology",
                                   "io_size",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ss|OOOik", (char **)kwlist,
                                     &file_name, &pop_name, &py_comm, 
                                     &py_node_rank_map, &py_attr_name_spaces,
                                     &topology_flag, &io_size))
      return NULL;
    MPI_Comm comm;

    if (py_comm != NULL)
      {
        comm_ptr = PyMPIComm_Get(py_comm);
        throw_assert(comm_ptr != NULL,
                     "py_scatter_read_trees: invalid MPI communicator");
        throw_assert(*comm_ptr != MPI_COMM_NULL,
                     "py_scatter_read_trees: invalid MPI communicator");
        status = MPI_Comm_dup(*comm_ptr, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_scatter_read_trees: unable to duplicate MPI communicator");
      }
    else
      {
        status = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_scatter_read_trees: unable to duplicate MPI communicator");
      }

    int rank, size;
    status = MPI_Comm_size(comm, &size);
    throw_assert(status == MPI_SUCCESS,
                 "py_scatter_read_trees: unable to obtain size of MPI communicator");
    status = MPI_Comm_rank(comm, &rank);
    throw_assert(status == MPI_SUCCESS,
                 "py_scatter_read_trees: unable to obtain rank of MPI communicator");

    if (io_size == 0)
      {
        io_size = size;
      }
    
    vector <string> attr_name_spaces;
    map<CELL_IDX_T, pair<uint32_t,pop_t> > pop_ranges;
    vector<pop_range_t> pop_vector;
    size_t n_nodes;
    
    // Read population info
    status = cell::read_population_ranges(comm, string(file_name),
                                          pop_ranges, pop_vector,
                                          n_nodes);
    throw_assert (status >= 0,
                 "py_scatter_read_trees: unable to read population ranges");

                  
    // Create C++ vector of namespace strings:
    if (py_attr_name_spaces != NULL)
      {
        for (size_t i = 0; (Py_ssize_t)i < PyList_Size(py_attr_name_spaces); i++)
          {
            PyObject *pyval = PyList_GetItem(py_attr_name_spaces, (Py_ssize_t)i);
            const char *str = PyStr_ToCString (pyval);
            attr_name_spaces.push_back(string(str));
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
    status = cell::read_population_labels(comm, string(file_name), pop_labels);
    throw_assert (status >= 0,
                  "py_scatter_read_trees: unable to read population labels");

    
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

    status = cell::scatter_read_trees (comm, string(file_name),
                                       io_size, attr_name_spaces,
                                       node_rank_map, string(pop_name),
                                       pop_vector[pop_idx].start,
                                       tree_map, attr_maps);
    throw_assert (status >= 0,
                 "py_scatter_read_trees: unable to read trees");


    PyObject* py_tree_iter = NeuroH5TreeIter_FromMap(tree_map,
                                                     attr_name_spaces,
                                                     attr_maps,
                                                     topology_flag>0);

    PyObject *py_result_tuple = PyTuple_New(2);
    PyTuple_SetItem(py_result_tuple, 0, py_tree_iter);
    PyTuple_SetItem(py_result_tuple, 1, PyLong_FromLong((long)n_nodes));

    status = MPI_Comm_free(&comm);
    throw_assert(status == MPI_SUCCESS,
                 "py_scatter_read_trees: unable to free MPI communicator");

    
    return py_result_tuple;
  }
  

  PyDoc_STRVAR(
    read_tree_selection_doc,
    "read_tree_selection(file_name, population_name, selection, namespaces=[], topology=True, comm=None)\n"
    "--\n"
    "\n"
    "Reads selected neuronal tree morphologies contained in the given file. "
    "Each rank will be assigned an equal number of morphologies, with the exception of the last rank if the number of morphologies is not evenly divisible by the number of ranks. \n"
    "\n"
    "Parameters\n"
    "----------\n"
    "file_name : string\n"
    "    The NeuroH5 file to read.\n"
    "    \n"
    "    .. warning::\n"
    "       The given file must be a valid HDF5 file that contains /H5Types and /Populations/Trees groups.\n"
    "\n"
    "population_name : string\n"
    "    Name of population from which to read.\n"
    "\n"
    "selection : int list\n"
    "    A list of gids to read.\n"
    "\n"
    "namespaces : string list\n"
    "    An optional list of namespaces from which additional attributes for the trees will be read.\n"
    "\n"
    "topology : boolean\n"
    "    An optional flag that specifies whether section topology dictionary should be returned.\n"
    "\n"
    "comm : MPI communicator\n"
    "    Optional MPI communicator. If None, the world communicator will be used.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "(tree_iter, n_nodes) : tuple\n"
    "    A tuple with the following elements:  \n"
    "    - tree_iter: ( gid, tree_dict }\n"
    "       An iterator that returns pairs (gid, tree_dict) where gid is the cell id and tree_dict is a morphology dictionary with the following fields: \n"
    "          - x: X coordinates of tree morphology points (float ndarray)\n"
    "          - y: Y coordinates of tree morphology points (float ndarray)\n"
    "          - z: Z coordinates of tree morphology points (float ndarray)\n"
    "          - radius: radiuses of tree morphology points (float ndarray)\n"
    "          - layer: layer index of tree morphology points (-1 is undefined) (int8 ndarray)\n"
    "          - parent: parent index of tree morphology points (-1 is undefined) (int32 ndarray)\n"
    "          - swc_type: SWC type of tree morphology points (enumerated ndarray)\n"
    "          If topology is True :\n"
    "             - section: the section to which each point is assigned (uint16 ndarray)\n"
    "             - section_topology: a dictionary with the following fields: \n"
    "               - num_sections: number of sections in the morphology\n"
    "               - nodes: a dictionary that maps each section indices to the point indices contained in that section\n"
    "               - src: a vector of source section indices (uint16 ndarray)\n"
    "               - dst: a vector of destination section indices (uint16 ndarray)\n"
    "          If topology is False :\n"
    "             - sections: for each section, the number of points, followed by the corresponding point indices (uint16 ndarray)\n"
    "             - src: a vector of source section indices (uint16 ndarray)\n"
    "             - dst: a vector of destination section indices (uint16 ndarray)\n"
    "    - n_nodes: int\n"
    "    - n_nodes: int\n"
    "       the total number of cells.\n"
    "\n");

  static PyObject *py_read_tree_selection (PyObject *self, PyObject *args, PyObject *kwds)
  {
    int status; int topology_flag=1;
    PyObject *py_comm = NULL;
    PyObject *py_mask = NULL;
    MPI_Comm *comm_ptr  = NULL;
    char *file_name, *pop_name;
    PyObject *py_attr_name_spaces=NULL;
    PyObject *py_selection=NULL;
    vector <CELL_IDX_T> selection;

    static const char *kwlist[] = {
                                   "file_name",
                                   "pop_name",
                                   "selection",
                                   "comm",
                                   "mask",
                                   "namespaces",
                                   "topology",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ssO|OOOi", (char **)kwlist,
                                     &file_name, &pop_name,
                                     &py_selection, &py_comm, 
                                     &py_attr_name_spaces, &py_mask,
                                     &topology_flag))
      return NULL;
    throw_assert(PyList_Check(py_selection) > 0,
                 "py_read_tree_selection: unable to read tree selection");

    set<string> attr_mask;

    if (py_mask != NULL)
      {
        throw_assert(PySet_Check(py_mask),
                     "py_read_trees: argument mask must be a set of strings");
        
        PyObject *py_iter = PyObject_GetIter(py_mask);
        if (py_iter != NULL)
          {
            PyObject *pyval;
            while((pyval = PyIter_Next(py_iter)))
              {
                const char* str = PyStr_ToCString (pyval);
                attr_mask.insert(string(str));
                Py_DECREF(pyval);
              }
          }

        Py_DECREF(py_iter);
      }

    MPI_Comm comm;

    if (py_comm != NULL)
      {
        comm_ptr = PyMPIComm_Get(py_comm);
        throw_assert(comm_ptr != NULL,
                     "py_read_tree_selection: invalid MPI communicator");
        throw_assert(*comm_ptr != MPI_COMM_NULL,
                     "py_read_tree_selection: invalid MPI communicator");
        status = MPI_Comm_dup(*comm_ptr, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_read_tree_selection: unable to duplicate MPI communicator");
      }
    else
      {
        status = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_scatter_read_tree_selection: unable to duplicate MPI communicator");
      }

    vector <string> attr_name_spaces;
    // Create C++ vector of namespace strings:
    if (py_attr_name_spaces != NULL)
      {
        throw_assert(PyList_Check(py_attr_name_spaces) > 0,
                     "py_scatter_read_tree_selection: attribute name spaces argument is not a list");
        for (size_t i = 0; (Py_ssize_t)i < PyList_Size(py_attr_name_spaces); i++)
          {
            PyObject *pyval = PyList_GetItem(py_attr_name_spaces, (Py_ssize_t)i);
            const char *str = PyStr_ToCString (pyval);
            attr_name_spaces.push_back(string(str));
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
    status = cell::read_population_labels(comm, string(file_name), pop_labels);
    throw_assert (status >= 0,
                  "py_scatter_read_tree_selection: unable to read population labels");

    
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
    status = cell::read_population_ranges(comm, string(file_name),
                                          pop_ranges, pop_vector,
                                          n_nodes);
    throw_assert(status >= 0,
                 "py_read_tree_selection: unable to read population ranges");

    vector<neurotree_t> tree_vector;

    status = cell::read_tree_selection (comm, string(file_name),
                                        string(pop_name), pop_vector[pop_idx].start,
                                        tree_vector, selection);
    throw_assert (status >= 0,
                  "py_read_tree_selection: unable to read trees");

    map <string, NamedAttrMap> attr_maps;
    
    for (string attr_namespace : attr_name_spaces)
      {
        data::NamedAttrMap attr_map;
        cell::read_cell_attribute_selection(comm, string(file_name), 
                                            attr_namespace, attr_mask,
                                            pop_name, pop_vector[pop_idx].start,
                                            selection, attr_map);
        attr_maps.insert(make_pair(attr_namespace, attr_map));
      }
    throw_assert(MPI_Comm_free(&comm) == MPI_SUCCESS,
                 "py_read_tree_selection: unable to free MPI communicator");


    PyObject* py_tree_iter = NeuroH5TreeIter_FromVector(tree_vector,
                                                        attr_name_spaces,
                                                        attr_maps,
                                                        topology_flag>0);

    PyObject *py_result_tuple = PyTuple_New(2);
    PyTuple_SetItem(py_result_tuple, 0, py_tree_iter);
    PyTuple_SetItem(py_result_tuple, 1, PyLong_FromLong((long)n_nodes));

    return py_result_tuple;
  }

  PyDoc_STRVAR(
    scatter_read_cell_attributes_doc,
    "scatter_read_cell_attributes(file_name, population_name, namespaces, node_rank_map=None, comm=None, io_size=0)\n"
    "--\n"
    "\n"
    "Reads cell attributes for all cell gids contained in the given file and namespaces, using scalable parallel read/scatter."
    "Each rank will be assigned an equal number of cell gids, with the exception of the last rank if the number of cells is not evenly divisible by the number of ranks. \n"
    "\n"
    "Parameters\n"
    "----------\n"
    "file_name : string\n"
    "    The NeuroH5 file to read.\n"
    "    \n"
    "    .. warning::\n"
    "       The given file must be a valid HDF5 file that contains /H5Types and /Populations groups.\n"
    "\n"
    "population_name : string\n"
    "    Name of population from which to read.\n"
    "\n"
    "namespaces : string list\n"
    "    The namespaces for which cell attributes will be read.\n"
    "\n"
    "comm : MPI communicator\n"
    "    Optional MPI communicator. If None, the world communicator will be used.\n"
    "\n"
    "io_size : \n"
    "    Optional number of ranks performing I/O operations. If 0, this number will be equal to the size of the MPI communicator.\n"
    "\n"
    "node_rank_map : \n"
    "    Optional dictionary mapping gid to rank. If None, round-robin assignment will be used.\n"
    "\n"
    "mask : set of string\n"
    "    Optional set of attributes to be read. If not set, all attributes in the namespace will be read.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "Dictionary of the form { namespace: cell_iter }, where: \n"
    "cell_iter : iterator\n"
    "   An iterator that returns pairs (gid, attr_dict) where gid is the cell id and attr_dict is a dictionary with attribute key-value pairs. \n"
    "\n");

  static PyObject *py_scatter_read_cell_attributes (PyObject *self, PyObject *args, PyObject *kwds)
  {
    int status;
    PyObject *py_comm = NULL;
    PyObject *py_mask = NULL;
    MPI_Comm *comm_ptr  = NULL;
    unsigned long io_size = 0;
    char *file_name, *pop_name;
    PyObject *py_node_rank_map=NULL;
    PyObject *py_attr_name_spaces=NULL;
    map<CELL_IDX_T, rank_t> node_rank_map;
    vector <string> attr_name_spaces;
    int return_struct = 0;
    
    static const char *kwlist[] = {
                                   "file_name",
                                   "pop_name",
                                   "comm",
                                   "mask",
                                   "node_rank_map",
                                   "namespaces",
                                   "io_size",
                                   "return_struct",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ss|OOOOkp", (char **)kwlist,
                                     &file_name, &pop_name, &py_comm, &py_mask, 
                                     &py_node_rank_map, &py_attr_name_spaces,
                                     &io_size, &return_struct))
      return NULL;

    set<string> attr_mask;

    if (py_mask != NULL)
      {
        throw_assert(PySet_Check(py_mask),
                     "py_scatter_read_cell_attributes: argument mask must be a set of strings");
        
        PyObject *py_iter = PyObject_GetIter(py_mask);
        if (py_iter != NULL)
          {
            PyObject *pyval;
            while((pyval = PyIter_Next(py_iter)))
              {
                const char* str = PyStr_ToCString (pyval);
                attr_mask.insert(string(str));
                Py_DECREF(pyval);
              }
          }

        Py_DECREF(py_iter);
      }

    MPI_Comm comm;

    if (py_comm != NULL)
      {
        comm_ptr = PyMPIComm_Get(py_comm);
        throw_assert(comm_ptr != NULL,
                     "py_scatter_read_cell_attributes: invalid MPI communicator");
        throw_assert(*comm_ptr != MPI_COMM_NULL,
                     "py_scatter_read_cell_attributes: invalid MPI communicator");
        status = MPI_Comm_dup(*comm_ptr, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_scatter_read_cell_attributes: unable to duplicate MPI communicator");
      }
    else
      {
        status = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_scatter_read_cell_attributes: unable to duplicate MPI communicator");
      }

    int rank, size;
    status = MPI_Comm_size(comm, &size);
    throw_assert(status == MPI_SUCCESS,
                 "py_scatter_read_cell_attributes: unable to obtain size of MPI communicator");
    status = MPI_Comm_rank(comm, &rank);
    throw_assert(status == MPI_SUCCESS,
                 "py_scatter_read_cell_attributes: unable to obtain rank of MPI communicator");

    if (io_size == 0)
      {
        io_size = size;
      }

    // Create C++ vector of namespace strings:
    if (py_attr_name_spaces != NULL)
      {
        for (size_t i = 0; (Py_ssize_t)i < PyList_Size(py_attr_name_spaces); i++)
          {
            PyObject *pyval = PyList_GetItem(py_attr_name_spaces, (Py_ssize_t)i);
            const char *str = PyStr_ToCString (pyval);
            if (str != NULL)
              {
                attr_name_spaces.push_back(string(str));
              }
          }
      }

    
    size_t n_nodes;
    map<CELL_IDX_T, pair<uint32_t,pop_t> > pop_ranges;
    vector<pop_range_t> pop_vector;

    status = cell::read_population_ranges(comm,
                                          string(file_name),
                                          pop_ranges, pop_vector,
                                          n_nodes);
    throw_assert(status >= 0,
                 "py_scatter_read_cell_attributes: unable to read population ranges");


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
    status = cell::read_population_labels(comm, string(file_name), pop_labels);
    throw_assert (status >= 0,
                 "py_scatter_read_cell_attributes: unable to read population labels");

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

    PyObject *py_namespace_dict = PyDict_New();
    for (string attr_name_space : attr_name_spaces)
      {
        data::NamedAttrMap attr_map;
        
        status = cell::scatter_read_cell_attributes (comm,
                                                     string(file_name),
                                                     io_size,
                                                     attr_name_space,
                                                     attr_mask,
                                                     node_rank_map,
                                                     string(pop_name),
                                                     pop_vector[pop_idx].start,
                                                     attr_map);
        throw_assert (status >= 0,
                      "py_scatter_read_cell_attributes: unable to read cell attributes");
                      


        vector<vector<string>> attr_names;
        attr_map.attr_names(attr_names);

        PyObject *py_idx_iter = NeuroH5CellAttrIter_FromMap(attr_name_space,
                                                            attr_names,
                                                            attr_map,
                                                            return_struct>0);
        
        PyDict_SetItemString(py_namespace_dict, attr_name_space.c_str(), py_idx_iter);
        Py_DECREF(py_idx_iter);
      }
    status = MPI_Comm_free(&comm);
    throw_assert(status == MPI_SUCCESS,
                 "py_scatter_read_cell_attributes: unable to free MPI communicator");

    
    return py_namespace_dict;
  }

  PyDoc_STRVAR(
    read_cell_attributes_doc,
    "read_cell_attributes(file_name, population_name, namespace, comm=None)\n"
    "--\n"
    "\n"
    "Reads cell attributes for all cell gids contained in the given file and namespace. "
    "Each rank will be assigned an equal number of cell gids, with the exception of the last rank if the number of cells is not evenly divisible by the number of ranks. \n"
    "\n"
    "Parameters\n"
    "----------\n"
    "file_name : string\n"
    "    The NeuroH5 file to read.\n"
    "    \n"
    "    .. warning::\n"
    "       The given file must be a valid HDF5 file that contains /H5Types and /Populations groups.\n"
    "\n"
    "population_name : string\n"
    "    Name of population from which to read.\n"
    "\n"
    "namespace : string\n"
    "    The namespace for which cell attributes will be read.\n"
    "\n"
    "comm : MPI communicator\n"
    "    Optional MPI communicator. If None, the world communicator will be used.\n"
    "\n"
    "mask : set of string\n"
    "    Optional set of attributes to be read. If not set, all attributes in the namespace will be read.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "cell_iter : iterator\n"
    "   An iterator that returns pairs (gid, attr_dict) where gid is the cell id and attr_dict is a dictionary with attribute key-value pairs. \n"
    "\n");

  
  static PyObject *py_read_cell_attributes (PyObject *self, PyObject *args, PyObject *kwds)
  {
    herr_t status;
    PyObject *py_comm = NULL;
    MPI_Comm *comm_ptr  = NULL;
    PyObject *py_mask = NULL;
    const string default_namespace = "Attributes";
    char *file_name, *pop_name, *attr_namespace = (char *)default_namespace.c_str();
    int return_struct = 0;
    
    static const char *kwlist[] = {
                                   "file_name",
                                   "pop_name",
                                   "namespace",
                                   "comm",
                                   "mask",
                                   "return_struct",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ss|sOOp", (char **)kwlist,
                                     &file_name, &pop_name, &attr_namespace,
                                     &py_comm, &py_mask, &return_struct))
      return NULL;

    set<string> attr_mask;

    if (py_mask != NULL)
      {
        throw_assert(PySet_Check(py_mask),
                     "py_read_cell_attributes: argument mask must be a set of strings");
        
        PyObject *py_iter = PyObject_GetIter(py_mask);
        if (py_iter != NULL)
          {
            PyObject *pyval;
            while((pyval = PyIter_Next(py_iter)))
              {
                const char* str = PyStr_ToCString (pyval);
                attr_mask.insert(string(str));
                Py_DECREF(pyval);
              }
          }

        Py_DECREF(py_iter);
      }

    MPI_Comm comm;

    if (py_comm != NULL)
      {
        comm_ptr = PyMPIComm_Get(py_comm);
        throw_assert(comm_ptr != NULL,
                     "py_read_cell_attributes: unable to obtain MPI communicator");
        throw_assert(*comm_ptr != MPI_COMM_NULL,
                     "py_read_cell_attributes: MPI communicator is null");
        status = MPI_Comm_dup(*comm_ptr, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_read_cell_attributes: unable to duplicate MPI communicator");
      }
    else
      {
        status = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_read_cell_attributes: unable to duplicate MPI communicator");
      }

    vector<pair <pop_t, string> > pop_labels;
    status = cell::read_population_labels(comm, string(file_name), pop_labels);
    throw_assert (status >= 0,
                  "py_read_cell_attributes: unable to read population labels");
    
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
    throw_assert(cell::read_population_ranges(comm,
                                              string(file_name),
                                              pop_ranges, pop_vector,
                                              n_nodes) >= 0,
                 "py_read_cell_attributes: unable to read population ranges");


    NamedAttrMap attr_values;
    cell::read_cell_attributes (comm,
                                (file_name), string(attr_namespace), attr_mask,
                                string(pop_name), pop_vector[pop_idx].start,
                                attr_values);
    vector<vector<string>> attr_names;
    attr_values.attr_names(attr_names);
    
    throw_assert(MPI_Comm_free(&comm) == MPI_SUCCESS,
                 "py_read_cell_attributes: unable to free MPI communicator");

    PyObject *py_idx_iter = NeuroH5CellAttrIter_FromMap(attr_namespace,
                                                        attr_names,
                                                        attr_values,
                                                        return_struct>0);
        
    
    return py_idx_iter;
  }

  
  PyDoc_STRVAR(
    read_cell_attribute_selection_doc,
    "read_cell_attribute_selection(file_name, population_name, selection, namespace, comm=None)\n"
    "--\n"
    "\n"
    "Reads cell attributes for the given cell gids from the given file and namespace. "
    "Each rank will be assigned an equal number of cell gids, with the exception of the last rank if the number of cells is not evenly divisible by the number of ranks. \n"
    "\n"
    "Parameters\n"
    "----------\n"
    "file_name : string\n"
    "    The NeuroH5 file to read.\n"
    "    \n"
    "    .. warning::\n"
    "       The given file must be a valid HDF5 file that contains /H5Types and /Populations groups.\n"
    "\n"
    "population_name : string\n"
    "    Name of population from which to read.\n"
    "\n"
    "selection : int list\n"
    "    A list of gids to read.\n"
    "\n"
    "namespace : string\n"
    "    The namespace for which cell attributes will be read.\n"
    "\n"
    "comm : MPI communicator\n"
    "    Optional MPI communicator. If None, the world communicator will be used.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "cell_iter : iterator\n"
    "   An iterator that returns pairs (gid, attr_dict) where gid is the cell id and attr_dict is a dictionary with attribute key-value pairs. \n"
    "\n");

  static PyObject *py_read_cell_attribute_selection (PyObject *self, PyObject *args, PyObject *kwds)
  {
    herr_t status;
    PyObject *py_comm = NULL;
    PyObject *py_mask = NULL;
    MPI_Comm *comm_ptr  = NULL;
    const string default_namespace = "Attributes";
    char *file_name, *pop_name, *attr_namespace = (char *)default_namespace.c_str();
    PyObject *py_selection = NULL;
    vector <CELL_IDX_T> selection;
    int return_struct = 0;
    
    static const char *kwlist[] = {
                                   "file_name",
                                   "pop_name",
                                   "selection",
                                   "namespace",
                                   "comm",
                                   "mask",
                                   "return_struct",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ssO|sOOp", (char **)kwlist,
                                     &file_name, &pop_name, &py_selection,
                                     &attr_namespace, &py_comm, &py_mask,
                                     &return_struct))
      return NULL;

    set<string> attr_mask;

    if (py_mask != NULL)
      {
        throw_assert(PySet_Check(py_mask),
                     "py_read_trees: argument mask must be a set of strings");
        
        PyObject *py_iter = PyObject_GetIter(py_mask);
        if (py_iter != NULL)
          {
            PyObject *pyval;
            while((pyval = PyIter_Next(py_iter)))
              {
                const char* str = PyStr_ToCString (pyval);
                attr_mask.insert(string(str));
                Py_DECREF(pyval);
              }
          }

        Py_DECREF(py_iter);
      }

    
    MPI_Comm comm;

    if (py_comm != NULL)
      {
        comm_ptr = PyMPIComm_Get(py_comm);
        throw_assert(comm_ptr != NULL, 
                     "py_read_cell_attribute_selection: pointer to MPI communicator is null");
        throw_assert(*comm_ptr != MPI_COMM_NULL,
                     "py_read_cell_attribute_selection: MPI communicator is null");
        status = MPI_Comm_dup(*comm_ptr, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_read_cell_attribute_selection: unable to duplicate MPI communicator");
      }
    else
      {
        status = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
                     
        throw_assert(status == MPI_SUCCESS,
                     "py_read_cell_attribute_selection: unable to duplicate MPI communicator");
      }
    
    vector<pair <pop_t, string> > pop_labels;
    status = cell::read_population_labels(comm, string(file_name), pop_labels);
    throw_assert (status >= 0,
                  "py_read_cell_attribute_selection: unable to read population labels");

    
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
    throw_assert(cell::read_population_ranges(comm,
                                              string(file_name),
                                              pop_ranges, pop_vector,
                                              n_nodes) >= 0,
                 "py_read_cell_attribute_selection: unable to read population ranges");


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
    else
      {
        size_t population_n = pop_vector[pop_idx].count;
        size_t population_start = pop_vector[pop_idx].start;
        for (size_t i = 0; (Py_ssize_t)i < population_n; i++)
          {
            selection.push_back(i + population_start);
          }
        
      }

    NamedAttrMap attr_values;
    cell::read_cell_attribute_selection (comm, string(file_name), string(attr_namespace), attr_mask,
                                         string(pop_name), pop_vector[pop_idx].start,
                                         selection, attr_values);
    vector<vector<string>> attr_names;
    attr_values.attr_names(attr_names);
    throw_assert(MPI_Comm_free(&comm) == MPI_SUCCESS,
                 "py_read_cell_attribute_selection: unable to free MPI communicator");

    PyObject *py_idx_iter = NeuroH5CellAttrIter_FromMap(attr_namespace,
                                                        attr_names,
                                                        attr_values,
                                                        return_struct>0);
        
    
    return py_idx_iter;
  }

  
  static PyObject *py_bcast_cell_attributes (PyObject *self, PyObject *args, PyObject *kwds)
  {
    herr_t status;
    unsigned long root;
    PyObject *py_comm = NULL;
    PyObject *py_mask = NULL;
    MPI_Comm *comm_ptr  = NULL;
    const string default_namespace = "Attributes";
    char *file_name_arg, *pop_name_arg, *attr_namespace_arg = (char *)default_namespace.c_str();
    NamedAttrMap attr_values;
    int return_struct = 0;

    
    static const char *kwlist[] = {"file_name",
                                   "pop_name",
                                   "root",
                                   "namespace",
                                   "comm",
                                   "mask",
                                   "return_struct",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ssk|sOOp", (char **)kwlist,
                                     &file_name_arg, &pop_name_arg, &root, &attr_namespace_arg, 
                                     &py_comm, &py_mask, &return_struct))
      return NULL;

    string file_name = string(file_name_arg);
    string pop_name = string(pop_name_arg);
    string attr_namespace = string(attr_namespace_arg);

    set<string> attr_mask;

    if (py_mask != NULL)
      {
        throw_assert(PySet_Check(py_mask),
                     "py_bcast_cell_attributes: argument mask must be a set of strings");
        
        PyObject *py_iter = PyObject_GetIter(py_mask);
        if (py_iter != NULL)
          {
            PyObject *pyval;
            while((pyval = PyIter_Next(py_iter)))
              {
                const char* str = PyStr_ToCString (pyval);
                attr_mask.insert(string(str));
                Py_DECREF(pyval);
              }
          }

        Py_DECREF(py_iter);
      }

    MPI_Comm comm;

    if (py_comm != NULL)
      {
        comm_ptr = PyMPIComm_Get(py_comm);
        throw_assert(comm_ptr != NULL, 
                     "py_bcast_cell_attributes: pointer to MPI communicator is null");
        throw_assert(*comm_ptr != MPI_COMM_NULL,
                     "py_bcast_cell_attributes: MPI communicator is null");
        status = MPI_Comm_dup(*comm_ptr, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_bcast_cell_attributes: unable to duplicate MPI communicator");
      }
    else
      {
        status = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_bcast_cell_attributes: unable to duplicate MPI communicator");
      }

    int srank, ssize; size_t size, rank;
    throw_assert(MPI_Comm_size(comm, &ssize) == MPI_SUCCESS,
                 "py_bcast_cell_attributes: unable to obtain data communicator size");
    throw_assert(MPI_Comm_rank(comm, &srank) == MPI_SUCCESS,
                 "py_bcast_cell_attributes: unable to obtain data communicator rank");

    throw_assert(ssize > 0, "py_bcast_cell_attributes: zero data communicator size");
    throw_assert(srank >= 0, "py_bcast_cell_attributes: invalid data communicator rank");

    size = ssize;
    rank = srank;

    vector<pair <pop_t, string> > pop_labels;
    status = cell::read_population_labels(comm, string(file_name), pop_labels);
    throw_assert (status >= 0,
                  "py_bcast_cell_attributes: unable to read population labels");

    
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
    throw_assert(cell::read_population_ranges(comm,
                                              string(file_name),
                                              pop_ranges, pop_vector,
                                              n_nodes) >= 0,
                 "py_bcast_cell_attributes: unable to read population ranges");

    cell::bcast_cell_attributes (comm, (int)root,
                                 file_name, attr_namespace, attr_mask, 
                                 pop_name, pop_vector[pop_idx].start,
                                 attr_values);
    throw_assert(MPI_Comm_free(&comm) == MPI_SUCCESS,
                 "py_bcast_cell_attributes: unable to free MPI communicator");

    vector<vector<string>> attr_names;
    attr_values.attr_names(attr_names);

    PyObject *py_idx_iter = NeuroH5CellAttrIter_FromMap(attr_namespace,
                                                        attr_names,
                                                        attr_values,
                                                        return_struct>0);
        
    return py_idx_iter;
  }

  
  static PyObject *py_write_cell_attributes (PyObject *self, PyObject *args, PyObject *kwds)
  {
    PyObject *idx_values;
    PyObject *py_comm = NULL;
    MPI_Comm *comm_ptr  = NULL;
    const unsigned long default_cache_size = 4*1024*1024;
    const unsigned long default_chunk_size = 4000;
    const unsigned long default_value_chunk_size = 4000;
    const string default_namespace = "Attributes";
    char *file_name_arg, *pop_name_arg, *namespace_arg = (char *)default_namespace.c_str();
    unsigned long io_size = 0;
    unsigned long chunk_size = default_chunk_size;
    unsigned long value_chunk_size = default_value_chunk_size;
    unsigned long cache_size = default_cache_size;
    herr_t status;
    
    static const char *kwlist[] = {
                                   "file_name",
                                   "pop_name",
                                   "values",
                                   "namespace",
                                   "comm",
                                   "io_size",
                                   "chunk_size",
                                   "value_chunk_size",
                                   "cache_size",
                                   NULL};


    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ssO|sOkkkk", (char **)kwlist,
                                     &file_name_arg, &pop_name_arg, &idx_values,
                                     &namespace_arg, &py_comm, 
                                     &io_size, &chunk_size, &value_chunk_size, &cache_size))
      return NULL;

    string file_name = string(file_name_arg);
    string pop_name = string(pop_name_arg);
    string attr_namespace = string(namespace_arg);

    MPI_Comm comm;

    if (py_comm != NULL)
      {
        comm_ptr = PyMPIComm_Get(py_comm);
        throw_assert(comm_ptr != NULL, 
                     "py_write_cell_attributes: pointer to MPI communicator is null");
        throw_assert(*comm_ptr != MPI_COMM_NULL,
                     "py_write_cell_attributes: MPI communicator is null");
        status = MPI_Comm_dup(*comm_ptr, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_write_cell_attributes: unable to duplicate MPI communicator");
      }
    else
      {
        status = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_write_cell_attributes: unable to duplicate MPI communicator");
      }

    Py_ssize_t dict_size = PyDict_Size(idx_values);
    int data_color = 2;
    
    MPI_Comm data_comm;
    // In cases where some ranks do not have any data to write, split
    // the communicator, so that collective operations can be executed
    // only on the ranks that do have data.
    if (dict_size > 0)
      {
        MPI_Comm_split(comm,data_color,0,&data_comm);
      }
    else
      {
        MPI_Comm_split(comm,0,0,&data_comm);
      }
    MPI_Comm_set_errhandler(data_comm, MPI_ERRORS_RETURN);

    if (dict_size > 0)
      {
        int srank, ssize; size_t size;
        throw_assert(MPI_Comm_size(data_comm, &ssize) == MPI_SUCCESS,
                     "py_write_cell_attributes: unable to obtain data communicator size");
        throw_assert(MPI_Comm_rank(data_comm, &srank) == MPI_SUCCESS,
                     "py_write_cell_attributes: unable to obtain data communicator rank");
        throw_assert(ssize > 0, "py_write_cell_attributes: zero data communicator size");
        throw_assert(srank >= 0, "py_write_cell_attributes: invalid data communicator rank");
        size = ssize;

        if ((io_size == 0) || (io_size > size))
          {
            io_size = size;
          }
        throw_assert(io_size <= size,
                     "py_write_cell_attributes: invalid I/O size");

        vector<pair <pop_t, string> > pop_labels;
        status = cell::read_population_labels(data_comm, string(file_name), pop_labels);
        throw_assert (status >= 0,
                      "py_write_cell_attributes: unable to read population labels");
        
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
        throw_assert(cell::read_population_ranges(data_comm, string(file_name),
                                                  pop_ranges, pop_vector,
                                                  n_nodes) >= 0,
                     "py_write_cell_attributes: unable to read population ranges");

        CELL_IDX_T pop_start = pop_vector[pop_idx].start;
    
        int npy_type=0;
        
        map <string, map <CELL_IDX_T, vector< uint32_t >>> all_attr_values_uint32;
        map <string, map <CELL_IDX_T, vector< int32_t >>>  all_attr_values_int32;
        map <string, map <CELL_IDX_T, vector< uint16_t >>> all_attr_values_uint16;
        map <string, map <CELL_IDX_T, vector< int16_t >>>  all_attr_values_int16;
        map <string, map <CELL_IDX_T, vector< uint8_t >>>  all_attr_values_uint8;
        map <string, map <CELL_IDX_T, vector< int8_t >>>   all_attr_values_int8;
        map <string, map <CELL_IDX_T, vector< float >>>    all_attr_values_float;
        
        build_cell_attr_value_maps(idx_values,
                                   all_attr_values_uint32,
                                   all_attr_values_uint16,
                                   all_attr_values_uint8,
                                   all_attr_values_int32,
                                   all_attr_values_int16,
                                   all_attr_values_int8,
                                   all_attr_values_float);
        
        const data::optional_hid dflt_data_type;

        for(auto it = all_attr_values_float.cbegin(); it != all_attr_values_float.cend(); ++it)
          {
            const string& attr_name = it->first;
            cell::write_cell_attribute_map<float> (data_comm, file_name, attr_namespace, pop_name, pop_start,
                                                   attr_name, it->second, io_size, dflt_data_type);
          }
        for(auto it = all_attr_values_uint32.cbegin(); it != all_attr_values_uint32.cend(); ++it)
          {
            const string& attr_name = it->first;
            cell::write_cell_attribute_map<uint32_t> (data_comm, file_name, attr_namespace, pop_name, pop_start,
                                                      attr_name, it->second, io_size, dflt_data_type);
          }
        for(auto it = all_attr_values_uint16.cbegin(); it != all_attr_values_uint16.cend(); ++it)
          {
            const string& attr_name = it->first;
            cell::write_cell_attribute_map<uint16_t> (data_comm, file_name, attr_namespace, pop_name, pop_start,
                                                      attr_name, it->second, io_size, dflt_data_type);
          }
        for(auto it = all_attr_values_uint8.cbegin(); it != all_attr_values_uint8.cend(); ++it)
          {
            const string& attr_name = it->first;
            cell::write_cell_attribute_map<uint8_t> (data_comm, file_name, attr_namespace, pop_name, pop_start,
                                                     attr_name, it->second, io_size, dflt_data_type);
          }
        for(auto it = all_attr_values_int32.cbegin(); it != all_attr_values_int32.cend(); ++it)
          {
            const string& attr_name = it->first;
            cell::write_cell_attribute_map<int32_t> (data_comm, file_name, attr_namespace, pop_name, pop_start,
                                                     attr_name, it->second, io_size, dflt_data_type);
          }
        for(auto it = all_attr_values_int16.cbegin(); it != all_attr_values_int16.cend(); ++it)
          {
            const string& attr_name = it->first;
            cell::write_cell_attribute_map<int16_t> (data_comm, file_name, attr_namespace, pop_name, pop_start,
                                                     attr_name, it->second, io_size, dflt_data_type);
          }
        for(auto it = all_attr_values_int8.cbegin(); it != all_attr_values_int8.cend(); ++it)
          {
            const string& attr_name = it->first;
            cell::write_cell_attribute_map<int8_t> (data_comm, file_name, attr_namespace, pop_name, pop_start,
                                                    attr_name, it->second, io_size, dflt_data_type);
          }

        
      }
    
    //throw_assert(MPI_Barrier(data_comm) == MPI_SUCCESS,
    //             "py_write_cell_attributes: error in MPI barrier on data communicator");
    throw_assert(MPI_Comm_free(&data_comm) == MPI_SUCCESS,
                 "py_write_cell_attributes: unable to free data MPI communicator");
    
    throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                 "py_write_cell_attributes: error in MPI barrier");
    throw_assert(MPI_Comm_free(&comm) == MPI_SUCCESS,
                 "py_write_cell_attributes: unable to free MPI communicator");
    
    Py_INCREF(Py_None);
    return Py_None;
  }
  
  
  static PyObject *py_append_cell_attributes (PyObject *self, PyObject *args, PyObject *kwds)
  {
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
    
    static const char *kwlist[] = {
                                   "file_name",
                                   "pop_name",
                                   "values",
                                   "namespace",
                                   "comm",
                                   "io_size",
                                   "chunk_size",
                                   "value_chunk_size",
                                   "cache_size",
                                   NULL};


    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ssO|sOkkkk", (char **)kwlist,
                                     &file_name_arg, &pop_name_arg, &idx_values,
                                     &namespace_arg, &py_comm, 
                                     &io_size, &chunk_size, &value_chunk_size, &cache_size))
      return NULL;
    MPI_Comm comm;

    if (py_comm != NULL)
      {
        comm_ptr = PyMPIComm_Get(py_comm);
        throw_assert(comm_ptr != NULL, 
                     "py_append_cell_attributes: pointer to MPI communicator is null");
        throw_assert(*comm_ptr != MPI_COMM_NULL,
                     "py_append_cell_attributes: MPI communicator is null");
        status = MPI_Comm_dup(*comm_ptr, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_append_cell_attributes: unable to duplicate MPI communicator");
      }
    else
      {
        status = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_append_cell_attributes: unable to duplicate MPI communicator");
      }

    Py_ssize_t dict_size = PyDict_Size(idx_values);
    int data_color = 2;


    MPI_Comm data_comm;
    // In cases where some ranks do not have any data to write, split
    // the communicator, so that collective operations can be executed
    // only on the ranks that do have data.
    if (dict_size > 0)
      {
        MPI_Comm_split(comm,data_color,0,&data_comm);
      }
    else
      {
        MPI_Comm_split(comm,0,0,&data_comm);
      }
    MPI_Comm_set_errhandler(data_comm, MPI_ERRORS_RETURN);

    if (dict_size > 0)
      {
        int srank, ssize; size_t size;
        throw_assert(MPI_Comm_size(data_comm, &ssize) == MPI_SUCCESS,
                     "py_append_cell_attributes: unable to obtain data communicator size");
        throw_assert(MPI_Comm_rank(data_comm, &srank) == MPI_SUCCESS,
                     "py_append_cell_attributes: unable to obtain data communicator rank");
        throw_assert(ssize > 0, "py_append_cell_attributes: zero data communicator size");
        throw_assert(srank >= 0, "py_append_cell_attributes: invalid data communicator rank");
        size = ssize;
    
        if ((io_size == 0) || (io_size > size))
          {
            io_size = size;
          }
        throw_assert(io_size <= size,
                     "py_append_cell_attributes: invalid I/O size");

        string file_name      = string(file_name_arg);
        string pop_name       = string(pop_name_arg);
        string attr_namespace = string(namespace_arg);

        vector<pair <pop_t, string> > pop_labels;
        status = cell::read_population_labels(data_comm, string(file_name), pop_labels);
        throw_assert (status >= 0,
                      "py_append_cell_attributes: unable to read population labels");
    
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
        throw_assert(cell::read_population_ranges(data_comm, string(file_name),
                                                  pop_ranges, pop_vector,
                                                  n_nodes) >= 0,
                     "py_append_cell_attributes: unable to read population ranges");                     

        CELL_IDX_T pop_start = pop_vector[pop_idx].start;
    
    
        int npy_type=0;
    
        vector<string> attr_names;
        vector<int> attr_types;
        
        map<string, map<CELL_IDX_T, vector<uint32_t> >> all_attr_values_uint32;
        map<string, map<CELL_IDX_T, vector<int32_t> >> all_attr_values_int32;
        map<string, map<CELL_IDX_T, vector<uint16_t> >> all_attr_values_uint16;
        map<string, map<CELL_IDX_T, vector<int16_t> >> all_attr_values_int16;
        map<string, map<CELL_IDX_T, vector<uint8_t> >>  all_attr_values_uint8;
        map<string, map<CELL_IDX_T, vector<int8_t> >>  all_attr_values_int8;
        map<string, map<CELL_IDX_T, vector<float> >>  all_attr_values_float;

        build_cell_attr_value_maps(idx_values,
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
        throw_assert(status == 0,
                     "py_append_cell_attributes: unable to create toplevel groups in file");

        MPI_Barrier(data_comm);

        const data::optional_hid dflt_data_type;
        
        for(auto it = all_attr_values_float.cbegin(); it != all_attr_values_float.cend(); ++it)
          {
            const string& attr_name = it->first;
            cell::append_cell_attribute_map<float> (data_comm, file_name, attr_namespace, pop_name, pop_start,
                                                    attr_name, it->second, io_size, dflt_data_type);
          }
        for(auto it = all_attr_values_uint32.cbegin(); it != all_attr_values_uint32.cend(); ++it)
          {
            const string& attr_name = it->first;
            cell::append_cell_attribute_map<uint32_t> (data_comm, file_name, attr_namespace, pop_name, pop_start,
                                                       attr_name, it->second, io_size, dflt_data_type);
          }
        for(auto it = all_attr_values_uint16.cbegin(); it != all_attr_values_uint16.cend(); ++it)
          {
            const string& attr_name = it->first;
            cell::append_cell_attribute_map<uint16_t> (data_comm, file_name, attr_namespace, pop_name, pop_start,
                                                      attr_name, it->second, io_size, dflt_data_type);
          }
        for(auto it = all_attr_values_uint8.cbegin(); it != all_attr_values_uint8.cend(); ++it)
          {
            const string& attr_name = it->first;
            cell::append_cell_attribute_map<uint8_t> (data_comm, file_name, attr_namespace, pop_name, pop_start,
                                                      attr_name, it->second, io_size, dflt_data_type);
          }
        for(auto it = all_attr_values_int32.cbegin(); it != all_attr_values_int32.cend(); ++it)
          {
            const string& attr_name = it->first;
            cell::append_cell_attribute_map<int32_t> (data_comm, file_name, attr_namespace, pop_name, pop_start,
                                                      attr_name, it->second, io_size, dflt_data_type);
          }
        for(auto it = all_attr_values_int16.cbegin(); it != all_attr_values_int16.cend(); ++it)
          {
            const string& attr_name = it->first;
            cell::append_cell_attribute_map<int16_t> (data_comm, file_name, attr_namespace, pop_name, pop_start,
                                                      attr_name, it->second, io_size, dflt_data_type);
          }
        for(auto it = all_attr_values_int8.cbegin(); it != all_attr_values_int8.cend(); ++it)
          {
            const string& attr_name = it->first;
            cell::append_cell_attribute_map<int8_t> (data_comm, file_name, attr_namespace, pop_name, pop_start,
                                                     attr_name, it->second, io_size, dflt_data_type);
          }


      }
    
    throw_assert(MPI_Barrier(data_comm) == MPI_SUCCESS,
                 "py_append_cell_attributes: error in MPI barrier on data communicator");
    throw_assert(MPI_Barrier(comm) == MPI_SUCCESS,
                 "py_append_cell_attributes: error in MPI barrier");

    throw_assert(MPI_Comm_free(&data_comm) == MPI_SUCCESS,
                 "py_append_cell_attributes: unable to free data MPI communicator");
    throw_assert(MPI_Comm_free(&comm) == MPI_SUCCESS,
                 "py_append_cell_attributes: unable to free MPI communicator");
    
    Py_INCREF(Py_None);
    return Py_None;
  }


  static PyObject *py_append_cell_trees (PyObject *self, PyObject *args, PyObject *kwds)
  {
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
    herr_t status;
    
    static const char *kwlist[] = {
                                   "file_name",
                                   "pop_name",
                                   "values",
                                   "create_index",
                                   "comm",
                                   "io_size",
                                   "chunk_size",
                                   "value_chunk_size",
                                   "cache_size",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ssO|kOkkkkk", (char **)kwlist,
                                     &file_name_arg, &pop_name_arg, &idx_values,
                                     &create_index, &py_comm, &io_size,
                                     &chunk_size, &value_chunk_size, &cache_size))
      return NULL;

    MPI_Comm comm;

    if (py_comm != NULL)
      {
        comm_ptr = PyMPIComm_Get(py_comm);
        throw_assert(comm_ptr != NULL, 
                     "py_append_cell_trees: pointer to MPI communicator is null");
        throw_assert(*comm_ptr != MPI_COMM_NULL,
                     "py_append_cell_trees: MPI communicator is null");
        status = MPI_Comm_dup(*comm_ptr, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_append_cell_trees: unable to duplicate MPI communicator");
      }
    else
      {
        status = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        throw_assert(status == MPI_SUCCESS,
                     "py_append_cell_trees: unable to duplicate MPI communicator");
      }

    Py_ssize_t dict_size = PyDict_Size(idx_values);
    int data_color = 2;


    MPI_Comm data_comm;
    // In cases where some ranks do not have any data to write, split
    // the communicator, so that collective operations can be executed
    // only on the ranks that do have data.
    if (dict_size > 0)
      {
        MPI_Comm_split(comm,data_color,0,&data_comm);
      }
    else
      {
        MPI_Comm_split(comm,0,0,&data_comm);
      }
    MPI_Comm_set_errhandler(data_comm, MPI_ERRORS_RETURN);

    
    int srank, ssize; size_t size;
    throw_assert(MPI_Comm_size(data_comm, &ssize) == MPI_SUCCESS,
                 "py_append_cell_trees: unable to obtain data communicator size");
    throw_assert(MPI_Comm_rank(data_comm, &srank) == MPI_SUCCESS,
                 "py_append_cell_trees: unable to obtain data communicator rank");
    throw_assert(ssize > 0, "py_append_cell_trees: zero data communicator size");
    throw_assert(srank >= 0, "py_append_cell_trees: invalid data communicator rank");

    size = ssize;
    
    if ((io_size == 0) || (io_size > size))
      {
        io_size = size;
      }
    throw_assert(io_size <= size,
                 "py_append_cell_trees: invalid I/O size");
    
    string file_name      = string(file_name_arg);
    string pop_name       = string(pop_name_arg);

    vector<pair <pop_t, string> > pop_labels;
    status = cell::read_population_labels(data_comm, string(file_name), pop_labels);
    throw_assert (status >= 0,
                  "py_append_cell_trees: unable to read population labels");
    
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
    throw_assert(cell::read_population_ranges(data_comm, string(file_name),
                                              pop_ranges, pop_vector,
                                              n_nodes) >= 0,
                 "py_append_cell_trees: unable to read population ranges");
    
    map<string, map<CELL_IDX_T, vector<uint32_t> >> all_attr_values_uint32;
    map<string, map<CELL_IDX_T, vector<int32_t> >> all_attr_values_int32;
    map<string, map<CELL_IDX_T, vector<uint16_t> >> all_attr_values_uint16;
    map<string, map<CELL_IDX_T, vector<int16_t> >> all_attr_values_int16;
    map<string, map<CELL_IDX_T, vector<uint8_t> >>  all_attr_values_uint8;
    map<string, map<CELL_IDX_T, vector<int8_t> >>  all_attr_values_int8;
    map<string, map<CELL_IDX_T, vector<float> >>  all_attr_values_float;
    
    build_cell_attr_value_maps(idx_values,
                               all_attr_values_uint32,
                               all_attr_values_uint16,
                               all_attr_values_uint8,
                               all_attr_values_int32,
                               all_attr_values_int16,
                               all_attr_values_int8,
                               all_attr_values_float);

    auto xcoord_map_it = all_attr_values_float.find("x");
    throw_assert(xcoord_map_it != all_attr_values_float.end(),
                 "py_append_cell_trees: input data has no x array");
    auto ycoord_map_it = all_attr_values_float.find("y");
    throw_assert(ycoord_map_it != all_attr_values_float.end(),
                 "py_append_cell_trees: input data has no y array");
    auto zcoord_map_it = all_attr_values_float.find("z");
    throw_assert(zcoord_map_it != all_attr_values_float.end(),
                 "py_append_cell_trees: input data has no z array");
    auto radius_map_it = all_attr_values_float.find("radius");
    throw_assert(radius_map_it != all_attr_values_float.end(),
                 "py_append_cell_trees: input data has no radius array");
    auto layer_map_it = all_attr_values_int8.find("layer");
    throw_assert(layer_map_it != all_attr_values_int8.end(),
                 "py_append_cell_trees: input data has no layer array");
    auto parent_map_it = all_attr_values_int32.find("parent");
    throw_assert(parent_map_it != all_attr_values_int32.end(),
                 "py_append_cell_trees: input data has no parent array");
    auto swc_type_map_it = all_attr_values_int8.find("swc_type");
    throw_assert(swc_type_map_it != all_attr_values_int8.end(),
                 "py_append_cell_trees: input data has no int8 array");
    auto sections_map_it = all_attr_values_uint16.find("sections");
    throw_assert(sections_map_it != all_attr_values_uint16.end(),
                 "py_append_cell_trees: input data has no sections array");
    auto src_map_it = all_attr_values_uint16.find("src");
    throw_assert(src_map_it != all_attr_values_uint16.end(),
                 "py_append_cell_trees: input data has no src array");
    auto dst_map_it = all_attr_values_uint16.find("dst");
    throw_assert(dst_map_it != all_attr_values_uint16.end(),
                 "py_append_cell_trees: input data has no dst array");
    
    vector<neurotree_t> tree_vector;

    map<CELL_IDX_T, vector<float> >& xcoord_values = xcoord_map_it->second;
    map<CELL_IDX_T, vector<float> >& ycoord_values = ycoord_map_it->second;
    map<CELL_IDX_T, vector<float> >& zcoord_values = zcoord_map_it->second;
    map<CELL_IDX_T, vector<float> >& radius_values = radius_map_it->second;
    map<CELL_IDX_T, vector<int32_t> >& parent_values = parent_map_it->second;
    map<CELL_IDX_T, vector<uint16_t> >& src_values = src_map_it->second;
    map<CELL_IDX_T, vector<uint16_t> >& dst_values = dst_map_it->second;
    map<CELL_IDX_T, vector<uint16_t> >& sections_values = sections_map_it->second;
    map<CELL_IDX_T, vector<int8_t> >& layer_values = layer_map_it->second;
    map<CELL_IDX_T, vector<int8_t> >& swc_type_values = swc_type_map_it->second;


    auto sections_it  = sections_values.begin();
    auto src_it  = src_values.begin();
    auto dst_it  = dst_values.begin();

    auto parent_it  = parent_values.begin();
    
    auto xcoord_it  = xcoord_values.begin();
    auto ycoord_it  = ycoord_values.begin();
    auto zcoord_it  = zcoord_values.begin();
    auto radius_it  = radius_values.begin();

    auto layer_it   = layer_values.begin();
    auto swc_type_it  = swc_type_values.begin();

    while (sections_it != sections_values.end())
      {
        CELL_IDX_T id = sections_it->first;
        tree_vector.push_back(make_tuple(id,
                                         src_it->second,
                                         dst_it->second,
                                         sections_it->second,
                                         xcoord_it->second,
                                         ycoord_it->second,
                                         zcoord_it->second,
                                         radius_it->second,
                                         layer_it->second,
                                         parent_it->second,
                                         swc_type_it->second));
        
        ++sections_it, ++src_it, ++dst_it, ++parent_it,
          ++xcoord_it, ++ycoord_it, ++zcoord_it, ++radius_it,
          ++layer_it, ++swc_type_it;
      }
    
    CELL_IDX_T pop_start = pop_vector[pop_idx].start;

    throw_assert(cell::append_trees (data_comm, file_name, pop_name, pop_start, tree_vector) >= 0,
                 "py_append_cell_trees: unable to append trees");
    throw_assert(MPI_Barrier(data_comm) == MPI_SUCCESS,
                 "py_append_cell_trees: MPI barrier error");
    
    throw_assert(MPI_Comm_free(&data_comm) == MPI_SUCCESS,
                 "py_append_cell_trees: unable to free data MPI communicator");
    throw_assert(MPI_Comm_free(&comm) == MPI_SUCCESS,
                 "py_append_cell_trees: unable to free MPI communicator");
    
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
   * seq_index: index of the next edge in the sequence to yield
   * start_index: starting index of the next batch of edges to read from file
   * cache_size: how many edge blocks to read from file at at time
   *
   */
  typedef struct {
    Py_ssize_t node_index, node_count, block_index, block_count, cache_index, cache_size, io_size, comm_size;

    string file_name;
    MPI_Comm comm;

    seq_pos pos;
    EdgeMapType edge_map_type;
    vector<pop_range_t> pop_vector;
    map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
    set< pair<pop_t, pop_t> > pop_pairs;
    vector<pair <pop_t, string> > pop_labels;
    vector<pair<string,string> > prj_names;
    edge_map_t edge_map;
    edge_map_iter_t edge_map_iter;
    map <string, vector< vector<string> > > edge_attr_names;
    vector<string> edge_attr_name_spaces;
    string src_pop_name, dst_pop_name;
    size_t total_num_nodes, local_num_nodes, total_num_edges, local_num_edges;
    hsize_t total_read_blocks, local_read_blocks;
    NODE_IDX_T dst_start, src_start;

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
    Py_ssize_t seq_index, cache_index, cache_size,
      io_size, comm_size, local_count, max_local_count, count;

    seq_pos pos;
    string pop_name;
    size_t pop_idx;
    string file_name;
    MPI_Comm comm;
    vector<pop_range_t> pop_vector;
    map<CELL_IDX_T, neurotree_t> tree_map;
    vector<string> attr_name_spaces;
    map <string, NamedAttrMap> attr_maps;
    map <string, vector< vector <string> > > attr_names;
    map<CELL_IDX_T, neurotree_t>::const_iterator it_tree;
    map<CELL_IDX_T, rank_t> node_rank_map;
    bool topology_flag;
    
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
   * seq_index: index of the next id in the sequence to yield
   * start_index: starting index of the next batch of trees to read from file
   * cache_size: how many trees to read from file at at time
   *
   */
  typedef struct {
    Py_ssize_t seq_index, cache_index, cache_size,
      io_size, comm_size, local_count, max_local_count, count;
     
    seq_pos pos;
    string pop_name;
    size_t pop_idx;
    string file_name;
    string att_namespace;
    MPI_Comm comm;
    vector<pop_range_t> pop_vector;
    string attr_namespace;
    set <string> attr_mask;
    NamedAttrMap attr_map;
    vector< vector <string> > attr_names;
    set<CELL_IDX_T>::const_iterator it_idx;
    map <CELL_IDX_T, rank_t> node_rank_map;
    PyTypeObject* struct_type;
    vector<PyStructSequence_Field> struct_descr_fields;
    bool return_struct;
    
  } NeuroH5CellAttrGenState;
  
  typedef struct {
    PyObject_HEAD
    NeuroH5CellAttrGenState *state;
  } PyNeuroH5CellAttrGenState;

  
  static PyObject *
  neuroh5_prj_gen_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
  {
    int status;
    int opt_edge_map_type=0;
    EdgeMapType edge_map_type = EdgeMapDst;
    PyObject *py_comm = NULL;
    MPI_Comm *comm_ptr  = NULL;
    unsigned int cache_size=1;
    char *file_name, *src_pop_name, *dst_pop_name;
    PyObject* py_attr_name_spaces = NULL;
    vector<pop_range_t> pop_vector;
    map<NODE_IDX_T,pair<uint32_t,pop_t> > pop_ranges;
    set< pair<pop_t, pop_t> > pop_pairs;
    vector<pair <pop_t, string> > pop_labels;
    vector<pair<string,string> > prj_names;
    size_t total_num_nodes;

    static const char *kwlist[] = {
                                   "file_name",
                                   "src_pop_name",
                                   "dst_pop_name",
                                   "namespaces",
                                   "edge_map_type",
                                   "comm",
                                   "cache_size",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sss|OiOi", (char **)kwlist,
                                     &file_name, &src_pop_name, &dst_pop_name, 
                                     &py_attr_name_spaces, &opt_edge_map_type,
                                     &py_comm, &cache_size))
      return NULL;

    MPI_Comm comm;

    if (py_comm != NULL)
      {
        comm_ptr = PyMPIComm_Get(py_comm);
        throw_assert(comm_ptr != NULL, "NeuroH5ProjectionGen: invalid MPI communicator");
        throw_assert(*comm_ptr != MPI_COMM_NULL, "NeuroH5ProjectionGen: null MPI communicator");
        status = MPI_Comm_dup(*comm_ptr, &comm);
        throw_assert(status == MPI_SUCCESS, "NeuroH5ProjectionGen: unable to duplicate MPI communicator");
      }
    else
      {
        status = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        throw_assert(status == MPI_SUCCESS, "NeuroH5ProjectionGen: unable to duplicate MPI communicator");
      }

    
    if (opt_edge_map_type == 1)
      {
        edge_map_type = EdgeMapSrc;
      }
    
    int size, rank;
    throw_assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS,
                 "NeuroH5ProjectionGen: unable to obtain MPI communicator size");
    throw_assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS,
                 "NeuroH5ProjectionGen: unable to obtain MPI communicator rank");


    if (cache_size <= 0)
      {
        cache_size = 1;
      }

    vector <string> attr_name_spaces;
    // Create C++ vector of namespace strings:
    if (py_attr_name_spaces != NULL)
      {
        for (size_t i = 0; (Py_ssize_t)i < PyList_Size(py_attr_name_spaces); i++)
          {
            PyObject *pyval = PyList_GetItem(py_attr_name_spaces, (Py_ssize_t)i);
            const char *str = PyStr_ToCString (pyval);
            attr_name_spaces.push_back(string(str));
          }
      }
    
    throw_assert(graph::read_projection_names(comm, string(file_name), prj_names) >= 0,
                 "NeuroH5ProjectionGen: unable to read projection names");

    // Read population info to determine total_num_nodes
    throw_assert(cell::read_population_ranges(comm, string(file_name),
                                              pop_ranges, pop_vector, total_num_nodes) >= 0,
                 "NeuroH5ProjectionGen: unable to read population ranges");                 
    throw_assert(cell::read_population_labels(comm, file_name, pop_labels) >= 0,
                 "NeuroH5ProjectionGen: unable to read population labels");
    throw_assert(cell::read_population_combos(comm, string(file_name), pop_pairs) >= 0,
                 "NeuroH5ProjectionGen: unable to read projection combinations");
    
    hsize_t num_blocks = hdf5::num_projection_blocks(comm, string(file_name),
                                                     src_pop_name, dst_pop_name);


    /* Create a new generator state and initialize it */
    PyNeuroH5ProjectionGenState *py_ngg = (PyNeuroH5ProjectionGenState *)type->tp_alloc(type, 0);
    if (!py_ngg) return NULL;
    py_ngg->state = new NeuroH5ProjectionGenState();

    throw_assert(MPI_Comm_dup(comm, &(py_ngg->state->comm)) == MPI_SUCCESS, 
                 "NeuroH5ProjectionGen: unable to duplicate MPI communicator");
    throw_assert(MPI_Comm_free(&comm) == MPI_SUCCESS,
                 "NeuroH5ProjectionGen: unable to free MPI communicator");

    py_ngg->state->pos             = seq_next;
    py_ngg->state->node_index      = 0;
    py_ngg->state->node_count      = 0;
    py_ngg->state->block_index     = 0;
    py_ngg->state->block_count     = num_blocks;
    py_ngg->state->cache_size      = cache_size;
    py_ngg->state->file_name       = string(file_name);
    py_ngg->state->src_pop_name    = string(src_pop_name);
    py_ngg->state->dst_pop_name    = string(dst_pop_name);
    py_ngg->state->pop_vector      = pop_vector;
    py_ngg->state->pop_ranges      = pop_ranges;
    py_ngg->state->pop_pairs       = pop_pairs;
    py_ngg->state->pop_labels      = pop_labels;
    py_ngg->state->edge_map_iter   = py_ngg->state->edge_map.cbegin();
    py_ngg->state->edge_map_type   = edge_map_type;
    py_ngg->state->edge_attr_name_spaces = attr_name_spaces;
    py_ngg->state->total_num_nodes = total_num_nodes;
    py_ngg->state->local_num_nodes = 0;
    py_ngg->state->total_num_edges = 0;
    py_ngg->state->local_num_edges = 0;
    py_ngg->state->total_read_blocks = 0;
    py_ngg->state->local_read_blocks = 0;

    uint32_t dst_pop_idx = 0, src_pop_idx = 0;
    bool src_pop_set = false, dst_pop_set = false;
    
    for (size_t i=0; i< pop_labels.size(); i++)
      {
        if (string(src_pop_name) == get<1>(pop_labels[i]))
          {
            src_pop_idx = get<0>(pop_labels[i]);
            src_pop_set = true;
          }
        if (string(dst_pop_name) == get<1>(pop_labels[i]))
          {
            dst_pop_idx = get<0>(pop_labels[i]);
            dst_pop_set = true;
          }
      }
    throw_assert(dst_pop_set && src_pop_set, 
                 "NeuroH5ProjectionGen: unable to determine source and destination population");
    
    NODE_IDX_T dst_start = pop_vector[dst_pop_idx].start;
    NODE_IDX_T src_start = pop_vector[src_pop_idx].start;

    py_ngg->state->dst_start = dst_start;
    py_ngg->state->src_start = src_start;
    
    return (PyObject *)py_ngg;
    
  }
  
  static PyObject *
  neuroh5_tree_gen_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
  {
    int status;
    int topology_flag=1;
    PyObject *py_comm = NULL;
    MPI_Comm *comm_ptr  = NULL;
    unsigned int io_size, cache_size=100;
    char *file_name, *pop_name;
    PyObject* py_attr_name_spaces = NULL;
    vector<string> attr_name_spaces;

    static const char *kwlist[] = {
                                   "file_name",
                                   "pop_name",
                                   "namespaces",
                                   "topology",
                                   "comm",
                                   "io_size",
                                   "cache_size",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ss|OiOii", (char **)kwlist,
                                     &file_name, &pop_name, 
                                     &py_attr_name_spaces, &topology_flag,
                                     &py_comm, &io_size, &cache_size))
      return NULL;

    MPI_Comm comm;

    if (py_comm != NULL)
      {
        comm_ptr = PyMPIComm_Get(py_comm);
        throw_assert(comm_ptr != NULL, "NeuroH5TreeGen: invalid MPI communicator");
        throw_assert(*comm_ptr != MPI_COMM_NULL, "NeuroH5TreeGen: null MPI communicator");
        status = MPI_Comm_dup(*comm_ptr, &comm);
        throw_assert(status == MPI_SUCCESS, "NeuroH5TreeGen: unable to duplicate MPI communicator");
      }
    else
      {
        status = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        throw_assert(status == MPI_SUCCESS, "NeuroH5TreeGen: unable to duplicate MPI communicator");
      }

    int rank, size;
    throw_assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS,
                 "NeuroH5TreeGen: unable to obtain MPI communicator size");
    throw_assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS,
                 "NeuroH5TreeGen: unable to obtain MPI communicator rank");
    
    if ((size > 0) && (io_size > (unsigned int)size))
      io_size = size;
    
    if ((size > 0) && (cache_size > (unsigned int)size))
      cache_size = size;

    // Create C++ vector of namespace strings:
    if (py_attr_name_spaces != NULL)
      {
        for (size_t i = 0; (Py_ssize_t)i < PyList_Size(py_attr_name_spaces); i++)
          {
            PyObject *pyval = PyList_GetItem(py_attr_name_spaces, (Py_ssize_t)i);
            const char *str = PyStr_ToCString (pyval);
            attr_name_spaces.push_back(string(str));
          }
      }
    
    vector<pair <pop_t, string> > pop_labels;
    status = cell::read_population_labels(comm, string(file_name), pop_labels);
    throw_assert (status >= 0,
                  "NeuroH5TreeGen: unable to read population labels");
    
    
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
    throw_assert(cell::read_population_ranges(comm,
                                              string(file_name),
                                              pop_ranges, pop_vector,
                                              n_nodes) >= 0,
                 "NeuroH5TreeGen: unable to read population ranges");
    
    vector<CELL_IDX_T> tree_index;
    throw_assert(cell::read_cell_index(comm,
                                       string(file_name),
                                       get<1>(pop_labels[pop_idx]),
                                       hdf5::TREES,
                                       tree_index) >= 0,
                 "NeuroH5TreeGen: unable to read cell index");
    
    size_t count = tree_index.size();
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
    rank_t r=0; size_t local_count=0; 
    for (size_t i = 0; i < tree_index.size(); i++)
      {
        if ((unsigned int)rank == r) local_count++;
        auto it = py_ntrg->state->node_rank_map.find(tree_index[i]);
        if (it == py_ntrg->state->node_rank_map.end())
          {
            py_ntrg->state->node_rank_map.insert(make_pair(tree_index[i], r++));
          }
        else
          {
            throw_err("NeuroH5TreeGen: tree generator requires unique index set");
          }
        if ((unsigned int)size <= r) r=0;
      }
    

    size_t max_local_count=0;
    status = MPI_Allreduce(&(local_count), &max_local_count, 1,
                           MPI_SIZE_T, MPI_MAX, comm);
    throw_assert(status == MPI_SUCCESS,
                 "NeuroH5TreeGen: MPI_Allreduce error");


    throw_assert(MPI_Comm_dup(comm, &(py_ntrg->state->comm)) == MPI_SUCCESS,
                 "NeuroH5TreeGen: unable to duplicate MPI communicator");

    throw_assert(MPI_Comm_free(&comm) == MPI_SUCCESS,
                 "NeuroH5TreeGen: unable to free MPI communicator");

    py_ntrg->state->pos             = seq_next;
    py_ntrg->state->count           = count;
    py_ntrg->state->local_count     = local_count;
    py_ntrg->state->max_local_count = max_local_count;
    py_ntrg->state->seq_index       = 0;
    py_ntrg->state->cache_index     = 0;
    py_ntrg->state->file_name  = string(file_name);
    py_ntrg->state->pop_name   = string(pop_name);
    py_ntrg->state->pop_idx    = pop_idx;
    py_ntrg->state->pop_vector = pop_vector;
    py_ntrg->state->io_size    = io_size;
    py_ntrg->state->comm_size  = size;
    py_ntrg->state->cache_size = cache_size;
    py_ntrg->state->attr_name_spaces  = attr_name_spaces;
    py_ntrg->state->topology_flag  = topology_flag;

    map<CELL_IDX_T, neurotree_t> tree_map;
    py_ntrg->state->tree_map = tree_map;
    py_ntrg->state->it_tree  = py_ntrg->state->tree_map.cbegin();

    return (PyObject *)py_ntrg;
  }


  static PyObject *
  neuroh5_cell_attr_gen_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
  {
    int status;
    PyObject *py_comm = NULL;
    PyObject *py_mask = NULL;
    MPI_Comm *comm_ptr  = NULL;
    unsigned long io_size=1, cache_size=100;
    const string default_namespace = "Attributes";
    char *file_name, *pop_name, *attr_namespace = (char *)default_namespace.c_str();
    int return_struct;
    
    static const char *kwlist[] = {
                                   "file_name",
                                   "pop_name",
                                   "namespace",
                                   "comm",
                                   "mask",
                                   "io_size",
                                   "cache_size",
                                   "return_struct",
                                   NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ss|sOOkib", (char **)kwlist,
                                     &file_name, &pop_name, &attr_namespace,
                                     &py_comm, &py_mask, &io_size, &cache_size,
                                     &return_struct))
      return NULL;

    set<string> attr_mask;

    if (py_mask != NULL)
      {
        throw_assert(PySet_Check(py_mask),
                     "py_read_trees: argument mask must be a set of strings");
        
        PyObject *py_iter = PyObject_GetIter(py_mask);
        if (py_iter != NULL)
          {
            PyObject *pyval;
            while((pyval = PyIter_Next(py_iter)))
              {
                const char* str = PyStr_ToCString (pyval);
                attr_mask.insert(string(str));
                Py_DECREF(pyval);
              }
          }

        Py_DECREF(py_iter);
      }

    MPI_Comm comm;

    if (py_comm != NULL)
      {
        comm_ptr = PyMPIComm_Get(py_comm);
        throw_assert(comm_ptr != NULL, "NeuroH5CellAttrGen: invalid MPI communicator");
        throw_assert(*comm_ptr != MPI_COMM_NULL, "NeuroH5TreeGen: null MPI communicator");
        status = MPI_Comm_dup(*comm_ptr, &comm);
        throw_assert(status == MPI_SUCCESS, "NeuroH5CellAttrGen: unable to duplicate MPI communicator");
      }
    else
      {
        status = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        throw_assert(status == MPI_SUCCESS, "NeuroH5CellAttrGen: unable to duplicate MPI communicator");
      }

    int rank, size;
    throw_assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS,
                 "NeuroH5CellAttrGen: unable to obtain MPI communicator size");
    throw_assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS,
                 "NeuroH5CellAttrGen: unable to obtain MPI communicator rank");
    
    if (io_size > (unsigned int)size)
      io_size = size;

    if ((size > 0) && (cache_size < (unsigned int)size))
      cache_size = size;

    vector<pair <pop_t, string> > pop_labels;
    status = cell::read_population_labels(comm, string(file_name), pop_labels);
    throw_assert (status >= 0,
                  "NeuroH5CellAttrGen: unable to read population labels");
    
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
    throw_assert(cell::read_population_ranges(comm,
                                              string(file_name),
                                              pop_ranges, pop_vector,
                                              n_nodes) >= 0,
                 "NeuroH5CellAttrGen: unable to read population ranges");
                 

    vector< pair<string,AttrKind> > attr_info;
    throw_assert(cell::get_cell_attributes (string(file_name), string(attr_namespace),
                                            get<1>(pop_labels[pop_idx]), attr_info) >= 0,
                 "NeuroH5CellAttrGen: unable to read cell attributes metadata");

    vector<CELL_IDX_T> cell_index;
    if (attr_info.size() > 0)
      {
        throw_assert(cell::read_cell_index(comm,
                                           string(file_name),
                                           get<1>(pop_labels[pop_idx]),
                                           string(attr_namespace) + "/" + attr_info[0].first,
                                           cell_index) >= 0,
                     "NeuroH5CellAttrGen: unable to read cell index");
      }
    
    size_t count = cell_index.size();
    for (size_t i=0; i<count; i++)
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
    rank_t r=0; size_t local_count=0; 
    for (size_t i = 0; i < cell_index.size(); i++)
      {
        if ((unsigned int)rank == r) local_count++;
        auto it = py_ntrg->state->node_rank_map.find(cell_index[i]);
        if (it == py_ntrg->state->node_rank_map.end())
          {
            py_ntrg->state->node_rank_map.insert(make_pair(cell_index[i], r++));
          }
        else
          {
            throw_err("NeuroH5CellAttrGen: cell attribute generator requires a unique index set");
          }
        if ((unsigned int)size <= r) r=0;
      }
    
    throw_assert(MPI_Comm_dup(comm, &(py_ntrg->state->comm)) == MPI_SUCCESS,
                 "NeuroH5CellAttrGen: unable to duplicate MPI communicator");

    size_t max_local_count=0;
    status = MPI_Allreduce(&(local_count), &max_local_count, 1,
                           MPI_SIZE_T, MPI_MAX, comm);
    throw_assert(status == MPI_SUCCESS,
                 "NeuroH5CellAttrGen: unable to free MPI communicator");

    
    py_ntrg->state->pos            = seq_next;
    py_ntrg->state->count          = count;
    py_ntrg->state->local_count    = local_count;
    py_ntrg->state->max_local_count= max_local_count;
    py_ntrg->state->seq_index      = 0;
    py_ntrg->state->cache_index    = 0;
    py_ntrg->state->file_name      = string(file_name);
    py_ntrg->state->pop_name       = string(pop_name);
    py_ntrg->state->pop_idx        = pop_idx;
    py_ntrg->state->pop_vector     = pop_vector;
    py_ntrg->state->io_size        = io_size;
    py_ntrg->state->comm_size      = size;
    py_ntrg->state->cache_size     = cache_size;
    py_ntrg->state->attr_namespace = string(attr_namespace);
    py_ntrg->state->attr_mask      = attr_mask;
    py_ntrg->state->struct_type    = NULL;
    py_ntrg->state->return_struct  = return_struct>0;
    
    NamedAttrMap attr_map;
    py_ntrg->state->attr_map  = attr_map;
    py_ntrg->state->it_idx = py_ntrg->state->attr_map.index_set.cbegin();

    return (PyObject *)py_ntrg;
  }

  static void
  neuroh5_tree_gen_dealloc(PyNeuroH5TreeGenState *py_ntrg)
  {
    if (py_ntrg->state->pos == seq_next)
      {
        int status = MPI_Comm_free(&(py_ntrg->state->comm));
        throw_assert(status == MPI_SUCCESS,
                     "NeuroH5TreeGen: unable to free MPI communicator");
      }
    delete py_ntrg->state;
    Py_TYPE(py_ntrg)->tp_free(py_ntrg);
  }

  static void
  neuroh5_cell_attr_gen_dealloc(PyNeuroH5CellAttrGenState *py_ntrg)
  {
    Py_XDECREF(py_ntrg->state->struct_type);
    if (py_ntrg->state->pos == seq_next)
      {
        int status = MPI_Comm_free(&(py_ntrg->state->comm));
        throw_assert(status == MPI_SUCCESS,
                     "NeuroH5CellAttrGen: unable to free MPI communicator");
      }
    delete py_ntrg->state;
    Py_TYPE(py_ntrg)->tp_free(py_ntrg);
  }

  static void
  neuroh5_prj_gen_dealloc(PyNeuroH5ProjectionGenState *py_ngg)
  {
    if (py_ngg->state->pos == seq_next)
      {
        int status = MPI_Comm_free(&(py_ngg->state->comm));
        throw_assert(status == MPI_SUCCESS, 
                     "NeuroH5ProjectionGen: unable to free MPI communicator");
      }
    delete py_ngg->state;
    Py_TYPE(py_ngg)->tp_free(py_ngg);
  }

  static PyObject *
  neuroh5_tree_gen_next(PyNeuroH5TreeGenState *py_ntrg)
  {
    PyObject *result = NULL; 

    /* 
     * Returning NULL in this case is enough. The next() builtin will raise the
     * StopIteration error for us.
     */
    switch (py_ntrg->state->pos)
      {
      case seq_next:
        {
          int size, rank;
          throw_assert(MPI_Comm_size(py_ntrg->state->comm, &size) == MPI_SUCCESS,
                       "NeuroH5TreeGen: unable to obtain MPI communicator size");
          throw_assert(MPI_Comm_rank(py_ntrg->state->comm, &rank) == MPI_SUCCESS,
                       "NeuroH5TreeGen: unable to obtain MPI communicator rank");


          // If the end of the current cache block has been reached,
          // and the iterator has not exceed its locally assigned elements,
          // read the next block
          if ((py_ntrg->state->it_tree == py_ntrg->state->tree_map.cend()) &&
              (py_ntrg->state->cache_index < py_ntrg->state->count))
            {
              int status;
              py_ntrg->state->tree_map.clear();
              py_ntrg->state->attr_maps.clear();
              status = cell::scatter_read_trees (py_ntrg->state->comm,
                                                 py_ntrg->state->file_name,
                                                 py_ntrg->state->io_size,
                                                 py_ntrg->state->attr_name_spaces,
                                                 py_ntrg->state->node_rank_map,
                                                 py_ntrg->state->pop_name,
                                                 py_ntrg->state->pop_vector[py_ntrg->state->pop_idx].start,
                                                 py_ntrg->state->tree_map,
                                                 py_ntrg->state->attr_maps,
                                                 py_ntrg->state->cache_index,
                                                 py_ntrg->state->cache_size);
              throw_assert (status >= 0,
                            "NeuroH5TreeGen: error in call to cell::scatter_read_trees");


              if (py_ntrg->state->cache_index < py_ntrg->state->count)
                {
                  py_ntrg->state->cache_index += py_ntrg->state->comm_size * py_ntrg->state->cache_size;
                }
              py_ntrg->state->it_tree = py_ntrg->state->tree_map.cbegin();
            }

          if (py_ntrg->state->it_tree == py_ntrg->state->tree_map.cend())
            {
              if (py_ntrg->state->seq_index == py_ntrg->state->max_local_count)
                {
                  int status = MPI_Comm_free(&(py_ntrg->state->comm));
                  throw_assert(status == MPI_SUCCESS,
                               "NeuroH5TreeGen: unable to free MPI communicator");

                  py_ntrg->state->pos = seq_last;
                }
              else
                {
                  py_ntrg->state->seq_index++;
                }
              result = PyTuple_Pack(2,
                                    (Py_INCREF(Py_None), Py_None),
                                    (Py_INCREF(Py_None), Py_None));

              break;
            }
          else
            {
              CELL_IDX_T key = py_ntrg->state->it_tree->first;
              const neurotree_t &tree = py_ntrg->state->it_tree->second;
              PyObject *elem = py_build_tree_value(key, tree, py_ntrg->state->attr_maps, py_ntrg->state->topology_flag);
              throw_assert(elem != NULL,
                           "NeuroH5TreeGen: invalid tree value");

              
              /* Exceptions from PySequence_GetItem are propagated to the caller
               * (elem will be NULL so we also return NULL).
               */
              result = Py_BuildValue("lN", key, elem);
              py_ntrg->state->it_tree++;
              py_ntrg->state->seq_index++;
            }
          break;
        }
      case seq_empty:
        {
          if (py_ntrg->state->seq_index == py_ntrg->state->max_local_count)
            {
              int status = MPI_Comm_free(&(py_ntrg->state->comm));
              throw_assert(status == MPI_SUCCESS,
                           "NeuroH5TreeGen: unable to free MPI communicator");

              py_ntrg->state->pos = seq_last;
            }
          else
            {
              py_ntrg->state->seq_index++;
            }
          
          result = PyTuple_Pack(2,
                                (Py_INCREF(Py_None), Py_None),
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
      }
    /* Exceptions from PySequence_GetItem are propagated to the caller
     * (elem will be NULL so we also return NULL).
     */
    return result;
  }

  
  static PyObject *
  neuroh5_cell_attr_gen_next(PyNeuroH5CellAttrGenState *py_ntrg)
  {
    PyObject *result = NULL; 
    
    switch (py_ntrg->state->pos)
      {
      case seq_next:
        {
          

          if ((py_ntrg->state->it_idx == py_ntrg->state->attr_map.index_set.cend()) &&
              (py_ntrg->state->cache_index < py_ntrg->state->count))
            {
              int size, rank;
              throw_assert(MPI_Comm_size(py_ntrg->state->comm, &size) == MPI_SUCCESS,
                           "NeuroH5CellAttrGen: unable to obtain MPI communicator size");
              throw_assert(MPI_Comm_rank(py_ntrg->state->comm, &rank) == MPI_SUCCESS,
                           "NeuroH5CellAttrGen: unable to obtain MPI communicator rank");

              // If the end of the current cache block has been reached,
              // read the next block
              py_ntrg->state->attr_map.clear();
              
              int status;
              status = cell::scatter_read_cell_attributes (py_ntrg->state->comm,
                                                           py_ntrg->state->file_name,
                                                           py_ntrg->state->io_size,
                                                           py_ntrg->state->attr_namespace,
                                                           py_ntrg->state->attr_mask,
                                                           py_ntrg->state->node_rank_map,
                                                           py_ntrg->state->pop_name,
                                                           py_ntrg->state->pop_vector[py_ntrg->state->pop_idx].start,
                                                           py_ntrg->state->attr_map,
                                                           py_ntrg->state->cache_index,
                                                           py_ntrg->state->cache_size);
              throw_assert (status >= 0,
                            "NeuroH5CellAttrGen: error in call to cell::scatter_read_cell_attributes");

              py_ntrg->state->attr_map.attr_names(py_ntrg->state->attr_names);
              py_ntrg->state->it_idx = py_ntrg->state->attr_map.index_set.cbegin();
              py_ntrg->state->cache_index += py_ntrg->state->comm_size * py_ntrg->state->cache_size;
              if (py_ntrg->state->struct_type == NULL)
                {
                  py_ntrg->state->struct_type = py_build_cell_attr_struct_type(py_ntrg->state->attr_map,
                                                                               py_ntrg->state->attr_names,
                                                                               py_ntrg->state->struct_descr_fields);
                }
            }


          if (py_ntrg->state->it_idx == py_ntrg->state->attr_map.index_set.cend())
            {
              if (py_ntrg->state->seq_index == py_ntrg->state->max_local_count)
                {
                  int status = MPI_Comm_free(&(py_ntrg->state->comm));
                  throw_assert(status == MPI_SUCCESS,
                               "NeuroH5CellAttrGen: unable to free MPI communicator");

                  py_ntrg->state->attr_map.clear();
                  py_ntrg->state->pos = seq_last;
                }
              else
                {
                  py_ntrg->state->seq_index++;
                }
              result = PyTuple_Pack(2,
                                    (Py_INCREF(Py_None), Py_None),
                                    (Py_INCREF(Py_None), Py_None));
              break;
            }
          else
            {
              const CELL_IDX_T key = *(py_ntrg->state->it_idx);
              PyObject *elem;

              if (py_ntrg->state->return_struct)
                {
                  elem = py_build_cell_attr_values_struct(key, py_ntrg->state->attr_map,
                                                          py_ntrg->state->attr_names,
                                                          py_ntrg->state->struct_type);
                }
              else
                {
                  elem = py_build_cell_attr_values_dict(key, py_ntrg->state->attr_map,
                                                        py_ntrg->state->attr_names);
                }
              
              throw_assert(elem != NULL,
                           "NeuroH5CellAttrGen: invalid cell attribute value");

              py_ntrg->state->it_idx++;
              py_ntrg->state->seq_index++;
              result = Py_BuildValue("lN", key, elem);
            }

          break;
        }
      case seq_empty:
        {
          py_ntrg->state->attr_map.clear();
          if (py_ntrg->state->seq_index == py_ntrg->state->max_local_count)
            {
              int status = MPI_Comm_free(&(py_ntrg->state->comm));
              throw_assert(status == MPI_SUCCESS,
                           "NeuroH5CellAttrGen: unable to free MPI communicator");

              py_ntrg->state->pos = seq_last;
            }
          else
            {
              py_ntrg->state->seq_index++;
            }

          result = PyTuple_Pack(2,
                                (Py_INCREF(Py_None), Py_None),
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
      }
    /* Exceptions from PySequence_GetItem are propagated to the caller
     * (elem will be NULL so we also return NULL).
     */
    return result;
  }

  
  static int neuroh5_prj_gen_next_block(PyNeuroH5ProjectionGenState *py_ngg)
  {
    int status = 0;
    int size, rank;

    throw_assert(MPI_Comm_size(py_ngg->state->comm, &size) == MPI_SUCCESS,
                 "NeuroH5ProjectionGen: invalid MPI communicator");
    throw_assert(MPI_Comm_rank(py_ngg->state->comm, &rank) == MPI_SUCCESS,
                 "NeuroH5ProjectionGen: invalid MPI communicator");
    
    if (!(py_ngg->state->block_index < py_ngg->state->block_count))
      return 0;

    // If the end of the current edge map has been reached,
    // read the next block
    py_ngg->state->edge_map.clear();

    vector < map <string, vector < vector<string> > > > edge_attr_name_vector;
    vector <edge_map_t> prj_vector;

    
    status = graph::read_projection(py_ngg->state->comm,
                                    py_ngg->state->file_name,
                                    py_ngg->state->pop_ranges,
                                    py_ngg->state->pop_pairs,
                                    py_ngg->state->src_pop_name,
                                    py_ngg->state->dst_pop_name,
                                    py_ngg->state->src_start,
                                    py_ngg->state->dst_start,
                                    py_ngg->state->edge_attr_name_spaces,
                                    prj_vector,
                                    edge_attr_name_vector,
                                    py_ngg->state->local_num_nodes,
                                    py_ngg->state->local_num_edges,
                                    py_ngg->state->total_num_edges,
                                    py_ngg->state->local_read_blocks,
                                    py_ngg->state->total_read_blocks,
                                    py_ngg->state->block_index,
                                    py_ngg->state->cache_size);

    throw_assert (status >= 0, "NeuroH5ProjectionGen: read_projection error");
    throw_assert(prj_vector.size() > 0, "NeuroH5ProjectionGen: empty projection");

    if (edge_attr_name_vector.size() > 0)
      {
        py_ngg->state->edge_attr_names = edge_attr_name_vector[0];
      }
    
    py_ngg->state->edge_map = prj_vector[0];
    //throw_assert(py_ngg->state->edge_map.size() > 0);
    py_ngg->state->edge_map_iter = py_ngg->state->edge_map.cbegin();
    
    py_ngg->state->block_index += py_ngg->state->total_read_blocks;

    size_t max_local_num_nodes=0;
    status = MPI_Allreduce(&(py_ngg->state->local_num_nodes), &max_local_num_nodes, 1,
                           MPI_SIZE_T, MPI_MAX, py_ngg->state->comm);
    throw_assert(status == MPI_SUCCESS, "NeuroH5ProjectionGen: MPI_Allreduce error");
    py_ngg->state->node_count += max_local_num_nodes;

    return status;
  }

  
  static PyObject *
  neuroh5_prj_gen_next(PyNeuroH5ProjectionGenState *py_ngg)
  {
    PyObject *result = NULL;

    int status = 0;

    throw_assert(py_ngg->state->node_index <= py_ngg->state->node_count,
                 "NeuroH5ProjectionGen: node index / node count mismatch");

    switch (py_ngg->state->pos)
      {
      case seq_next:
        {
          int size, rank;
          throw_assert(MPI_Comm_size(py_ngg->state->comm, &size) == MPI_SUCCESS,
                       "NeuroH5ProjectionGen: invalid MPI communicator");
          throw_assert(MPI_Comm_rank(py_ngg->state->comm, &rank) == MPI_SUCCESS,
                       "NeuroH5ProjectionGen: invalid MPI communicator");

          if ((py_ngg->state->edge_map_iter == py_ngg->state->edge_map.cend()) &&
              (py_ngg->state->node_index == py_ngg->state->node_count))
            {
              if (py_ngg->state->block_index < py_ngg->state->block_count)
                {
                  neuroh5_prj_gen_next_block(py_ngg);
                }
              else
                {
                  if (py_ngg->state->node_index == py_ngg->state->node_count)
                    {
                      int status = MPI_Comm_free(&(py_ngg->state->comm));
                      throw_assert(status == MPI_SUCCESS,
                                   "NeuroH5ProjectionGen: unable to free MPI communicator");

                      py_ngg->state->pos = seq_last;
                    }
                  else
                    {
                      py_ngg->state->node_index++;
                    }

                  result = PyTuple_Pack(2,
                                        (Py_INCREF(Py_None), Py_None),
                                        (Py_INCREF(Py_None), Py_None));
                  break;
                }
            }

          

          if ((py_ngg->state->edge_map_iter != py_ngg->state->edge_map.cend()))
            {
              const NODE_IDX_T key = py_ngg->state->edge_map_iter->first;
              PyObject *py_edge = py_build_edge_array_dict_value(key,
                                                                 py_ngg->state->edge_map_iter->second,
                                                                 py_ngg->state->edge_attr_name_spaces,
                                                                 py_ngg->state->edge_attr_names
                                                                 );
              PyObject *py_key = PyLong_FromLong(key);
              result = PyTuple_Pack(2, py_key, py_edge);
              
              py_ngg->state->edge_map_iter = next(py_ngg->state->edge_map_iter);
            }
          else
            {
              result = PyTuple_Pack(2,
                                    (Py_INCREF(Py_None), Py_None),
                                    (Py_INCREF(Py_None), Py_None));
            }

          if (py_ngg->state->node_index == py_ngg->state->node_count)
            {
              int status = MPI_Comm_free(&(py_ngg->state->comm));
              throw_assert(status == MPI_SUCCESS,
                     "NeuroH5ProjectionGen: unable to free MPI communicator");
              py_ngg->state->pos = seq_last;
            }
          else
            {
              py_ngg->state->node_index++;
            }

          break;
        }
      case seq_empty:
        {
          if (py_ngg->state->node_index == py_ngg->state->node_count)
            {
              int status = MPI_Comm_free(&(py_ngg->state->comm));
              throw_assert(status == MPI_SUCCESS, "NeuroH5ProjectionGen: unable to free MPI communicator");
              py_ngg->state->pos = seq_last;
            }
          else
            {
              py_ngg->state->node_index++;
            }
          result = PyTuple_Pack(2,
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


  
  
  // NeuroH5 tree read generator
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

  
  // NeuroH5 attribute read generator
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
    { "read_population_ranges", (PyCFunction)py_read_population_ranges, METH_VARARGS | METH_KEYWORDS,
       read_population_ranges_doc },
    { "read_population_names", (PyCFunction)py_read_population_names, METH_VARARGS | METH_KEYWORDS,
      read_population_names_doc },
    { "read_projection_names", (PyCFunction)py_read_projection_names, METH_VARARGS | METH_KEYWORDS,
      read_projection_names_doc },
    { "read_trees", (PyCFunction)py_read_trees, METH_VARARGS | METH_KEYWORDS,
      read_trees_doc },
    { "read_tree_selection", (PyCFunction)py_read_tree_selection, METH_VARARGS | METH_KEYWORDS,
      read_tree_selection_doc },
    { "scatter_read_trees", (PyCFunction)py_scatter_read_trees, METH_VARARGS | METH_KEYWORDS,
      scatter_read_trees_doc },
    { "read_cell_attribute_info", (PyCFunction)py_read_cell_attribute_info, METH_VARARGS | METH_KEYWORDS,
      read_cell_attribute_info_doc },
    { "read_cell_attribute_selection", (PyCFunction)py_read_cell_attribute_selection, METH_VARARGS | METH_KEYWORDS,
       read_cell_attribute_selection_doc },
    { "read_cell_attributes", (PyCFunction)py_read_cell_attributes, METH_VARARGS | METH_KEYWORDS,
      read_cell_attributes_doc },
    { "scatter_read_cell_attributes", (PyCFunction)py_scatter_read_cell_attributes, METH_VARARGS | METH_KEYWORDS,
      scatter_read_cell_attributes_doc },
    { "bcast_cell_attributes", (PyCFunction)py_bcast_cell_attributes, METH_VARARGS | METH_KEYWORDS,
      "Reads attributes for the given range of cells and broadcasts to all ranks." },
    { "write_cell_attributes", (PyCFunction)py_write_cell_attributes, METH_VARARGS | METH_KEYWORDS,
      "Writes attributes for the given range of cells." },
    { "append_cell_attributes", (PyCFunction)py_append_cell_attributes, METH_VARARGS | METH_KEYWORDS,
      "Appends additional attributes for the given range of cells." },
    { "append_cell_trees", (PyCFunction)py_append_cell_trees, METH_VARARGS | METH_KEYWORDS,
      "Appends tree morphologies." },
    { "read_graph", (PyCFunction)py_read_graph, METH_VARARGS | METH_KEYWORDS,
      "Reads graph connectivity in Destination Block Sparse format." },
    { "scatter_read_graph", (PyCFunction)py_scatter_read_graph, METH_VARARGS | METH_KEYWORDS,
      "Reads and scatters graph connectivity in Destination Block Sparse format." },
    { "bcast_graph", (PyCFunction)py_bcast_graph, METH_VARARGS | METH_KEYWORDS,
      "Reads and broadcasts graph connectivity in Destination Block Sparse format." },
    { "read_graph_selection", (PyCFunction)py_read_graph_selection, METH_VARARGS | METH_KEYWORDS,
      "Reads graph connectivity in Destination Block Sparse format." },
    { "write_graph", (PyCFunction)py_write_graph, METH_VARARGS | METH_KEYWORDS,
      "Writes graph connectivity in Destination Block Sparse format." },
    { "append_graph", (PyCFunction)py_append_graph, METH_VARARGS | METH_KEYWORDS,
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
  throw_assert(import_mpi4py() >= 0, "Error importing mpi4py");
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


  if (PyType_Ready(&PyNeuroH5TreeIter_Type) < 0)
    {
      printf("NeuroH5TreeIter type cannot be added\n");
#if PY_MAJOR_VERSION >= 3
      return NULL;
#else      
      return;
#endif
    }

  Py_INCREF((PyObject *)&PyNeuroH5TreeIter_Type);
  PyModule_AddObject(module, "NeuroH5TreeIter", (PyObject *)&PyNeuroH5TreeIter_Type);

  
  if (PyType_Ready(&PyNeuroH5CellAttrIter_Type) < 0)
    {
      printf("NeuroH5CellAttrIter type cannot be added\n");
#if PY_MAJOR_VERSION >= 3
      return NULL;
#else      
      return;
#endif
    }

  Py_INCREF((PyObject *)&PyNeuroH5CellAttrIter_Type);
  PyModule_AddObject(module, "NeuroH5CellAttrIter", (PyObject *)&PyNeuroH5CellAttrIter_Type);


  if (PyType_Ready(&PyNeuroH5EdgeIter_Type) < 0)
    {
      printf("NeuroH5EdgeIter type cannot be added\n");
#if PY_MAJOR_VERSION >= 3
      return NULL;
#else      
      return;
#endif
    }

  Py_INCREF((PyObject *)&PyNeuroH5EdgeIter_Type);
  PyModule_AddObject(module, "NeuroH5EdgeIter", (PyObject *)&PyNeuroH5EdgeIter_Type);

#if PY_MAJOR_VERSION >= 3
  return module;
#else
  return;
#endif
}

  

