// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_txt_projection.cc
///
///  Read a projection in text format.
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================


#include <cassert>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <sstream>


#include "read_txt_projection.hh"
#include "neuroh5_types.hh"
#include "attr_val.hh"

using namespace std;

namespace neuroh5
{
  namespace io
  {

    int read_txt_projection (const string&          file_name,
                             const vector <size_t>& num_attrs,
                             vector<NODE_IDX_T>&    dst_idx,
                             vector<DST_PTR_T>&     src_idx_ptr,
                             vector<NODE_IDX_T>&    src_idx,
                             neuroh5::data::AttrVal& attrs)
    {
      ifstream infile(file_name.c_str());
      string line;
      size_t i = 0;
      map<NODE_IDX_T, vector<NODE_IDX_T> > dst_src_map;
      
      vector <vector <float>>    float_attrs(num_attrs[data::AttrVal::attr_index_float]);
      vector <vector <uint8_t>>  uint8_attrs(num_attrs[data::AttrVal::attr_index_uint8]);
      vector <vector <uint16_t>> uint16_attrs(num_attrs[data::AttrVal::attr_index_uint16]);
      vector <vector <uint32_t>> uint32_attrs(num_attrs[data::AttrVal::attr_index_uint32]);
      vector <vector <int8_t>>   int8_attrs(num_attrs[data::AttrVal::attr_index_int8]);
      vector <vector <int16_t>>  int16_attrs(num_attrs[data::AttrVal::attr_index_int16]);
      vector <vector <int32_t>>  int32_attrs(num_attrs[data::AttrVal::attr_index_int32]);

      while (getline(infile, line))
        {
          istringstream iss(line);
          NODE_IDX_T src, dst;
          
          assert (iss >> dst);
          assert (iss >> src);

          dst_src_map[dst].push_back(src);
          
          // floating point attrs
          if (num_attrs[data::AttrVal::attr_index_float] > 0)
            for (size_t a=0; a<num_attrs[data::AttrVal::attr_index_float]; a++)
              {
                float v;
                iss >> v;
                float_attrs[a].push_back(v);
              }
          // uint8 point attrs
          if (num_attrs[data::AttrVal::attr_index_uint8] > 0)
            for (size_t a=0; a<num_attrs[data::AttrVal::attr_index_uint8]; a++)
              {
                uint8_t v;
                iss >> v;
                uint8_attrs[a].push_back(v);
              }
          // uint16 point attrs
          if (num_attrs[data::AttrVal::attr_index_uint16] > 0)
            for (size_t a=0; a<num_attrs[data::AttrVal::attr_index_uint16]; a++)
              {
                uint16_t v;
                iss >> v;
                uint16_attrs[a].push_back(v);
              }
          // uint32 point attrs
          if (num_attrs[data::AttrVal::attr_index_uint32] > 0)
            for (size_t a=0; a<num_attrs[data::AttrVal::attr_index_uint32]; a++)
              {
                uint32_t v;
                iss >> v;
                uint32_attrs[a].push_back(v);
              }
          // int8 point attrs
          if (num_attrs[data::AttrVal::attr_index_int8] > 0)
            for (size_t a=0; a<num_attrs[data::AttrVal::attr_index_int8]; a++)
              {
                int8_t v;
                iss >> v;
                int8_attrs[a].push_back(v);
              }
          // int16 point attrs
          if (num_attrs[data::AttrVal::attr_index_int16] > 0)
            for (size_t a=0; a<num_attrs[data::AttrVal::attr_index_int16]; a++)
              {
                int16_t v;
                iss >> v;
                int16_attrs[a].push_back(v);
              }
          // int32 point attrs
          if (num_attrs[data::AttrVal::attr_index_int32] > 0)
            for (size_t a=0; a<num_attrs[data::AttrVal::attr_index_int32]; a++)
              {
                int32_t v;
                iss >> v;
                int32_attrs[a].push_back(v);
              }
          
          i++;
        }

      infile.close();
      i = 0;
      src_idx_ptr.push_back(0);
      for (auto & element : dst_src_map)
        {
          dst_idx.push_back(element.first);
          vector<NODE_IDX_T>& src = element.second;
          src_idx.insert(src_idx.end(), src.begin(), src.end());
          src_idx_ptr.push_back(src_idx_ptr[i] + src.size());
          i++;
        }

      for (size_t a=0; a<num_attrs[data::AttrVal::attr_index_float]; a++)
        {
          attrs.insert(float_attrs[a]);
        }
      for (size_t a=0; a<num_attrs[data::AttrVal::attr_index_uint8]; a++)
        {
          attrs.insert(uint8_attrs[a]);
        }
      for (size_t a=0; a<num_attrs[data::AttrVal::attr_index_uint16]; a++)
        {
          attrs.insert(uint16_attrs[a]);
        }
      for (size_t a=0; a<num_attrs[data::AttrVal::attr_index_uint32]; a++)
        {
          attrs.insert(uint32_attrs[a]);
        }
      for (size_t a=0; a<num_attrs[data::AttrVal::attr_index_int8]; a++)
        {
          attrs.insert(int8_attrs[a]);
        }
      for (size_t a=0; a<num_attrs[data::AttrVal::attr_index_int16]; a++)
        {
          attrs.insert(int16_attrs[a]);
        }
      for (size_t a=0; a<num_attrs[data::AttrVal::attr_index_int32]; a++)
        {
          attrs.insert(int32_attrs[a]);
        }
      

      return 0;
    }
    
  }

}

