// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_txt_projection.cc
///
///  Read a projection in text format.
///
///  Copyright (C) 2016-2020 Project NeuroH5.
//==============================================================================


#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iomanip>
#include <sstream>


#include "read_txt_projection.hh"
#include "neuroh5_types.hh"
#include "attr_val.hh"
#include "throw_assert.hh"

using namespace std;

namespace neuroh5
{
  namespace io
  {

    int read_txt_projection (const string&           file_name,
                             const map <string, vector <size_t> >&  num_attrs,
                             vector <NODE_IDX_T>&    dst_idx,
                             vector <DST_PTR_T>&     src_idx_ptr,
                             vector <NODE_IDX_T>&    src_idx,
                             map <string, neuroh5::data::AttrVal>& attrs_map)
    {
      ifstream infile(file_name.c_str());
      string line;
      size_t i = 0;
      map<NODE_IDX_T, vector<NODE_IDX_T> > dst_src_map;
      
      map <string, vector <vector <float> > >    float_attr_map;
      map <string, vector <vector <uint8_t> > >  uint8_attr_map;
      map <string, vector <vector <uint16_t> > > uint16_attr_map;
      map <string, vector <vector <uint32_t> > > uint32_attr_map;
      map <string, vector <vector <int8_t> > >   int8_attr_map;
      map <string, vector <vector <int16_t> > >  int16_attr_map;
      map <string, vector <vector <int32_t> > >  int32_attr_map;

      while (getline(infile, line))
        {
          istringstream iss(line);
          NODE_IDX_T src, dst;
          
          throw_assert_nomsg (iss >> dst);
          throw_assert_nomsg (iss >> src);

          dst_src_map[dst].push_back(src);

          for (auto iter : num_attrs)
            {
              auto & float_attrs  = float_attr_map[iter.first];
              auto & uint8_attrs  = uint8_attr_map[iter.first];
              auto & uint16_attrs = uint16_attr_map[iter.first];
              auto & uint32_attrs = uint32_attr_map[iter.first];
              auto & int8_attrs   = int8_attr_map[iter.first];
              auto & int16_attrs  = int16_attr_map[iter.first];
              auto & int32_attrs  = int32_attr_map[iter.first];

              float_attrs.resize(iter.second[data::AttrVal::attr_index_float]);
              uint8_attrs.resize(iter.second[data::AttrVal::attr_index_uint8]);
              uint16_attrs.resize(iter.second[data::AttrVal::attr_index_uint16]);
              uint32_attrs.resize(iter.second[data::AttrVal::attr_index_uint32]);
              int8_attrs.resize(iter.second[data::AttrVal::attr_index_int8]);
              int16_attrs.resize(iter.second[data::AttrVal::attr_index_int16]);
              int32_attrs.resize(iter.second[data::AttrVal::attr_index_int32]);

              // floating point attrs
              if (iter.second[data::AttrVal::attr_index_float] > 0)
                for (size_t a=0; a<iter.second[data::AttrVal::attr_index_float]; a++)
                  {
                    float v;
                    iss >> v;
                    float_attrs[a].push_back(v);
                  }
              // uint8 point attrs
              if (iter.second[data::AttrVal::attr_index_uint8] > 0)
                for (size_t a=0; a<iter.second[data::AttrVal::attr_index_uint8]; a++)
                  {
                    uint8_t v;
                    iss >> v;
                    uint8_attrs[a].push_back(v);
                  }
              // uint16 point attrs
              if (iter.second[data::AttrVal::attr_index_uint16] > 0)
                for (size_t a=0; a<iter.second[data::AttrVal::attr_index_uint16]; a++)
                  {
                    uint16_t v;
                    iss >> v;
                    uint16_attrs[a].push_back(v);
                  }
              // uint32 point attrs
              if (iter.second[data::AttrVal::attr_index_uint32] > 0)
                for (size_t a=0; a<iter.second[data::AttrVal::attr_index_uint32]; a++)
                  {
                    uint32_t v;
                    iss >> v;
                    uint32_attrs[a].push_back(v);
                  }
              // int8 point attrs
              if (iter.second[data::AttrVal::attr_index_int8] > 0)
                for (size_t a=0; a<iter.second[data::AttrVal::attr_index_int8]; a++)
                  {
                    int8_t v;
                    iss >> v;
                    int8_attrs[a].push_back(v);
                  }
              // int16 point attrs
              if (iter.second[data::AttrVal::attr_index_int16] > 0)
                for (size_t a=0; a<iter.second[data::AttrVal::attr_index_int16]; a++)
                  {
                    int16_t v;
                    iss >> v;
                    int16_attrs[a].push_back(v);
                  }
              // int32 point attrs
              if (iter.second[data::AttrVal::attr_index_int32] > 0)
                for (size_t a=0; a<iter.second[data::AttrVal::attr_index_int32]; a++)
                  {
                    int32_t v;
                    iss >> v;
                    int32_attrs[a].push_back(v);
                  }
            }
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
      
      for (auto iter : num_attrs)
        {
          auto & float_attrs  = float_attr_map[iter.first];
          auto & uint8_attrs  = uint8_attr_map[iter.first];
          auto &  uint16_attrs = uint16_attr_map[iter.first];
          auto &  uint32_attrs = uint32_attr_map[iter.first];
          auto &  int8_attrs   = int8_attr_map[iter.first];
          auto &  int16_attrs  = int16_attr_map[iter.first];
          auto &  int32_attrs  = int32_attr_map[iter.first];

          auto attrs = attrs_map[iter.first];
          
          for (size_t a=0; a<iter.second[data::AttrVal::attr_index_float]; a++)
            {
              attrs.insert(float_attrs[a]);
            }
          for (size_t a=0; a<iter.second[data::AttrVal::attr_index_uint8]; a++)
            {
              attrs.insert(uint8_attrs[a]);
            }
          for (size_t a=0; a<iter.second[data::AttrVal::attr_index_uint16]; a++)
            {
              attrs.insert(uint16_attrs[a]);
            }
          for (size_t a=0; a<iter.second[data::AttrVal::attr_index_uint32]; a++)
            {
              attrs.insert(uint32_attrs[a]);
            }
          for (size_t a=0; a<iter.second[data::AttrVal::attr_index_int8]; a++)
            {
              attrs.insert(int8_attrs[a]);
            }
          for (size_t a=0; a<iter.second[data::AttrVal::attr_index_int16]; a++)
            {
              attrs.insert(int16_attrs[a]);
            }
          for (size_t a=0; a<iter.second[data::AttrVal::attr_index_int32]; a++)
            {
              attrs.insert(int32_attrs[a]);
            }
        }

      return 0;
    }
    
  }

}

