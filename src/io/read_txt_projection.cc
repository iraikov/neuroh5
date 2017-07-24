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
      
      vector <vector <float>>    float_attrs(num_attrs[0]);
      vector <vector <uint8_t>>  uint8_attrs(num_attrs[1]);
      vector <vector <uint16_t>> uint16_attrs(num_attrs[2]);
      vector <vector <uint32_t>> uint32_attrs(num_attrs[3]);
      
      while (getline(infile, line))
        {
          istringstream iss(line);
          NODE_IDX_T src, dst;
          
          assert (iss >> dst);
          assert (iss >> src);

          dst_src_map[dst].push_back(src);
          
          // floating point attrs
          if (num_attrs[0] > 0)
            for (size_t a=0; a<num_attrs[0]; a++)
              {
                float v;
                iss >> v;
                float_attrs[a].push_back(v);
              }
          // uint8 point attrs
          if (num_attrs[1] > 0)
            for (size_t a=0; a<num_attrs[1]; a++)
              {
                uint8_t v;
                iss >> v;
                uint8_attrs[a].push_back(v);
              }
          // uint16 point attrs
          if (num_attrs[2] > 0)
            for (size_t a=0; a<num_attrs[2]; a++)
              {
                uint16_t v;
                iss >> v;
                uint16_attrs[a].push_back(v);
              }
          // uint32 point attrs
          if (num_attrs[3] > 0)
            for (size_t a=0; a<num_attrs[3]; a++)
              {
                uint32_t v;
                iss >> v;
                uint32_attrs[a].push_back(v);
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
      for (size_t a=0; a<num_attrs[0]; a++)
        {
          attrs.insert(float_attrs[a]);
        }
      for (size_t a=0; a<num_attrs[1]; a++)
        {
          attrs.insert(uint8_attrs[a]);
        }
      for (size_t a=0; a<num_attrs[2]; a++)
        {
          attrs.insert(uint16_attrs[a]);
        }
      for (size_t a=0; a<num_attrs[3]; a++)
        {
          attrs.insert(uint32_attrs[a]);
        }
      

      return 0;
    }
    
  }

}

