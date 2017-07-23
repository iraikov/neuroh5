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
                             vector<NODE_IDX_T>&    dst,
                             vector<NODE_IDX_T>&    src,
                             neuroh5::data::AttrVal& attrs)
    {
      ifstream infile(file_name.c_str());
      string line;
      size_t i = 0;
      
      vector <vector <float>>    float_attrs(num_attrs[0]);
      vector <vector <uint8_t>>  uint8_attrs(num_attrs[1]);
      vector <vector <uint16_t>> uint16_attrs(num_attrs[2]);
      vector <vector <uint32_t>> uint32_attrs(num_attrs[3]);
      
      while (getline(infile, line))
        {
          istringstream iss(line);
          NODE_IDX_T src, dst;
          
          assert (iss >> src);
          assert (iss >> dst);

          // floating point attrs
          for (size_t a=0; a<num_attrs[0]; a++)
            {
              float v;
              iss >> v;
              float_attrs[a].push_back(v);
            }
          // uint8 point attrs
          for (size_t a=0; a<num_attrs[1]; a++)
            {
              uint8_t v;
              iss >> v;
              uint8_attrs[a].push_back(v);
            }
          // uint16 point attrs
          for (size_t a=0; a<num_attrs[2]; a++)
            {
              uint16_t v;
              iss >> v;
              uint16_attrs[a].push_back(v);
            }
          // uint32 point attrs
          for (size_t a=0; a<num_attrs[3]; a++)
            {
              uint32_t v;
              iss >> v;
              uint32_attrs[a].push_back(v);
            }

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
      
      infile.close();

      return 0;
    }
    
  }

}
