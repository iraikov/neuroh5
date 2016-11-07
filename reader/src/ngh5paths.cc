// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file ngh5paths.cc
///
///  Definitions of Neurograph dataset paths.
///
///  Copyright (C) 2016 Project Neurograph.
//==============================================================================

#include "ngh5paths.hh"

using namespace std;

namespace ngh5
{
  const string H5PathNames::CONN = "/Connectivity";

  const string H5PathNames::POP  = "/Populations";

  const string H5PathNames::PRJ  = "/Projections";

  const string H5PathNames::DST_BLK_PTR = "/Connectivity/Destination Block Pointer";

  const string H5PathNames::DST_IDX = "/Connectivity/Destination Block Index";

  const string H5PathNames::DST_PTR = "/Connectivity/Destination Pointer";

  const string H5PathNames::DST_POP = "/Destination Population";

  const string H5PathNames::SRC_IDX = "/Connectivity/Source Index";

  const string H5PathNames::SRC_POP = "/Source Population";

  const string H5PathNames::POP_RNG = "/Population range";

  const string H5PathNames::POP_COMB = "Valid population projections";

  string ngh5_prj_path (const string& proj_name, const string& name) 
  {
    string result;
    result = H5PathNames::PRJ + string("/") + proj_name + name;
    return result;
  }

  
  string ngh5_pop_path (const string& name) 
  {
    string result;
    result = string("/H5Types/") + name;
    return result;
  }

}
