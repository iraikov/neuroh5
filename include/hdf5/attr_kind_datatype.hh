#ifndef ATTR_KIND_DATATYPE
#define ATTR_KIND_DATATYPE

#include <cassert>
#include <type_traits>

#include "neuroh5_types.hh"
#include "hdf5.h"


namespace neuroh5
{
  namespace hdf5
  {


    AttrKind h5type_attr_kind(hid_t attr_h5type);

    hid_t attr_kind_h5type(const AttrKind& k);
  }
}

#endif
