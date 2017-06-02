#ifndef ENUM_TYPE
#define ENUM_TYPE

#include "hdf5.h"

#include <cassert>
#include <map>
#include <string>
#include <vector>

namespace neuroh5
{
  namespace hdf5
  {
    // The datatype handle returned by this function MUST be freed
    // (via H5Tclose) by the caller or a resource leak will occur.

    template <typename T>
    hid_t create_H5Tenum
    (
     const std::vector< std::pair<T, std::string> > enumeration,
     bool                                           is_signed = true
     )
    {
      hid_t result = -1;

      if (enumeration.size() == 0)
        {
          assert("Empty enumeration found.");
        }

      switch (sizeof(T))
        {
        case 1:
          {
            if (is_signed)
              {
                result = H5Tenum_create(H5T_STD_I8LE);
              }
            else
              {
                result = H5Tenum_create(H5T_STD_U8LE);
              }
            break;
          }
        case 2:
          {
            if (is_signed)
              {
                result = H5Tenum_create(H5T_STD_I16LE);
              }
            else
              {
                result = H5Tenum_create(H5T_STD_U16LE);
              }
            break;
          }
        case 4:
          {
            if (is_signed)
              {
                result = H5Tenum_create(H5T_STD_I32LE);
              }
            else
              {
                result = H5Tenum_create(H5T_STD_U32LE);
              }
            break;
          }
        case 8:
          {
            if (is_signed)
              {
                result = H5Tenum_create(H5T_STD_I64LE);
              }
            else
              {
                result = H5Tenum_create(H5T_STD_U64LE);
              }
            break;
          }
        default:
          assert("Unspported type for enumeration base type found.");
          break;
        }

      assert(result >= 0);

      for (size_t i = 0; i < enumeration.size(); ++i)
        {
          assert(H5Tenum_insert(result, enumeration[i].second.c_str(),
                                &enumeration[i].first) >= 0);
        }

      return result;
    }
  }
}

#endif
