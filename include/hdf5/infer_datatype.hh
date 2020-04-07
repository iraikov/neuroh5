#ifndef INFER_DATATYPE
#define INFER_DATATYPE

#include <hdf5.h>

#include <type_traits>

#include "throw_assert.hh"

template <typename T>
hid_t infer_datatype(const T& x)
{
  hid_t result = -1;

  if (std::is_same<T, double>::value)
    {
      result = H5T_IEEE_F64LE;
    }
  else if (std::is_same<T, float>::value)
    {
      result = H5T_IEEE_F32LE;
    }
  else if (std::is_same<T, int8_t>::value)
    {
      result = H5T_STD_I8LE;
    }
  else if (std::is_same<T, int16_t>::value)
    {
      result = H5T_STD_I16LE;
    }
  else if (std::is_same<T, int32_t>::value)
    {
      result = H5T_STD_I32LE;
    }
  else if (std::is_same<T, int64_t>::value)
    {
      result = H5T_STD_I64LE;
    }
  else if (std::is_same<T, uint8_t>::value)
    {
      result = H5T_STD_U8LE;
    }
  else if (std::is_same<T, uint16_t>::value)
    {
      result = H5T_STD_U16LE;
    }
  else if (std::is_same<T, uint32_t>::value)
    {
      result = H5T_STD_U32LE;
    }
  else if (std::is_same<T, uint64_t>::value)
    {
      result = H5T_STD_U64LE;
    }
  else
    {
      throw_assert(result >= 0, "unknown datatype");
    }

  return result;
}

#endif
