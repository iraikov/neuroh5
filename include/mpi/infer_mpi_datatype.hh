#ifndef INFER_MPI_DATATYPE
#define INFER_MPI_DATATYPE

#include <cassert>
#include <mpi.h>

#include <type_traits>

template <typename T>
hid_t infer_mpi_datatype(const T& x)
{
  MPI_Datatype result;

  if (std::is_same<T, double>::value)
    {
      result = MPI_DOUBLE;
    }
  else if (std::is_same<T, float>::value)
    {
      result = MPI_FLOAT;
    }
  else if (std::is_same<T, int8_t>::value)
    {
      result = MPI_INT8_T;
    }
  else if (std::is_same<T, int16_t>::value)
    {
      result = MPI_INT16_T;
    }
  else if (std::is_same<T, int32_t>::value)
    {
      result = MPI_INT32_T;
    }
  else if (std::is_same<T, int64_t>::value)
    {
      result = MPI_INT64_T;
    }
  else if (std::is_same<T, uint8_t>::value)
    {
      result = MPI_UINT8_T;
    }
  else if (std::is_same<T, uint16_t>::value)
    {
      result = MPI_UINT16_T;
    }
  else if (std::is_same<T, uint32_t>::value)
    {
      result = MPI_UINT32_T;
    }
  else if (std::is_same<T, uint64_t>::value)
    {
      result = MPI_UINT64_T;
    }
  else
    {
      assert(result >= 0);
    }

  return result;
}

#endif
