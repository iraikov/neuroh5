

#include <type_traits>

#include "neuroh5_types.hh"
#include "attr_kind_datatype.hh"
#include "throw_assert.hh"

namespace neuroh5
{
  namespace hdf5
  {

    AttrKind h5type_attr_kind(hid_t attr_h5type)
    {
      size_t attr_size = H5Tget_size(attr_h5type);
      switch (H5Tget_class(attr_h5type))
        {
        case H5T_INTEGER:
          if (H5Tget_sign( attr_h5type ) == H5T_SGN_NONE)
            {
              return AttrKind(UIntVal, attr_size);
            }
          else
            {
              return AttrKind(SIntVal, attr_size);
            }
          break;
        case H5T_FLOAT:
          return AttrKind(FloatVal, attr_size);
          break;
        case H5T_ENUM:
          if (attr_size == 1)
            {
              return AttrKind(EnumVal, attr_size);
            }
          else
            {
              throw runtime_error("Unsupported enumerated attribute size");
            };
          break;
        default:
          throw runtime_error("Unsupported attribute type");
          break;
        }
  
      return AttrKind(UIntVal, attr_size);
    }

    hid_t attr_kind_h5type(const AttrKind& k)
    {
      hid_t result = -1;

      switch (k.type)
        {
        case SIntVal:
          {
            switch (k.size)
              {
              case 4:
                result = H5T_NATIVE_INT;
                break;
              case 2:
                result = H5T_NATIVE_SHORT;
                break;
              case 1:
                result = H5T_NATIVE_CHAR;
                break;
              default:
                throw runtime_error("Unsupported attribute type");
              }
          }
          break;
        case UIntVal:
          {
            switch (k.size)
              {
              case 4:
                result = H5T_NATIVE_UINT;
                break;
              case 2:
                result = H5T_NATIVE_USHORT;
                break;
              case 1:
                result = H5T_NATIVE_UCHAR;
                break;
              default:
                throw runtime_error("Unsupported attribute type");
              }
          }
          break;
        case FloatVal:
          {
            switch (k.size)
              {
              case 4:
                result = H5T_NATIVE_FLOAT;
                break;
              case 8:
                result = H5T_NATIVE_DOUBLE;
                break;
              default:
                throw runtime_error("Unsupported attribute type");
              }
          }
          break;
        case EnumVal:
          {
            switch (k.size)
              {
              case 1:
                result = H5T_NATIVE_CHAR;
                break;
              default:
                throw runtime_error("Unsupported attribute type");
              }
          }
          break;
        default:
          throw runtime_error("Unsupported attribute type");
        }

      throw_assert(result >= 0, "unable to determine attribute kind");
      return result;
    }
  }
}
