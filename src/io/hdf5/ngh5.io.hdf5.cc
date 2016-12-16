#include "ngh5.io.hdf5.hh"

#include "hdf5_enum_type.hh"
#include "edge_attributes.hh"
#include "node_attributes.hh"

using namespace std;

namespace ngh5
{
  namespace io
  {
    namespace hdf5
    {
      void test_hdf5_enum()
      {
        vector< pair<int,string> > enumeration;
        enumeration.push_back(make_pair(1, "HELLO"));
        enumeration.push_back(make_pair(2, "WORLD"));

        hid_t etype = create_H5Tenum(enumeration);
        H5Tclose(etype);

        etype = create_H5Tenum(enumeration, false);
        H5Tclose(etype);
      }

      void test_edge_attribute_write()
      {
        hid_t loc = -1;
        string path = "foo";
        vector<NODE_IDX_T> foo(20);
        vector<double> bar(10);

        write_node_attribute(loc, path, foo, bar);

        read_edge_attribute(loc, path, foo, bar);
      }

      void test_node_attribute_write()
      {
        hid_t loc = -1;
        string path = "foo";
        vector<NODE_IDX_T> foo(10);
        vector<double> bar(10);

        write_node_attribute(loc, path, foo, bar);

        read_node_attribute(loc, path, foo, bar);
      }
    }
  }
}
