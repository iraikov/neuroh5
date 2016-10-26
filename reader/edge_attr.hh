
#include <map>
#include <vector>
#include <memory>
#include <stdexcept>

struct EdgeAttr
{
  std::vector < std::vector <float> > float_values;
  std::vector < std::vector <uint8_t> > uint8_values;
  std::vector < std::vector <uint8_t> > uint16_values;
  std::vector < std::vector <uint8_t> > uint32_values;
  

  template<class T>
  const size_t size () const
  {
    size_t result;
    if (typeid(T) == typeid(std::vector<float>))
      result = float_values.size();
    else
      if (typeid(T) == typeid(std::vector<uint8_t>))
        result = uint8_values.size();
      else if (typeid(T) == typeid(std::vector<uint16_t>))
        result = uint16_values.size();
      else if (typeid(T) == typeid(std::vector<uint32_t>))
        result = uint32_values.size();
      else
        std::runtime_error("Unknown type for size");
    return result;
  }

  template<class T>
  const void resize (size_t size) 
  {
    if (typeid(T) == typeid(std::vector<float>))
      float_values.resize(size);
    else
      if (typeid(T) == typeid(std::vector<uint8_t>))
        uint8_values.resize(size);
      else if (typeid(T) == typeid(std::vector<uint16_t>))
        uint16_values.resize(size);
      else if (typeid(T) == typeid(std::vector<uint32_t>))
        uint32_values.resize(size);
      else
        std::runtime_error("Unknown type for resize");
  }

  template<class T>
  const std::vector<T>& get (size_t i) const
  {
    std::vector<t> result;
    if (typeid(T) == typeid(std::vector<float>))
      return(float_values[i]);
    else
      if (typeid(T) == typeid(std::vector<uint8_t>))
        return(uint8_values[i]);
      else if (typeid(T) == typeid(std::vector<uint16_t>))
          return(uint16_values[i]);
      else if (typeid(T) == typeid(std::vector<uint32_t>))
          return(uint32_values[i]);
      else
        std::runtime_error("Unknown type for get");
  }

  template<class T>
  size_t insert (const std::vector<T> &value) 
  {
    size_t index;
    if (typeid(T) == typeid(std::vector<float>))
      {
        index = float_values.size();
        float_values.push_back(value);
      }
    else
      if (typeid(T) == typeid(std::vector<uint8_t>))
        {
          index = uint8_values.size();
          uint8_values.push_back (value);
        }
      else if (typeid(T) == typeid(std::vector<uint16_t>))
        {
          index = uint16_values.size();
          uint16_values.push_back (value);
        }
      else if (typeid(T) == typeid(std::vector<uint32_t>))
        {
          index = uint32_values.size();
          uint32_values.push_back (value);
        }
      else
        std::runtime_error("Unknown type for insert");
    return index;
  }
  

  template<class T>
  void push_back (size_t vindex, T value) 
  {
    if (typeid(T) == typeid(std::vector<float>))
      {
        float_values[vindex].push_back(value);
      }
    else
      if (typeid(T) == typeid(std::vector<uint8_t>))
        {
          uint8_values[vindex].push_back (value);
        }
      else if (typeid(T) == typeid(std::vector<uint16_t>))
        {
          uint16_values[vindex].push_back (value);
        }
      else if (typeid(T) == typeid(std::vector<uint32_t>))
        {
          uint32_values[vindex].push_back (value);
        }
      else
        std::runtime_error("Unknown type for push_back");
  }
  
  

  template<class T>
  const T at (size_t vindex, size_t index) const
  {
    T result;
    if (typeid(T) == typeid(std::vector<float>))
      {
        result = float_values[vindex][index];
      }
    else
      if (typeid(T) == typeid(std::vector<uint8_t>))
        {
          result = uint8_values[vindex][index];
        }
      else if (typeid(T) == typeid(std::vector<uint16_t>))
        {
          result = uint16_values[vindex][index];
        }
      else if (typeid(T) == typeid(std::vector<uint32_t>))
        {
          result = uint32_values[vindex][index];
        }
      else
        std::runtime_error("Unknown type for push_back");
    return result;
  }
  
  
  
};

struct EdgeNamedAttr : EdgeAttr
{

  std::map<std::string, size_t> float_names;
  std::map<std::string, size_t> uint8_names;
  std::map<std::string, size_t> uint16_names;
  std::map<std::string, size_t> uint32_names;
  

  template<class T>
  size_t insert (std::string name, const std::vector<T> &value) 
  {
    size_t index;
    if (typeid(T) == typeid(std::vector<float>))
      {
        index = float_values.size();
        float_names.insert(make_pair(name, index));
        float_values.push_back(value);
      }
    else
      if (typeid(T) == typeid(std::vector<uint8_t>))
        {
          index = uint8_values.size();
          uint8_names.insert(make_pair(name, index));
          uint8_values.push_back (value);
        }
      else if (typeid(T) == typeid(std::vector<uint16_t>))
        {
          index = uint16_values.size();
          uint16_names.insert(make_pair(name, index));
          uint16_values.push_back (value);
        }
      else if (typeid(T) == typeid(std::vector<uint32_t>))
        {
          index = uint32_values.size();
          uint32_names.insert(make_pair(name, index));
          uint32_values.push_back (value);
        }
      else
        std::runtime_error("Unknown type for insert");
    return index;
  }
  
  
  
};
