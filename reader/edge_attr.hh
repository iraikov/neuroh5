
#include <vector>
#include <memory>
#include <stdexcept>

struct EdgeAttr
{
  enum active_type
    {
      at_null,
      at_float,
      at_uint8,
      at_uint16,
      at_uint32
      
    } tag_active_type = at_null;
  
  EdgeAttr()      : tag_active_type(at_null)  { }
  EdgeAttr(const std::vector <float> value)      : tag_active_type(at_float), data_v(value)  { }
  EdgeAttr(const std::vector <uint8_t> value)    : tag_active_type(at_uint8), data_v(value)  { }
  EdgeAttr(const std::vector <uint16_t> value)   : tag_active_type(at_uint16), data_v(value) { }
  EdgeAttr(const std::vector <uint32_t> value)   : tag_active_type(at_uint32), data_v(value) { }
  ~EdgeAttr() { delete_data(); }
  
  union data
  {
    data() : null(0) { }
    data(const std::vector <float> value)    : float_v(new std::vector <float> (value)) { }
    data(const std::vector <uint8_t> value)  : uint8_v(new std::vector <uint8_t> (value)) { }
    data(const std::vector <uint16_t> value) : uint16_v(new std::vector <uint16_t> (value)) { }
    data(const std::vector <uint32_t> value) : uint32_v(new std::vector <uint32_t> (value)) { }

    int null;
    std::vector <float>    *float_v;
    std::vector <uint8_t>  *uint8_v;
    std::vector <uint16_t> *uint16_v;
    std::vector <uint32_t> *uint32_v;
    
  } data_v;
  
  void delete_data ()
  {
    switch (tag_active_type)
      {
      case at_null:     break;
      case at_float:    delete this->data_v.float_v; break;
      case at_uint8:    delete this->data_v.uint8_v; break;
      case at_uint16:   delete this->data_v.uint16_v; break;
      case at_uint32:   delete this->data_v.uint32_v; break;
      }
      
  }

  template<class T>
  const T& get () 
  {
    switch (tag_active_type)
      {
      at_float: 
        if (typeid(T) == typeid(std::vector<float>))
          return(*(this->data_v.float_v));
        else throw std::domain_error("Non-matching types for get float");
        break;
      at_uint8:
        if (typeid(T) == typeid(std::vector<uint8_t>))
          return(*(this->data_v.uint8_v));
        else throw std::domain_error("Non-matching types for get uint8");
        break;
      at_uint16:
        if (typeid(T) == typeid(std::vector<uint16_t>))
          return(*(this->data_v.uint16_v));
        else throw std::domain_error("Non-matching types for get uint16");
        break;
      at_uint32:
        if (typeid(T) == typeid(std::vector<uint32_t>))
          return(*(this->data_v.uint32_v));
        else throw std::domain_error("Non-matching types for get uint32");
        break;
      at_null:
        std::domain_error("Null type for get");
        break;
      }
  }

  void set (const std::vector <float> &value)
  {
    delete_data();
    this->data_v.float_v = new std::vector <float> (value);
    tag_active_type = at_float;
  }

  void set (const std::vector <uint8_t> &value)
  {
    delete_data();
    this->data_v.uint8_v = new std::vector <uint8_t> (value);
    tag_active_type = at_uint8;
  }

  void set (const std::vector <uint16_t> &value)
  {
    delete_data();
    this->data_v.uint16_v = new std::vector <uint16_t> (value);
    tag_active_type = at_uint16;
  }

  void set (const std::vector <uint32_t> &value)
  {
    delete_data();
    this->data_v.uint32_v = new std::vector <uint32_t> (value);
    tag_active_type = at_uint32;
  }

  template<class T>
  void push_back (const T &value) 
  {
    switch (tag_active_type)
      {
      case at_float: 
        if (typeid(T) == typeid(std::vector<float>))
          data_v.float_v->push_back(value);
        else throw std::domain_error("Non-matching types for push_back float");
        break;
      case at_uint8:
        if (typeid(T) == typeid(std::vector<uint8_t>))
          data_v.uint8_v->push_back(value);
        else throw std::domain_error("Non-matching types for push_back uint8");
        break;
      case at_uint16:
        if (typeid(T) == typeid(std::vector<uint16_t>))
          data_v.uint16_v->push_back(value);
        else throw std::domain_error("Non-matching types for push_back uint16");
        break;
      case at_uint32:
        if (typeid(T) == typeid(std::vector<uint32_t>))
          data_v.uint32_v->push_back(value);
        else throw std::domain_error("Non-matching types for push_back uint32");
        break;
      case at_null:
        std::domain_error("Null type for push_back");
        break;
      }
  }

  template<class T>
  T at (typename std::vector<T>::size_type i) const
  {
    T result;
    switch (tag_active_type)
      {
      case at_float: 
        if (typeid(T) == typeid(std::vector<float>))
          result = data_v.float_v->at(i);
        else throw std::domain_error("Non-matching types for at float");
        break;
      case at_uint8:
        if (typeid(T) == typeid(std::vector<uint8_t>))
          result = data_v.uint8_v->at(i);
        else throw std::domain_error("Non-matching types for at uint8");
        break;
      case at_uint16:
        if (typeid(T) == typeid(std::vector<uint16_t>))
          result = data_v.uint16_v->at(i);
        else throw std::domain_error("Non-matching types for at uint16");
        break;
      case at_uint32:
        if (typeid(T) == typeid(std::vector<uint32_t>))
          result = data_v.uint32_v->at(i);
        else throw std::domain_error("Non-matching types for at uint32");
        break;
      case at_null:
        std::domain_error("Null type for push_back");
        break;
      }
    return result;
  }
  
  
  
};
