#ifndef NEUROH5_SHARED_ARRAY_HH
#define NEUROH5_SHARED_ARRAY_HH

#include <vector>
#include <deque>
#include <forward_list>
#include <algorithm>
#include <iterator>
#include <memory>
#include <memory>
#include <type_traits>
#include <cstring>

#include <Python.h>
#include <numpy/arrayobject.h>


// Type mapping struct to convert C++ types to NumPy types
template<typename T>
struct NumpyTypeMap {
    static constexpr int type_num = -1;  // Invalid default
};

// Specializations for supported types
template<> struct NumpyTypeMap<float> { static constexpr int type_num = NPY_FLOAT; };
template<> struct NumpyTypeMap<int32_t> { static constexpr int type_num = NPY_INT32; };
template<> struct NumpyTypeMap<int16_t> { static constexpr int type_num = NPY_INT16; };
template<> struct NumpyTypeMap<int8_t> { static constexpr int type_num = NPY_INT8; };
template<> struct NumpyTypeMap<uint32_t> { static constexpr int type_num = NPY_UINT32; };
template<> struct NumpyTypeMap<uint16_t> { static constexpr int type_num = NPY_UINT16; };
template<> struct NumpyTypeMap<uint8_t> { static constexpr int type_num = NPY_UINT8; };

// Structure to hold type information
struct TypeInfo {
    int numpy_type;
    size_t item_size;
};

// Template class to manage array memory
template<typename T>
class SharedArrayHolder {
private:
    std::unique_ptr<T[]> data;
    size_t size;

public:
    SharedArrayHolder(size_t n) : size(n) {
        data = std::make_unique<T[]>(n);
        // Initialize array with sequential values
        for (size_t i = 0; i < n; i++) {
            data[i] = static_cast<T>(i);
        }
    }
  
  // Approach 1: Constructor that takes ownership of vector's data
  explicit SharedArrayHolder(std::vector<T>&& vec) : size(vec.size())
  {
    // Get the vector's allocator
    auto alloc = vec.get_allocator();
    
    // Get pointer to vector's data
    T* ptr = vec.data();
    size = vec.size();
    
    // Release ownership from vector (C++17)
    vec.release();
    
    // Take ownership of the pointer
    data.reset(ptr);
  }

  // Approach 2: Construct array from pointer and size
  SharedArrayHolder(T* ptr, size_t n) : data(ptr), size(n) {}

  // Approach 3: Constructor that copies vector's buffer
  explicit SharedArrayHolder(const std::vector<T>& vec) : size(vec.size())
  {
    // Allocate new buffer
    data = std::make_unique<T[]>(size);
    // Use memcpy for POD types (more efficient than std::copy)
    if (std::is_trivially_copyable_v<T>)
      {
        std::memcpy(data.get(), vec.data(), size * sizeof(T));
      } else
      {
        std::copy(vec.begin(), vec.end(), data.get());
      }
  }

  // Constructor for deque input
  explicit SharedArrayHolder(const std::deque<T>& deq) : size(deq.size())
  {
    data = std::make_unique<T[]>(size);
    
    if constexpr (std::is_trivially_copyable_v<T>) {
      // For POD types, copy each chunk efficiently
      size_t pos = 0;
      for (auto it = deq.begin(); it != deq.end(); ++it) {
        data[pos++] = *it;
      }
    } else {
      // For non-POD types, use standard copy
      std::copy(deq.begin(), deq.end(), data.get());
    }
  }

  // Move constructor for deque
  explicit SharedArrayHolder(std::deque<T>&& deq) : size(deq.size())
  {
    data = std::make_unique<T[]>(size);
    
    if constexpr (std::is_trivially_copyable_v<T>) {
      // For POD types, move each element efficiently
      size_t pos = 0;
      while (!deq.empty()) {
        data[pos++] = std::move(deq.front());
        deq.pop_front();
      }
    } else {
      // For non-POD types, use move iterator
      std::move(deq.begin(), deq.end(), data.get());
      deq.clear();
    }
  }
  
  T* get_data() { return data.get(); }
  size_t get_size() const { return size; }
};



// Cleanup function template
template<typename T>
static void shared_array_dealloc(PyObject* capsule) {
    auto* holder = static_cast<SharedArrayHolder<T>*>(
        PyCapsule_GetPointer(capsule, "array_memory")
    );
    delete holder;
}

// Helper function to create shared numpy array of specific type
template<typename T>
static PyObject* create_typed_shared_array(Py_ssize_t size)
{
    // Create the array holder
    auto holder = new SharedArrayHolder<T>(size);

    // Create dimensions for the numpy array
    npy_intp dims[1] = {static_cast<npy_intp>(size)};

    // Create a capsule to own the memory
    PyObject* capsule = PyCapsule_New(
        holder,
        "array_memory",
        shared_array_dealloc<T>
    );

    if (!capsule) {
        delete holder;
        return nullptr;
    }

    // Create the numpy array
    PyObject* array = PyArray_NewFromDescr(
        &PyArray_Type,
        PyArray_DescrFromType(NumpyTypeMap<T>::type_num),
        1,              // nd
        dims,           // dimensions
        nullptr,        // strides
        holder->get_data(),  // data
        NPY_ARRAY_WRITEABLE, // flags
        nullptr         // obj
    );

    if (!array) {
        Py_DECREF(capsule);
        return nullptr;
    }

    // Set the array's base object to our capsule
    if (PyArray_SetBaseObject((PyArrayObject*)array, capsule) < 0)
      {
        Py_DECREF(capsule);
        Py_DECREF(array);
        return nullptr;
      }

    return array;
}


// Create shared array from deque (by const reference)
template<typename T>
static PyObject* create_shared_array_from_deque(const std::deque<T>& deq)
{
    auto holder = new SharedArrayHolder<T>(deq);
    
    npy_intp dims[1] = {static_cast<npy_intp>(holder->get_size())};
    
    PyObject* capsule = PyCapsule_New(
        holder,
        "array_memory",
        shared_array_dealloc<T>
    );

    if (!capsule) {
        delete holder;
        return nullptr;
    }

    PyObject* array = PyArray_NewFromDescr(
        &PyArray_Type,
        PyArray_DescrFromType(NumpyTypeMap<T>::type_num),
        1,
        dims,
        nullptr,
        holder->get_data(),
        NPY_ARRAY_WRITEABLE,
        nullptr
    );

    if (!array)
      {
        Py_DECREF(capsule);
        return nullptr;
      }

    if (PyArray_SetBaseObject((PyArrayObject*)array, capsule) < 0)
      {
        Py_DECREF(capsule);
        Py_DECREF(array);
        return nullptr;
      }

    return array;
}



// Create shared array from deque (by rvalue reference)
template<typename T>
static PyObject* create_shared_array_from_deque(std::deque<T>&& deq)
{
    auto holder = new SharedArrayHolder<T>(std::move(deq));

    npy_intp dims[1] = {static_cast<npy_intp>(holder->get_size())};
    
    PyObject* capsule = PyCapsule_New(
        holder,
        "array_memory",
        shared_array_dealloc<T>
    );

    if (!capsule)
      {
        delete holder;
        return nullptr;
      }

    PyObject* array = PyArray_NewFromDescr(
        &PyArray_Type,
        PyArray_DescrFromType(NumpyTypeMap<T>::type_num),
        1,
        dims,
        nullptr,
        holder->get_data(),
        NPY_ARRAY_WRITEABLE,
        nullptr
    );

    if (!array) {
        Py_DECREF(capsule);
        return nullptr;
    }

    if (PyArray_SetBaseObject((PyArrayObject*)array, capsule) < 0) {
        Py_DECREF(capsule);
        Py_DECREF(array);
        return nullptr;
    }

    return array;
}

// Function to create shared numpy array from existing C++ vector using move semantics
template<typename T>
static PyObject* create_shared_array_from_vector(std::vector<T>&& vec)
{

  // Create the array holder using move constructor
  auto holder = new SharedArrayHolder<T>(std::move(vec));
    
  npy_intp dims[1] = {static_cast<npy_intp>(holder->get_size())};
    
  PyObject* capsule = PyCapsule_New(
        holder,
        "array_memory",
        shared_array_dealloc<T>
    );

    if (!capsule) {
        delete holder;
        return nullptr;
    }

    PyObject* array = PyArray_NewFromDescr(
        &PyArray_Type,
        PyArray_DescrFromType(NumpyTypeMap<T>::type_num),
        1,
        dims,
        nullptr,
        holder->get_data(),
        NPY_ARRAY_WRITEABLE,
        nullptr
    );

    if (!array) {
        Py_DECREF(capsule);
        return nullptr;
    }

    if (PyArray_SetBaseObject((PyArrayObject*)array, capsule) < 0) {
        Py_DECREF(capsule);
        Py_DECREF(array);
        return nullptr;
    }

    return array;
}


// Function to create shared numpy array from existing C++ vector by const reference
template<typename T>
static PyObject* create_shared_array_from_vector(const std::vector<T>& vec)
{

  // Create the array holder using move constructor
  auto holder = new SharedArrayHolder<T>(vec);
    
  npy_intp dims[1] = {static_cast<npy_intp>(holder->get_size())};
    
  PyObject* capsule = PyCapsule_New(
        holder,
        "array_memory",
        shared_array_dealloc<T>
    );

    if (!capsule) {
        delete holder;
        return nullptr;
    }

    PyObject* array = PyArray_NewFromDescr(
        &PyArray_Type,
        PyArray_DescrFromType(NumpyTypeMap<T>::type_num),
        1,
        dims,
        nullptr,
        holder->get_data(),
        NPY_ARRAY_WRITEABLE,
        nullptr
    );

    if (!array) {
        Py_DECREF(capsule);
        return nullptr;
    }

    if (PyArray_SetBaseObject((PyArrayObject*)array, capsule) < 0) {
        Py_DECREF(capsule);
        Py_DECREF(array);
        return nullptr;
    }

    return array;
}

#endif

