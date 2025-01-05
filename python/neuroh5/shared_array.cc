// shared_array.cpp
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <memory>
#include <type_traits>

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

    T* get_data() { return data.get(); }
    size_t get_size() const { return size; }
};

// Structure to hold type information
struct TypeInfo {
    int numpy_type;
    size_t item_size;
};

// Cleanup function template
template<typename T>
static void array_dealloc(PyObject* capsule) {
    auto* holder = static_cast<SharedArrayHolder<T>*>(
        PyCapsule_GetPointer(capsule, "array_memory")
    );
    delete holder;
}

// Helper function to create array of specific type
template<typename T>
static PyObject* create_typed_array(Py_ssize_t size) {
    // Create the array holder
    auto holder = new SharedArrayHolder<T>(size);

    // Create dimensions for the numpy array
    npy_intp dims[1] = {static_cast<npy_intp>(size)};

    // Create a capsule to own the memory
    PyObject* capsule = PyCapsule_New(
        holder,
        "array_memory",
        array_dealloc<T>
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
    if (PyArray_SetBaseObject((PyArrayObject*)array, capsule) < 0) {
        Py_DECREF(capsule);
        Py_DECREF(array);
        return nullptr;
    }

    return array;
}

// Function to create the shared numpy array
static PyObject* create_shared_array(PyObject* self, PyObject* args) {
    Py_ssize_t size;
    const char* dtype_str;
    
    // Parse size and dtype arguments
    if (!PyArg_ParseTuple(args, "ns", &size, &dtype_str)) {
        return nullptr;
    }

    // Map dtype string to appropriate creation function
    std::string dtype(dtype_str);
    if (dtype == "float32" || dtype == "float") {
        return create_typed_array<float>(size);
    }
    else if (dtype == "int32") {
        return create_typed_array<int32_t>(size);
    }
    else if (dtype == "int16") {
        return create_typed_array<int16_t>(size);
    }
    else if (dtype == "int8") {
        return create_typed_array<int8_t>(size);
    }
    else if (dtype == "uint32") {
        return create_typed_array<uint32_t>(size);
    }
    else if (dtype == "uint16") {
        return create_typed_array<uint16_t>(size);
    }
    else if (dtype == "uint8") {
        return create_typed_array<uint8_t>(size);
    }
    
    PyErr_SetString(PyExc_ValueError, "Unsupported dtype");
    return nullptr;
}

// Module method definition
static PyMethodDef ModuleMethods[] = {
    {"create_shared_array", create_shared_array, METH_VARARGS, 
     "Create a numpy array that shares memory with C++ array. Args: size, dtype"},
    {nullptr, nullptr, 0, nullptr}
};

// Module definition structure
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "shared_array_module",
    "Module that provides shared memory array between C++ and Python",
    -1,
    ModuleMethods
};

// Module initialization function
PyMODINIT_FUNC PyInit_shared_array_module(void) {
    import_array();  // Initialize numpy C-API
    
    PyObject* module = PyModule_Create(&moduledef);
    if (!module) {
        return nullptr;
    }

    return module;
}
