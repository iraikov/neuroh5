
#include "shared_array.hh"

// Function to create shared numpy array
static PyObject* create_shared_array(PyObject* self, PyObject* args)
{
    Py_ssize_t size;
    const char* dtype_str;
    
    // Parse size and dtype arguments
    if (!PyArg_ParseTuple(args, "ns", &size, &dtype_str)) {
        return nullptr;
    }

    // Map dtype string to appropriate creation function
    std::string dtype(dtype_str);
    if (dtype == "float32" || dtype == "float") {
        return create_typed_shared_array<float>(size);
    }
    else if (dtype == "int32") {
        return create_typed_shared_array<int32_t>(size);
    }
    else if (dtype == "int16") {
        return create_typed_shared_array<int16_t>(size);
    }
    else if (dtype == "int8") {
        return create_typed_shared_array<int8_t>(size);
    }
    else if (dtype == "uint32") {
        return create_typed_shared_array<uint32_t>(size);
    }
    else if (dtype == "uint16") {
        return create_typed_shared_array<uint16_t>(size);
    }
    else if (dtype == "uint8") {
        return create_typed_shared_array<uint8_t>(size);
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
