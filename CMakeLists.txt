cmake_minimum_required(VERSION 3.1)

project(NeuroH5)
string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWER)

enable_language(CXX)
set (CMAKE_CXX_STANDARD 11)
if ( CMAKE_COMPILER_IS_GNUCC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall")
endif()
  
include(${PROJECT_SOURCE_DIR}/cmake/neuroh5_utils.cmake)
set(NEUROH5_VERSION 1.0.0)

# Set the build type.  Options are:
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries

if (NOT CMAKE_BUILD_TYPE)
    #set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
    #set(CMAKE_BUILD_TYPE Debug)
endif()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
# set output path for tests
set(TEST_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/test)

option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_TESTS "Build tests" ON)

# find python
if (BUILD_PYTHON_BINDINGS)
find_package(PythonInterp 2.7 REQUIRED)
if (NOT PYTHON_EXECUTABLE)
  set(BUILD_PYTHON_BINDINGS OFF)
else()
  configure_file( setup.py.tmpl setup.py )
endif()
endif()

# check if can use the NumPy C API
exec_program(${PYTHON_EXECUTABLE}
             ARGS "-c \"import numpy as np; print(np.get_include())\""
             OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
             RETURN_VALUE NUMPY_NOT_FOUND
            )
if(NUMPY_NOT_FOUND)
  message(SEND_ERROR "NumPy headers not found")
else()
  message("found NumPy include dir " ${NUMPY_INCLUDE_DIR} )
endif()

# find hdf5
find_hdf5()
if(HDF5_FOUND)
  message("found HDF5 version " ${HDF5_VERSION} )
  if (HDF5_IS_PARALLEL)
    include_directories(${HDF5_INCLUDE_DIR})
    set(_hdf5_libs hdf5 hdf5_cpp)
  else()
      message(SEND_ERROR "Could not find HDF5 library with MPI support.")
  endif()
else()
  message(SEND_ERROR "Could not find HDF5 library.")
endif()


# find mpi
find_package(MPI)
if (MPI_FOUND)
  message("found MPI version " ${MPI_VERSION} )
else()
  message(SEND_ERROR "Could not find the MPI library. Define the MPI_COMPILER variable to the path of your MPI compiler.")
endif()

# Parallel HDF5 needs to find the "mpi.h" header file
include_directories(${MPI_INCLUDE_PATH})

if (BUILD_TESTS)
find_package(Threads)
find_package(GTest)
if (NOT GTEST_FOUND)
	message(WARNING "gtest library not found, tests will not be run")
endif()
if(NOT MSVC)
  set(PThreadLib -pthread)
endif()
endif()


#set the C/C++ include path to the "include" directory
include_directories(BEFORE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include/data
  ${PROJECT_SOURCE_DIR}/include/mpi
  ${PROJECT_SOURCE_DIR}/include/lowio
  ${PROJECT_SOURCE_DIR}/include/io
  ${PROJECT_SOURCE_DIR}/include/test)


# require proper c++
#add_definitions( "-Wall -ansi -pedantic" )
# HDF5 uses long long which is not ansi
if(CMAKE_C_COMPILER_ID MATCHES "MSVC" OR CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # lots of warnings with cl.exe right now, use /W1
    add_definitions("/W1 -D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS /bigobj")
else()
    add_definitions( "-Wall -Wno-unknown-pragmas -Wno-unused-function" )
endif()

add_subdirectory( src )

#if (BUILD_EXAMPLES)
#  add_subdirectory( examples )
#endif(BUILD_EXAMPLES)
if (BUILD_TESTS)
  add_subdirectory( tests )
endif (BUILD_TESTS)
#if (BUILD_DOC)
#  add_subdirectory( doc )
#endif (BUILD_DOC)

# CMake configuration file creation
# Include module with fuction 'write_basic_package_version_file'
include(CMakePackageConfigHelpers)

# RPM
find_program(RPM_PROGRAM rpm)
if(EXISTS ${RPM_PROGRAM})
  list(APPEND CPACK_GENERATOR "RPM")
endif(EXISTS ${RPM_PROGRAM})
# DEB
find_program(DPKG_PROGRAM dpkg)
if(EXISTS ${DPKG_PROGRAM})
  list(APPEND CPACK_GENERATOR "DEB")
endif(EXISTS ${DPKG_PROGRAM})
# NSIS
find_program(NSIS_PROGRAM makensis MakeNSIS)
if(EXISTS ${NSIS_PROGRAM})
  list(APPEND CPACK_GENERATOR "NSIS")
endif(EXISTS ${NSIS_PROGRAM})
# dpkg
find_program(PACKAGE_MAKER_PROGRAM PackageMaker
	    HINTS /Developer/Applications/Utilities)
if(EXISTS ${PACKAGE_MAKER_PROGRAM})
  list(APPEND CPACK_GENERATOR "PackageMaker")
endif(EXISTS ${PACKAGE_MAKER_PROGRAM})

message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Building tests: ${BUILD_TESTS}")
message(STATUS "Building documentation: ${BUILD_DOC}")
message(STATUS "Building python bindings: ${BUILD_PYTHON_BINDINGS}")
