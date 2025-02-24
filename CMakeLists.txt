cmake_minimum_required(VERSION 3.12)
include(CMakePrintHelpers)

project(NeuroH5)
string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWER)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

enable_language(CXX)
set (CMAKE_CXX_STANDARD 17)

if ( CMAKE_COMPILER_IS_GNUCC )
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -DMPICH_SKIP_MPICXX=1 -DOMPI_SKIP_MPICXX=1 -DH5_USE_110_API")
endif()
  
include(${PROJECT_SOURCE_DIR}/cmake/neuroh5_utils.cmake)

set(NEUROH5_VERSION 0.1.15)

cmake_policy(SET CMP0074 NEW) # enables use of HDF5_ROOT variable

# Set the build type.  Options are:
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
    #set(CMAKE_BUILD_TYPE Release)
    #set(CMAKE_BUILD_TYPE Debug)
endif()

# when build type is Debug, enable additional logging functionality
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -DNEUROH5_DEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DNEUROH5_DEBUG")

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
# set output path for tests
set(TEST_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/test)

option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_TESTS "Build tests" ON)

# find hdf5
find_hdf5()
if(HDF5_FOUND)
  message("found HDF5 version " ${HDF5_VERSION} )
  include_directories(${HDF5_INCLUDE_DIR})
  set(_hdf5_libs hdf5 hdf5_cpp)
  if (HDF5_IS_PARALLEL)
    add_definitions( "-DHDF5_IS_PARALLEL" )
  else()
    message(WARNING "HDF5 library does not have MPI support.")
  endif()
else()
  message(SEND_ERROR "Could not find HDF5 library.")
endif()


# find mpi
set(MPI_CXX_SKIP_MPICXX TRUE)
find_package(MPI )
if (MPI_C_FOUND)
  message("found MPI version: ${MPI_C_VERSION}")
  message("MPI libraries: ${MPI_C_LIBRARIES}")
else()
  message(SEND_ERROR "Could not find the MPI library. ")
endif()

# Parallel HDF5 needs to find the MPI header file
include_directories(${MPI_C_INCLUDE_DIRS})

add_library(mpi INTERFACE)
target_link_libraries(mpi INTERFACE ${MPI_C_LIBRARIES})
if(MPI_C_EXTRA_LIBRARY)
  target_link_libraries(mpi INTERFACE ${MPI_C_EXTRA_LIBRARY})
endif()
target_compile_options(mpi INTERFACE ${MPI_C_COMPILE_OPTIONS})
set_property(TARGET mpi PROPERTY
  INTERFACE_LINK_OPTIONS
  ${MPI_C_LINK_OPTIONS})


set(NEUROH5_IO_PYTHON_C_MODULE_NAME "io" CACHE STRING "Name of the C extension module")
# avoid picking system framework prematurely
set(CMAKE_FIND_FRAMEWORK LAST)
# find python
if (BUILD_PYTHON_BINDINGS)
  # set default python search strategy to location
  cmake_policy(SET CMP0094 NEW)
  find_package(Python 3 COMPONENTS Interpreter Development REQUIRED)
  cmake_print_variables(Python_EXECUTABLE)

if (NOT Python_EXECUTABLE)
  set(BUILD_PYTHON_BINDINGS OFF)
  message(SEND_ERROR "no Python found; skip building python binding")
else()
  
  # check if can use the NumPy C API
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import numpy as np; print(np.get_include(), end='')"
    RESULT_VARIABLE NUMPY_NOT_FOUND
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NUMPY_NOT_FOUND)
    message(SEND_ERROR "NumPy headers not found")
  else()
    message("found NumPy include dir " ${NUMPY_INCLUDE_DIR} )
  endif()

  # check if can use the mpi4py C API
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import mpi4py; print(mpi4py.get_include(), end='')"
    OUTPUT_VARIABLE MPI4PY_INCLUDE_DIR
    RESULT_VARIABLE MPI4PY_NOT_FOUND
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(MPI4PY_NOT_FOUND)
    message(SEND_ERROR "mpi4py headers not found")
  else()
    message("found mpi4py include dir " ${MPI4PY_INCLUDE_DIR} )
  endif()
  
#  configure_file( setup.py.tmpl setup.py )
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()
endif()


# Try to link with jemalloc [optional]
find_package(JeMalloc)

#find parmetis [optional]
find_library(PARMETIS_LIBRARY NAME libparmetis.a libparmetis.dylib
  HINTS ${METIS_DIR}/lib PATHS /usr/lib ${METIS_DIR}/lib)
message(STATUS "ParMETIS library: ${PARMETIS_LIBRARY}")
find_path(METIS_INCLUDE_DIR NAME metis.h
  HINTS ${METIS_DIR}/include PATHS /usr/include)

if(METIS_INCLUDE_DIR AND PARMETIS_LIBRARY)
  include_directories(${METIS_INCLUDE_DIR})
  set(HAVE_PARMETIS TRUE)
endif()

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
  ${PROJECT_SOURCE_DIR}/include/cell
  ${PROJECT_SOURCE_DIR}/include/graph
  ${PROJECT_SOURCE_DIR}/include/io
  ${PROJECT_SOURCE_DIR}/include/mpi
  ${PROJECT_SOURCE_DIR}/include/data
  ${PROJECT_SOURCE_DIR}/include/hdf5
  ${PROJECT_SOURCE_DIR}/include/ngraph
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

#add_subdirectory( src )

#if (BUILD_EXAMPLES)
#  add_subdirectory( examples )
#endif(BUILD_EXAMPLES)
#if (BUILD_TESTS)
#  add_subdirectory( tests )
#endif (BUILD_TESTS)
#if (BUILD_DOC)
#  add_subdirectory( doc )
#endif (BUILD_DOC)

# CMake configuration file creation
# Include module with fuction 'write_basic_package_version_file'
include(CMakePackageConfigHelpers)

file(GLOB cell_SRC ${PROJECT_SOURCE_DIR}/src/cell/*.cc)
add_library(neuroh5.cell OBJECT ${cell_SRC})

file(GLOB data_SRC ${PROJECT_SOURCE_DIR}/src/data/*.cc)
add_library(neuroh5.data OBJECT ${data_SRC})

file(GLOB graph_SRC ${PROJECT_SOURCE_DIR}/src/graph/*.cc)
if(NOT HAVE_PARMETIS)
  list(REMOVE_ITEM graph_SRC "${PROJECT_SOURCE_DIR}/src/graph/partition_graph.cc")
endif()
add_library(neuroh5.graph OBJECT ${graph_SRC})

file(GLOB hdf5_SRC ${PROJECT_SOURCE_DIR}/src/hdf5/*.cc)
add_library(neuroh5.hdf5 OBJECT ${hdf5_SRC})

file(GLOB io_SRC ${PROJECT_SOURCE_DIR}/src/io/*.cc)
add_library(neuroh5.io OBJECT ${io_SRC})

file(GLOB mpi_SRC ${PROJECT_SOURCE_DIR}/src/mpi/*.cc)
add_library(neuroh5.mpi OBJECT ${mpi_SRC})


add_executable(neurograph_reader
  ${PROJECT_SOURCE_DIR}/src/driver/neurograph_reader.cc
  $<TARGET_OBJECTS:neuroh5.cell>
  $<TARGET_OBJECTS:neuroh5.data>
  $<TARGET_OBJECTS:neuroh5.graph>
  $<TARGET_OBJECTS:neuroh5.hdf5>
  $<TARGET_OBJECTS:neuroh5.io>
  $<TARGET_OBJECTS:neuroh5.mpi>)
target_link_libraries(neurograph_reader PUBLIC ${HDF5_LIBRARIES} mpi)


add_executable(neurograph_scatter_read
  ${PROJECT_SOURCE_DIR}/src/driver/neurograph_scatter_read.cc
  $<TARGET_OBJECTS:neuroh5.cell>
  $<TARGET_OBJECTS:neuroh5.data>
  $<TARGET_OBJECTS:neuroh5.graph>
  $<TARGET_OBJECTS:neuroh5.hdf5>
  $<TARGET_OBJECTS:neuroh5.io>
  $<TARGET_OBJECTS:neuroh5.mpi>)
target_link_libraries(neurograph_scatter_read PUBLIC ${HDF5_LIBRARIES} mpi)

add_executable(balance_indegree
  ${PROJECT_SOURCE_DIR}/src/driver/balance_indegree.cc
  $<TARGET_OBJECTS:neuroh5.cell>
  $<TARGET_OBJECTS:neuroh5.data>
  $<TARGET_OBJECTS:neuroh5.graph>
  $<TARGET_OBJECTS:neuroh5.hdf5>
  $<TARGET_OBJECTS:neuroh5.io>
  $<TARGET_OBJECTS:neuroh5.mpi>)
target_link_libraries(balance_indegree PUBLIC ${HDF5_LIBRARIES} mpi)

add_executable(neurograph_vertex_metrics
  ${PROJECT_SOURCE_DIR}/src/driver/vertex_metrics.cc
  $<TARGET_OBJECTS:neuroh5.cell>
  $<TARGET_OBJECTS:neuroh5.data>
  $<TARGET_OBJECTS:neuroh5.graph>
  $<TARGET_OBJECTS:neuroh5.hdf5>
  $<TARGET_OBJECTS:neuroh5.io>
  $<TARGET_OBJECTS:neuroh5.mpi>)
target_link_libraries(neurograph_vertex_metrics PUBLIC ${HDF5_LIBRARIES} mpi)

add_executable(neurograph_import
  ${PROJECT_SOURCE_DIR}/src/driver/neurograph_import.cc
  $<TARGET_OBJECTS:neuroh5.cell>
  $<TARGET_OBJECTS:neuroh5.data>
  $<TARGET_OBJECTS:neuroh5.graph>
  $<TARGET_OBJECTS:neuroh5.hdf5>
  $<TARGET_OBJECTS:neuroh5.io>
  $<TARGET_OBJECTS:neuroh5.mpi>)
target_link_libraries(neurograph_import PUBLIC ${HDF5_LIBRARIES} mpi)

add_executable(neurotrees_copy
  ${PROJECT_SOURCE_DIR}/src/driver/neurotrees_copy.cc
  $<TARGET_OBJECTS:neuroh5.cell>
  $<TARGET_OBJECTS:neuroh5.data>
  $<TARGET_OBJECTS:neuroh5.hdf5>
  $<TARGET_OBJECTS:neuroh5.io>
  $<TARGET_OBJECTS:neuroh5.mpi>)
target_link_libraries(neurotrees_copy PUBLIC ${HDF5_LIBRARIES} mpi)

add_executable(neurotrees_select
  ${PROJECT_SOURCE_DIR}/src/driver/neurotrees_select.cc
  $<TARGET_OBJECTS:neuroh5.cell>
  $<TARGET_OBJECTS:neuroh5.data>
  $<TARGET_OBJECTS:neuroh5.hdf5>
  $<TARGET_OBJECTS:neuroh5.io>
  $<TARGET_OBJECTS:neuroh5.mpi>)
target_link_libraries(neurotrees_select PUBLIC ${HDF5_LIBRARIES} mpi)

add_executable(neurotrees_import
  ${PROJECT_SOURCE_DIR}/src/driver/neurotrees_import.cc
  $<TARGET_OBJECTS:neuroh5.cell>
  $<TARGET_OBJECTS:neuroh5.data>
  $<TARGET_OBJECTS:neuroh5.hdf5>
  $<TARGET_OBJECTS:neuroh5.io>
  $<TARGET_OBJECTS:neuroh5.mpi>)
target_link_libraries(neurotrees_import PUBLIC ${HDF5_LIBRARIES} mpi)

add_executable(neurotrees_read
  ${PROJECT_SOURCE_DIR}/src/driver/neurotrees_read.cc
  $<TARGET_OBJECTS:neuroh5.cell>
  $<TARGET_OBJECTS:neuroh5.data>
  $<TARGET_OBJECTS:neuroh5.hdf5>
  $<TARGET_OBJECTS:neuroh5.io>
  $<TARGET_OBJECTS:neuroh5.mpi>)
target_link_libraries(neurotrees_read PUBLIC ${HDF5_LIBRARIES} mpi)

add_executable(neurotrees_scatter_read
  ${PROJECT_SOURCE_DIR}/src/driver/neurotrees_scatter_read.cc
  $<TARGET_OBJECTS:neuroh5.cell>
  $<TARGET_OBJECTS:neuroh5.data>
  $<TARGET_OBJECTS:neuroh5.hdf5>
  $<TARGET_OBJECTS:neuroh5.io>
  $<TARGET_OBJECTS:neuroh5.mpi>)
target_link_libraries(neurotrees_scatter_read PUBLIC ${HDF5_LIBRARIES} mpi)

if (JeMalloc_FOUND)

target_link_libraries(balance_indegree PUBLIC ${JEMALLOC_LIBRARIES})
target_link_libraries(neurograph_vertex_metrics PUBLIC ${JEMALLOC_LIBRARIES})
target_link_libraries(neurograph_reader PUBLIC ${JEMALLOC_LIBRARIES})
target_link_libraries(neurograph_scatter_read PUBLIC ${JEMALLOC_LIBRARIES})
target_link_libraries(neurograph_import PUBLIC ${JEMALLOC_LIBRARIES})
target_link_libraries(neurotrees_select PUBLIC ${JEMALLOC_LIBRARIES})
target_link_libraries(neurotrees_copy PUBLIC ${JEMALLOC_LIBRARIES})
target_link_libraries(neurotrees_import PUBLIC ${JEMALLOC_LIBRARIES})
target_link_libraries(neurotrees_read PUBLIC ${JEMALLOC_LIBRARIES})
target_link_libraries(neurotrees_scatter_read PUBLIC ${JEMALLOC_LIBRARIES})

endif()

if (BUILD_PYTHON_BINDINGS)
  add_subdirectory(python/neuroh5 EXCLUDE_FROM_ALL)
endif()

message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Building tests: ${BUILD_TESTS}")
message(STATUS "Building documentation: ${BUILD_DOC}")
message(STATUS "Building python bindings: ${BUILD_PYTHON_BINDINGS}")
