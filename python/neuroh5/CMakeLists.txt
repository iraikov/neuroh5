set(NEUROH5_IO_SRCS
    "iomodule.cc"
)

include_directories(${MPI_C_INCLUDE_DIRS})

add_library(python_neuroh5_io SHARED
  $<TARGET_OBJECTS:neuroh5.cell>
  $<TARGET_OBJECTS:neuroh5.data>
  $<TARGET_OBJECTS:neuroh5.graph>
  $<TARGET_OBJECTS:neuroh5.hdf5>
  $<TARGET_OBJECTS:neuroh5.io>
  $<TARGET_OBJECTS:neuroh5.mpi>
  ${NEUROH5_IO_SRCS})

set_target_properties(
    python_neuroh5_io
    PROPERTIES
        PREFIX ""
        OUTPUT_NAME ${NEUROH5_IO_PYTHON_C_MODULE_NAME}
        LINKER_LANGUAGE C
)

if(WIN32)
    set_target_properties(
        python_neuroh5_io
        PROPERTIES
        SUFFIX ".pyd"
    )
endif()

target_include_directories(python_neuroh5_io PUBLIC
    ${PYTHON_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIR}
    ${MPI4PY_INCLUDE_DIR}
    ${MPI_C_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    )
target_link_libraries(python_neuroh5_io PUBLIC ${HDF5_LIBRARIES} mpi)

# On Windows and Mac OS, it is required to link to the Python libraries
if(WIN32)
    target_link_libraries(python_neuroh5_io ${PYTHON_LIBRARIES})
endif()
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    target_link_libraries(python_neuroh5_io ${PYTHON_LIBRARIES})
endif()

