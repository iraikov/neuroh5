macro(neuroh5_add_gtest exe)
    # add build target
    add_executable(${exe} EXCLUDE_FROM_ALL ${ARGN})
    target_link_libraries(${exe} ${GTEST_LIBRARIES})
    # add dependency to 'tests' target
    add_dependencies(neuroh5_gtests ${exe})

    # add target for running test
    string(REPLACE "/" "_" _testname ${exe})
    add_custom_target(test_${_testname}
                    COMMAND ${exe}
                    ARGS --gtest_print_time
                    DEPENDS ${exe}
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test
                    VERBATIM
                    COMMENT "Runnint gtest test(s) ${exe}")
                  
    # add dependency to 'test' target
    add_dependencies(neuroh5_gtest test_${_testname})
    
endmacro(neuroh5_add_gtest)

macro(neuroh5_add_pyunit file)
    # find test file
    set(_file_name _file_name-NOTFOUND)
    find_file(_file_name ${file} ${CMAKE_CURRENT_SOURCE_DIR})
    if(NOT _file_name)
        message(FATAL_ERROR "Can't find pyunit file \"${file}\"")
    endif(NOT _file_name)

    # add target for running test
    string(REPLACE "/" "_" _testname ${file})
    add_custom_target(pyunit_${_testname}
                    COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/bin/run_test.py ${_file_name}
                    DEPENDS ${_file_name}
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test
                    VERBATIM
                    COMMENT "Running pyunit test(s) ${file}" )
    # add dependency to 'test' target
    add_dependencies(pyunit_${_testname} neuroh5)
    add_dependencies(test pyunit_${_testname})
endmacro(neuroh5_add_pyunit)

# workaround a FindHDF5 bug
macro(find_hdf5)
    set(HDF5_PREFER_PARALLEL TRUE)
    find_package(HDF5)
    set( HDF5_IS_PARALLEL FALSE )
    foreach( _dir ${HDF5_INCLUDE_DIRS} )
        if( EXISTS "${_dir}/H5pubconf.h" )
            file( STRINGS "${_dir}/H5pubconf.h"
                HDF5_HAVE_PARALLEL_DEFINE
                REGEX "HAVE_PARALLEL 1" )
            if( HDF5_HAVE_PARALLEL_DEFINE )
                set( HDF5_IS_PARALLEL TRUE )
            endif()
        endif()
    endforeach()
    # Fallback: on systems where find_package(HDF5) takes the compiler-
    # wrapper "no interrogate" path (e.g. the Cray PE `cc`/`CC` wrappers),
    # HDF5_INCLUDE_DIRS is left empty, so the scan above cannot determine
    # whether HDF5 was built with MPI support. Probe H5pubconf.h directly
    # under HDF5_ROOT / HDF5_DIR / HDF5_INCLUDEDIR as a backstop.
    if( NOT HDF5_IS_PARALLEL )
        set( _hdf5_probe_paths
            "$ENV{HDF5_INCLUDEDIR}/H5pubconf.h"
            "$ENV{HDF5_ROOT}/include/H5pubconf.h"
            "$ENV{HDF5_DIR}/include/H5pubconf.h"
            "${HDF5_ROOT}/include/H5pubconf.h"
            "${HDF5_DIR}/include/H5pubconf.h" )
        foreach( _hdr ${_hdf5_probe_paths} )
            if( _hdr AND EXISTS "${_hdr}" )
                file( STRINGS "${_hdr}"
                    HDF5_HAVE_PARALLEL_DEFINE
                    REGEX "HAVE_PARALLEL 1" )
                if( HDF5_HAVE_PARALLEL_DEFINE )
                    set( HDF5_IS_PARALLEL TRUE )
                    break()
                endif()
            endif()
        endforeach()
    endif()
    # Allow a build-time override (e.g. -DHDF5_IS_PARALLEL=TRUE or via the
    # environment) to win over auto-detection.
    if( DEFINED ENV{HDF5_IS_PARALLEL} )
        set( HDF5_IS_PARALLEL "$ENV{HDF5_IS_PARALLEL}" )
    endif()
    set( HDF5_IS_PARALLEL ${HDF5_IS_PARALLEL} CACHE BOOL
        "HDF5 library compiled with parallel IO support" FORCE )
    mark_as_advanced( HDF5_IS_PARALLEL )
endmacro(find_hdf5)
  
