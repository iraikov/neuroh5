#!/usr/bin/env python
import os, sys
import numpy as np
from distutils.core import setup, Extension

if sys.platform == 'darwin':
    HDF5_INCDIR = os.environ.get("HDF5_INCDIR", "/usr/local/hdf5/include")
    HDF5_LIBDIR = os.environ.get("HDF5_LIBDIR", "/usr/local/hdf5/lib")
    HDF5_LIB    = os.environ.get("HDF5_LIB", "hdf5")
    MPI_INCDIR  = os.environ.get("MPI_INCDIR", "/usr/local/Cellar/mpich/3.2_2/include")
    MPI_LIB     = os.environ.get("MPI_LIB", "mpich")
    NUMPY_INCDIR=np.get_include()
    extra_compile_args = ["-std=c++11",
                          "-stdlib=libc++",	
                          "-mmacosx-version-min=10.9",
                          "-UNDEBUG",
                          "-I"+HDF5_INCDIR,
                          "-I"+MPI_INCDIR,
                          "-I"+NUMPY_INCDIR,
                          "-I.",
                          "-g"]
    extra_link_args=["-L"+HDF5_LIBDIR, "-L"+MPI_LIBDIR]
    libraries = [HDF5_LIB, MPI_LIB]
else:
    HDF5_INCDIR = os.environ.get("HDF5_INCDIR", "/usr/include/hdf5/mpich")
    HDF5_LIBDIR = os.environ.get("HDF5_LIBDIR", "/usr/lib")
    HDF5_LIB    = os.environ.get("HDF5_LIB", "hdf5_mpich")
    MPI_INCDIR  = os.environ.get("MPI_INCDIR", "/usr/include/mpich")
    MPI_LIBDIR  = os.environ.get("MPI_LIBDIR", "/usr/lib")
    MPI_LIB     = os.environ.get("MPI_LIB", "mpich")
    NUMPY_INCDIR= os.environ.get("NUMPY_INCDIR", np.get_include())
    extra_compile_args = ["-std=c++11",
                          "-UNDEBUG",
                          "-I"+HDF5_INCDIR,
                          "-I"+MPI_INCDIR,
                          "-I"+MPI_LIBDIR,
                          "-I"+NUMPY_INCDIR,
                          "-I.",
                          "-g"]
    extra_link_args = ["-L"+HDF5_LIBDIR, "-L"+MPI_LIBDIR]
    if MPI_LIB != "":
        libraries = [HDF5_LIB, MPI_LIB]
    else:
        libraries = [HDF5_LIB]


HDF5_INCDIR = os.environ.get("HDF5_INCDIR", "/usr/include/hdf5/mpich")
HDF5_LIBDIR = os.environ.get("HDF5_LIBDIR", "/usr/lib")
HDF5_LIB    = os.environ.get("HDF5_LIB", "hdf5_mpich")
MPI_INCDIR  = os.environ.get("MPI_INCDIR", "/usr/include/mpich")
MPI_LIBDIR  = os.environ.get("MPI_LIBDIR", "/usr/lib")
MPI_LIB     = os.environ.get("MPI_LIB", "mpich")
NUMPY_INCDIR= os.environ.get("NUMPY_INCDIR", np.get_include())
extra_compile_args = ["-std=c++11",
                      "-UNDEBUG",
                      "-I"+HDF5_INCDIR,
                      "-I"+MPI_INCDIR,
                      "-I"+MPI_LIBDIR,
                      "-I"+NUMPY_INCDIR,
                      "-Iinclude", "-Iinclude/cell", "-Iinclude/graph", "-Iinclude/ngraph",
                      "-Iinclude/data", "-Iinclude/mpi", "-Iinclude/hdf5",
                      "-g"]
extra_link_args = ["-L"+HDF5_LIBDIR, "-L"+MPI_LIBDIR]
if MPI_LIB != "":
    libraries = [HDF5_LIB, MPI_LIB]
else:
    libraries = [HDF5_LIB]

NUMPY_INCDIR= os.environ.get("NUMPY_INCDIR", np.get_include())

setup(
    name='NeuroH5',
    package_dir = {'': 'python'},
    packages = ["neuroh5"],
    version='0.0.1',
    maintainer = "Ivan Raikov",
    maintainer_email = "ivan.g.raikov@gmail.com",
    description = "NeuroH5 library",
    url = "http://github.com/gheber/neurographdf5",
    include_package_data=True,
    install_requires=[
        'click', 'h5py', 'numpy'
    ],
    entry_points='''
        [console_scripts]
        importdbs=neuroh5.importdbs:cli
        initrange=neuroh5.initrange:cli
        importcoords=neuroh5.importcoords:cli
    ''',
    ext_modules = [
        Extension('neuroh5.io',
                  extra_compile_args = extra_compile_args,
                  extra_link_args = extra_link_args,
                  libraries = libraries,
                  define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
                  sources = [
                      'src/data/append_rank_tree_map.cc',
                      'src/data/attr_map.cc',
                      'src/data/attr_val.cc',
                      'src/cell/read_tree.cc',
                      'src/cell/cell_populations.cc',
                      'src/cell/validate_tree.cc',
                      'src/cell/contract_tree.cc',
                      'src/cell/scatter_read_tree.cc',
                      'src/cell/cell_index.cc',
                      'src/cell/append_tree.cc',
                      'src/cell/cell_attributes.cc',
                      'src/graph/scatter_graph.cc',
                      'src/graph/edge_attributes.cc',
                      'src/graph/read_graph.cc',
                      'src/graph/write_projection.cc',
                      'src/graph/read_projection.cc',
                      'src/graph/projection_names.cc',
                      'src/graph/merge_edge_map.cc',
                      'src/graph/validate_edge_list.cc',
                      'src/graph/balance_graph_indegree.cc',
                      'src/graph/node_attributes.cc',
                      'src/graph/compute_vertex_metrics.cc',
                      'src/graph/vertex_degree.cc',
                      'src/graph/bcast_graph.cc',
                      'src/graph/write_graph.cc',
                      'src/mpi/alltoallv_packed.cc',
                      'src/mpi/pack_edge.cc',
                      'src/mpi/rank_range.cc',
                      'src/mpi/bcast_string_vector.cc',
                      'src/mpi/pack_tree.cc',
                      'src/hdf5/dataset_num_elements.cc',
                      'src/hdf5/read_link_names.cc',
                      'src/hdf5/exists_h5types.cc',
                      'src/hdf5/copy_h5types.cc',
                      'src/hdf5/create_file_toplevel.cc',
                      'src/hdf5/path_names.cc',
                      'src/hdf5/group_contents.cc',
                      'src/hdf5/dataset_type.cc',
                      'src/hdf5/create_group.cc',
                      'src/hdf5/file_access.cc',
                      'python/neuroh5/iomodule.cc'
                  ])
        ]
    )



