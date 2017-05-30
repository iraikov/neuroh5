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
                      "-Iinclude", "-Iinclude/graph", "-Iinclude/model", "-Iinclude/mpi",
                      "-Iinclude/io", "-Iinclude/io/hdf5",
                      "-g"]
extra_link_args = ["-L"+HDF5_LIBDIR, "-L"+MPI_LIBDIR]
if MPI_LIB != "":
    libraries = [HDF5_LIB, MPI_LIB]
else:
    libraries = [HDF5_LIB]

NUMPY_INCDIR= os.environ.get("NUMPY_INCDIR", np.get_include())

setup(
    name='neurograph',
    package_dir = {'': 'python'},
    packages = ["neurograph"],
    version='0.0.1',
    maintainer = "Ivan Raikov",
    maintainer_email = "ivan.g.raikov@gmail.com",
    description = "Neurograph library",
    url = "http://github.com/gheber/neurographdf5",
    include_package_data=True,
    install_requires=[
        'click', 'h5py', 
    ],
    entry_points='''
        [console_scripts]
        importdbs=neurograph.importdbs:cli
        importpairs=neurograph.importpairs:cli
    ''',
    ext_modules = [
        Extension('neurograph.io',
                  extra_compile_args = extra_compile_args,
                  extra_link_args = extra_link_args,
                  libraries = libraries,
                  sources = [
                      'src/io/hdf5/read_dbs_projection.cc',
                      'src/io/hdf5/ngh5.io.hdf5.cc',
                      'src/io/hdf5/projection_names.cc',
                      'src/io/hdf5/dataset_num_elements.cc',
                      'src/io/hdf5/hdf5_path_names.cc',
                      'src/io/hdf5/population_reader.cc',
                      'src/io/hdf5/read_link_names.cc',
                      'src/io/hdf5/edge_attributes.cc',
                      'src/io/ngh5.io.cc',
                      'src/io/read_population.cc',
                      'src/graph/read_graph.cc',
                      'src/graph/scatter_graph.cc',
                      'src/graph/bcast_graph.cc',
                      'src/graph/validate_edge_list.cc',
                      'src/model/edge_attr.cc',
                      'src/mpi/bcast_string_vector.cc',
                      'src/mpi/pack_edge.cc',
                      'src/python/iomodule.cc'
                  ])
        ]
    )

setup(
    name='neurotrees',
    package_dir = {'': 'python'},
    packages = ["neurotrees"],
    version='0.0.1',
    maintainer = "Ivan Raikov",
    maintainer_email = "ivan.g.raikov@gmail.com",
    description = "Neurotrees library",
    py_modules=['neurotrees.range'],
    include_package_data=True,
    install_requires=[
        'click', 'h5py', 'numpy'
    ],
    entry_points='''
        [console_scripts]
        initrange=neurotrees.initrange:cli
        importcoords=neurotrees.importcoords:cli
    ''',
    ext_modules = [
        Extension('neurotrees.io',
                  extra_compile_args = extra_compile_args,
                  extra_link_args = extra_link_args,
                  libraries = libraries,
                  sources = [
                      'hdf5_path_names.cc',
                      'dataset_type.cc',
                      'rank_range.cc',
                      'attrmap.cc',
                      'pack_tree.cc',
                      'read_tree.cc',
                      'scatter_read_tree.cc',
                      'cell_attributes.cc',
                      'dataset_num_elements.cc',
                      'read_population_names.cc',
                      'enum_population_names.cc',
                      'read_population_ranges.cc',
                      'read_tree_index.cc',
                      'read_cell_index.cc',
                      'mpi_bcast_string_vector.cc',
                      'python/neurotrees/iomodule.cc'
                     ])
    ]
)

