#!/usr/bin/env python
import os, sys
import mpi4py
import numpy as np
from distutils.core import setup, Extension


if sys.platform == 'darwin':
    HDF5_INCDIR = os.environ.get("HDF5_INCDIR", "/usr/local/hdf5/include")
    HDF5_LIBDIR = os.environ.get("HDF5_LIBDIR", "/usr/local/hdf5/lib")
    HDF5_LIB    = os.environ.get("HDF5_LIB", "hdf5")
    MPI_LIBDIR  = os.environ.get("MPI_LIBDIR", "/usr/local/Cellar/mpich/3.2_2/lib")
    MPI_INCDIR  = os.environ.get("MPI_INCDIR", "/usr/local/Cellar/mpich/3.2_2/include")
    MPI_LIB     = os.environ.get("MPI_LIB", "mpich")
    NUMPY_INCDIR= np.get_include()
    MPI4PY_INCDIR=mpi4py.get_include()
    extra_compile_args = ["-std=c++11",
                          "-stdlib=libc++",	
                          "-mmacosx-version-min=10.9",
                          "-UNDEBUG",
                          "-I"+HDF5_INCDIR,
                          "-I"+MPI_INCDIR,
                          "-I"+NUMPY_INCDIR,
                          "-I"+MPI4PY_INCDIR,
                          "-Iinclude", "-Iinclude/cell", "-Iinclude/graph", "-Iinclude/ngraph",
                          "-Iinclude/data", "-Iinclude/mpi", "-Iinclude/hdf5",
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
    MPI4PY_INCDIR=os.environ.get("MPI4PY_INCDIR", mpi4py.get_include())
    extra_compile_args = ["-std=c++11",
                        "-UNDEBUG",
                        "-I"+HDF5_INCDIR,
                        "-I"+NUMPY_INCDIR,
                        "-I"+MPI4PY_INCDIR,
                        "-Iinclude", "-Iinclude/cell", "-Iinclude/graph", "-Iinclude/ngraph",
                        "-Iinclude/data", "-Iinclude/mpi", "-Iinclude/hdf5",
                          "-g", "-O0"]
    extra_link_args = ["-L"+HDF5_LIBDIR]
    if MPI_LIBDIR != "":
        extra_link_args = extra_link_args + ["-L"+MPI_LIBDIR]
    if MPI_INCDIR != "":
        extra_compile_args = extra_compile_args + ["-I"+MPI_INCDIR]
    if MPI_LIB != "":
        libraries = [HDF5_LIB, MPI_LIB]
    else:
        libraries = [HDF5_LIB]

setup(
    name='NeuroH5',
    package_dir = {'': 'python'},
    packages = ["neuroh5"],
    version='0.0.2',
    maintainer = "Ivan Raikov",
    maintainer_email = "ivan.g.raikov@gmail.com",
    description = "NeuroH5 library",
    url = "http://github.com/gheber/neurographdf5",
    include_package_data=True,
    install_requires=[
        'click', 'h5py', 'numpy', 'mpi4py'
    ],
    entry_points='''
        [console_scripts]
        initrange=neuroh5.initrange:cli
        initprj=neuroh5.initprj:cli
        importdbs=neuroh5.importdbs:cli
        importcoords=neuroh5.importcoords:cli
    ''',
    ext_modules = [
        Extension('neuroh5.io',
                  extra_compile_args = extra_compile_args,
                  extra_link_args = extra_link_args,
                  libraries = libraries,
                  define_macros = [('NPY_NO_DEPRECATED_API','NPY_1_7_API_VERSION')],
                  sources = [
                      'python/neuroh5/iomodule.cc',
                      'src/data/append_rank_tree_map.cc',
                      'src/data/attr_map.cc',
                      'src/data/attr_val.cc',
                      'src/data/serialize_edge.cc',
                      'src/data/serialize_tree.cc',
                      'src/data/serialize_cell_attributes.cc',
                      'src/data/append_rank_edge_map.cc',
                      'src/data/append_edge_map.cc',
                      'src/data/append_edge_map_selection.cc',
                      'src/data/append_rank_attr_map.cc',
                      'src/data/tokenize.cc',
                      'src/data/range_sample.cc',
                      'src/cell/read_tree.cc',
                      'src/cell/cell_populations.cc',
                      'src/cell/validate_tree.cc',
                      'src/cell/contract_tree.cc',
                      'src/cell/scatter_read_tree.cc',
                      'src/cell/cell_index.cc',
                      'src/cell/cell_attributes.cc',
                      'src/cell/append_tree.cc',
                      'src/graph/scatter_read_graph.cc',
                      'src/graph/edge_attributes.cc',
                      'src/graph/read_graph.cc',
                      'src/graph/read_graph_selection.cc',
                      'src/graph/write_projection.cc',
                      'src/graph/read_projection.cc',
                      'src/graph/read_projection_selection.cc',
                      'src/graph/scatter_read_projection.cc',
                      'src/graph/projection_names.cc',
                      'src/graph/merge_edge_map.cc',
                      'src/graph/validate_edge_list.cc',
                      'src/graph/validate_selection_edge_list.cc',
                      'src/graph/balance_graph_indegree.cc',
                      'src/graph/node_attributes.cc',
                      'src/graph/compute_vertex_metrics.cc',
                      'src/graph/vertex_degree.cc',
                      'src/graph/bcast_graph.cc',
                      'src/graph/write_graph.cc',
                      'src/graph/append_graph.cc',
                      'src/graph/append_projection.cc',
                      'src/mpi/rank_range.cc',
                      'src/mpi/mpe_seq.cc',
                      'src/hdf5/num_projection_blocks.cc',
                      'src/hdf5/dataset_num_elements.cc',
                      'src/hdf5/read_cell_index_ptr.cc',
                      'src/hdf5/read_link_names.cc',
                      'src/hdf5/read_projection_datasets.cc',
                      'src/hdf5/read_projection_dataset_selection.cc',
                      'src/hdf5/attr_kind_datatype.cc',
                      'src/hdf5/exists_dataset.cc',
                      'src/hdf5/exists_group.cc',
                      'src/hdf5/exists_h5types.cc',
                      'src/hdf5/copy_h5types.cc',
                      'src/hdf5/create_file_toplevel.cc',
                      'src/hdf5/path_names.cc',
                      'src/hdf5/group_contents.cc',
                      'src/hdf5/dataset_type.cc',
                      'src/hdf5/create_group.cc',
                      'src/hdf5/file_access.cc'
                  ])
        ]
    )



