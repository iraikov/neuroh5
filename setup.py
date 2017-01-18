#!/usr/bin/env python
import os, sys
from distutils.core import setup, Extension

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
    ext_package = 'neurograph',
    ext_modules = [
        Extension('io',
                  extra_compile_args = ["-std=c++11",
                                        "-U NDEBUG",
                                        "-I/usr/include/hdf5/mpich",
                                        "-I/usr/include/mpich", "-Iinclude",
                                        "-Iinclude/graph", "-Iinclude/io", "-Iinclude/model",
                                        "-Iinclude/io/hdf5",
                                        "-g"],
                  libraries = ['hdf5_mpich', 'mpich'],
                  sources = [
                      'src/io/hdf5/read_dbs_projection.cc',
                      'src/io/hdf5/ngh5.io.hdf5.cc',
                      'src/io/hdf5/projection_names.cc',
                      'src/io/hdf5/dataset_num_elements.cc',
                      'src/io/hdf5/hdf5_path_names.cc',
                      'src/io/hdf5/population_reader.cc',
                      'src/io/hdf5/read_link_names.cc',
                      'src/io/ngh5.io.cc',
                      'src/io/read_population.cc',
                      'src/graph/read_graph.cc',
                      'src/graph/scatter_graph.cc',
                      'src/graph/validate_edge_list.cc',
                      'src/python/iomodule.cc'
                  ])
        ]
    )

