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
    py_modules=['importdbs','importpairs'],
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
        Extension('reader',
                  extra_compile_args = ["-std=c++11", "-Ireader/include"],
                  libraries = ['hdf5_mpich', 'mpich'],
                  sources = [
                      'reader/src/graph_reader.cc',
                      'reader/src/dbs_edge_reader.cc',
                      'reader/src/population_reader.cc',
                      'reader/src/attributes.cc',
                      'reader/src/readermodule.cc',
                  ])
        ]
    )

