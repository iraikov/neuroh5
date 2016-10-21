#!/usr/bin/env python
import os, sys
from distutils.core import setup, Extension

setup(
    name='neurograph',
    version='0.0.1',
    maintainer = "Ivan Raikov",
    maintainer_email = "ivan.g.raikov@gmail.com",
    description = "Neurograph library",
    url = "http://github.com/gheber/neurographdf5",
    py_modules=['neurograph'],
    include_package_data=True,
    install_requires=[
        'click', 'h5py', 
    ],
    entry_points='''
        [console_scripts]
        neurograph=neurograph:cli
        importpairs=importpairs:cli
    ''',
    ext_modules = [
        Extension('neurograph_reader',
                    extra_compile_args = ["-std=c++11"],
                    libraries = ['hdf5_mpich', 'mpich'],
                    sources = ['reader/readermodule.cc',
                               'reader/dbs_graph_reader.cc',
                               'reader/population_reader.cc'
                               ])
        ]
    )

