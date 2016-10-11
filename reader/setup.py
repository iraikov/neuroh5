#!/usr/bin/env python
import os, sys
from distutils.core import setup, Extension

support_dir = os.path.normpath(
                   os.path.join(
			sys.prefix,
			'share',
			'python%d.%d' % (sys.version_info[0],sys.version_info[1]),
			'CXX') )

if os.name == 'posix':
	CXX_libraries = ['stdc++','m']
else:
	CXX_libraries = []

setup (name = "neurograph_reader",
       version = "0.1",
       maintainer = "Ivan Raikov",
       maintainer_email = "ivan.g.raikov@gmail.com",
       description = "Python interface to Neurographdf5 library",
       url = "http://github.com",
       
       ext_modules = [
         Extension('neurograph_reader',
                   extra_compile_args = ["-std=c++11"],
                   libraries = ['hdf5_mpich', 'mpich'],
                   sources = ['readermodule.cc',
                              'dbs_graph_reader.cc',
                              'population_reader.cc'
                            ])
           ]
           )

