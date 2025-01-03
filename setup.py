#!/usr/bin/env python
import os, sys, math, subprocess, platform, sysconfig
import multiprocessing
from setuptools import setup
from setuptools.extension import Extension
from distutils.command import build_ext

cmake_cmd_args = []
for f in sys.argv:
    if f.startswith("-D"):
        cmake_cmd_args.append(f)


def num_available_cpu_cores(ram_per_build_process_in_gb=1):
    try:
        mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')  
        mem_gib = mem_bytes/(1024.**3)
        num_cores = multiprocessing.cpu_count() 
        # make sure we have enough ram for each build process.
        mem_cores = int(math.floor(mem_gib/float(ram_per_build_process_in_gb)+0.5));
        # We are limited either by RAM or CPU cores.  So pick the limiting amount
        # and return that.
        return max(min(num_cores, mem_cores), 1)
    except ValueError:
        return 2 # just assume 2 if we can't get the os to tell us the right answer.

    
class CMakeExtension(Extension):
    def __init__(self, name, target="all", cmake_lists_dir=".", **kwa):
        Extension.__init__(self, name, sources=[], **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)
        self.target = target


class cmake_build_ext(build_ext.build_ext):
    def build_extensions(self):
        # Ensure that CMake is present and working
        cmake_path = "cmake"
        if os.environ.get("CMAKE_PATH", False):
            cmake_path = os.environ.get("CMAKE_PATH", False)
            
        try:
            out = subprocess.check_output([cmake_path, "--version"])
        except OSError:
            raise RuntimeError("Cannot find CMake executable")

        options = {"debug_build": False, "HDF5_ROOT": False, "JEMALLOC_ROOT": False}

        if os.environ.get("NEUROH5_DEBUG", False):
            options["debug_build"] = True
        if os.environ.get("HDF5_ROOT", False):
            options["HDF5_ROOT"] = os.environ.get("HDF5_ROOT")
            options["HDF5_DIR"] = os.environ.get("HDF5_ROOT")
        if os.environ.get("JEMALLOC_ROOT", False):
            options["JEMALLOC_ROOT"] = os.environ.get("JEMALLOC_ROOT")

        for ext in self.extensions:

            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            cfg = "Debug" if options.get("debug_build", False) else "RelWithDebInfo"

            cmake_args = [
                "-DCMAKE_BUILD_TYPE=%s" % cfg,
                # Ask CMake to place the resulting library in the directory
                # containing the extension
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir),
                # Other intermediate static libraries are placed in a
                # temporary build directory instead
                "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{}={}".format(
                    cfg.upper(), self.build_temp
                ),
                # Hint CMake to use the same Python executable that
                # is launching the build, prevents possible mismatching if
                # multiple versions of Python are installed
                "-DPYTHON_EXECUTABLE={}".format(sys.executable),
                # Add other project-specific CMake arguments if needed
                # ...
            ]

            # We can handle some platform-specific settings at our discretion
            if platform.system() == "Windows":
                plat = "x64" if platform.architecture()[0] == "64bit" else "Win32"
                cmake_args += [
                    # These options are likely to be needed under Windows
                    "-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE",
                    "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}".format(
                        cfg.upper(), extdir
                    ),
                ]
                # Assuming that Visual Studio and MinGW are supported compilers
                if self.compiler.compiler_type == "msvc":
                    cmake_args += [
                        "-DCMAKE_GENERATOR_PLATFORM=%s" % plat,
                    ]
                else:
                    cmake_args += [
                        "-G",
                        "MinGW Makefiles",
                    ]

            python_config_vars = sysconfig.get_config_vars()
            mdt = os.getenv("MACOSX_DEPLOYMENT_TARGET")
            if mdt is None:
                mdt = python_config_vars.get("MACOSX_DEPLOYMENT_TARGET", None)
            if mdt:
                cmake_args.append("-DCMAKE_OSX_DEPLOYMENT_TARGET={}".format(mdt))

            if options.get("HDF5_ROOT", False):
                cmake_args += [
                    "-DHDF5_ROOT=%s" % options.get("HDF5_ROOT"),
                ]

            if options.get("JEMALLOC_ROOT", False):
                cmake_args += [
                    "-DJEMALLOC_ROOT_DIR=%s" % options.get("JEMALLOC_ROOT"),
                ]

            cmake_args += cmake_cmd_args

            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)

            # Config
            subprocess.check_call(
                ["cmake", ext.cmake_lists_dir] + cmake_args, cwd=self.build_temp
            )

            # Build
            cmake_build_args = ["--build", ".", "--config", cfg, "--target", ext.target]
            cmake_build_args += ["--", f"-j{num_available_cpu_cores()}"]

            subprocess.check_call(
                ["cmake"] + cmake_build_args,
                cwd=self.build_temp,
            )

with open("README.md", 'r') as f:
    long_description = f.read()            

setup(
    name="NeuroH5",
    package_dir={"": "python"},
    packages=["neuroh5"],
    version="0.1.12",
    maintainer="Ivan Raikov",
    maintainer_email="ivan.g.raikov@gmail.com",
    description="A parallel HDF5-based library for storage and processing of large-scale graphs and neural cell model attributes.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="http://github.com/iraikov/neuroh5",
    include_package_data=True,
#    entry_points={
#        "console_scripts": [
#            'initrange=neuroh5.initrange:cli',
#            'initprj=neuroh5.initprj:cli',
#            'importdbs=neuroh5.importdbs:cli',
#            'importcoords=neuroh5.importcoords:cli',
#        ]
#    },
    cmdclass={"build_ext": cmake_build_ext},
    ext_modules=[CMakeExtension("neuroh5.io", target="python_neuroh5_io")],
)
