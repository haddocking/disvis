"""Build the compiled extensions of DisVis.

The package metadata is in pyproject.toml.
"""

import os.path

import numpy
from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

ext_modules = [
    Extension(
        "disvis.libdisvis",
        [os.path.join("src", "disvis", "libdisvis.pyx")],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-ffast-math'],
    ),
    Extension(
        "disvis._extensions",
        [os.path.join("src", "disvis", "_extensions.c")],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-ffast-math'],
    ),
]

# Generate the C sources in the build directory, out of the source tree.
ext_modules = cythonize(
    ext_modules, build_dir='build', compiler_directives={'language_level': 3}
)

setup(ext_modules=ext_modules)
