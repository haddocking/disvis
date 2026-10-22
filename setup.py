"""Build and install DisVis."""

import os.path
from setuptools import setup
from setuptools.extension import Extension

import numpy
from Cython.Build import cythonize

description = (
    'Quantifying and visualizing the interaction space '
    'of distance-constrained macromolecular complexes'
)

packages = ['disvis', 'disvis.IO']
package_data = {'disvis': [os.path.join('data', '*.npy'), 'kernels.cl']}

ext_modules = [
    Extension(
        "disvis.libdisvis",
        [os.path.join("src", "libdisvis.pyx")],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-ffast-math'],
    ),
    Extension(
        "disvis._extensions",
        [os.path.join("src", "_extensions.c")],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-ffast-math'],
    ),
]

# Generate the C sources in the build directory, out of the source tree.
ext_modules = cythonize(
    ext_modules, build_dir='build', compiler_directives={'language_level': 3}
)

# pyFFTW roughly halves the run time compared with the NumPy FFT.
requirements = ["numpy", "pyfftw"]
# GPU acceleration also needs clFFT and gpyfft, which are not on PyPI; see
# INSTALLATION.md.
extras = {'gpu': ['pyopencl', 'mako']}

setup(
    name="disvis",
    version='3.0.0',
    description=description,
    url="https://github.com/haddocking/disvis",
    author='Gydo C.P. van Zundert',
    author_email='gvanzundert51@gmail.com',
    packages=packages,
    ext_modules=ext_modules,
    package_data=package_data,
    install_requires=requirements,
    extras_require=extras,
    python_requires='>=3.11',
    entry_points={
        'console_scripts': [
            'disvis = disvis.main:main',
        ]
    },
)
