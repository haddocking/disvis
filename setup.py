#! env/bin/python
import os.path
from setuptools import setup
from setuptools.extension import Extension

import numpy
try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    CYTHON = True
except ImportError:
    CYTHON = False

description = ('Quantifying and visualizing the interaction space '
               'of distance-constrainted macromolecular complexes')

packages = ['disvis', 'disvis.IO']
package_data = {'disvis': [os.path.join('data', '*.npy'), 'kernels.cl']}

ext = '.pyx' if CYTHON else '.c'
ext_modules = [Extension("disvis.libdisvis",
                  [os.path.join("src", "libdisvis" + ext)],
                  include_dirs=[numpy.get_include()],
                  extra_compile_args=['-ffast-math']),
               Extension("disvis._extensions",
                  [os.path.join("src", "_extensions.c")],
                  include_dirs=[numpy.get_include()],
                  extra_compile_args=['-ffast-math'],
                  ),
              ]

cmdclass = {}
if CYTHON:
    ext_modules = cythonize(ext_modules)
    cmdclass['build_ext'] = build_ext

scripts = [os.path.join('scripts', 'disvis')]
entry_points = {
        'console_scripts': [
            'disvis3 = disvis.disvis2:main',
            ]
        }
requirements = ["numpy", "pyparsing"]

setup(name="disvis",
      version='3.0.0',
      description=description,
      url="https://github.com/haddocking/disvis",
      author='Gydo C.P. van Zundert',
      author_email='gvanzundert51@gmail.com',
      packages=packages,
      cmdclass = cmdclass,
      ext_modules=ext_modules,
      package_data = package_data,
      scripts=scripts,
      entry_points=entry_points,
      install_requires=requirements,
     )
