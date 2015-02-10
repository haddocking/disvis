#! env/bin/python
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy
import os.path

packages = ['disvis', 'disvis.IO']

ext_modules = [Extension("disvis.libdisvis",
                  [os.path.join("src", "libdisvis.pyx")],
                  include_dirs = [numpy.get_include()],
              )]

package_data = {'disvis': [os.path.join('data', '*.npy'), 
                           os.path.join('IO', '*.py'),
                           os.path.join('kernels', '*.cl')]}

scripts = [os.path.join('scripts', 'disvis')]

setup(name="disvis",
      version='1.0.0',
      description='Quantifying and visualizing the interaction space of distance-constrainted macromolecular complexes',
      author='Gydo C.P. van Zundert',
      author_email='g.c.p.vanzundert@uu.nl',
      packages=packages,
      cmdclass = {'build_ext': build_ext},
      ext_modules = cythonize(ext_modules),
      package_data = package_data,
      scripts=scripts,
      requires=['numpy', 'cython'],
      include_dirs=[numpy.get_include()],
     )
