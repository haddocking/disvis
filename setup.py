from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

ext_modules = [Extension("disvis/libdisvis",
        ["src/libdisvis.pyx"])]

package_data = {'disvis': ['data/*.npy', 'IO/*', 'kernels/*']}
scripts = ['scripts/disvis']

setup(name="disvis",
      version='1.0.0',
      description='Quantifying and visualizing the interaction space of distance-constrainted macromolecular complexes',
      author='Gydo C.P. van Zundert',
      author_email='g.c.p.vanzundert@uu.nl',
      packages=['disvis'],
      cmdclass = {'build_ext': build_ext},
      ext_modules = cythonize(ext_modules),
      package_data = package_data,
      scripts=scripts,
      requires=['numpy', 'cython'],
      include_dirs=[numpy.get_include()],
    )
