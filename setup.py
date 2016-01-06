#! env/bin/python
from os.path import join
from distutils.core import setup
from distutils.extension import Extension
try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    CYTHON = True
except ImportError:
    CYTHON = False

# check numpy version
import numpy
np_major, np_minor = [int(x) for x in numpy.version.short_version.split('.')[:2]]
if np_major < 1 or (np_major == 1 and np_minor < 8):
    raise ImportError('DisVis requires NumPy version 1.8 or ' \
        'higher. You have version {:s}'.format(numpy.version.short_version))

def main():
    packages = ['disvis', 'disvis.IO']
    requirements = ['numpy']

    ext = '.pyx' if CYTHON else '.c'
    ext_modules = [Extension("disvis.libdisvis",
                      [join("src", "libdisvis" + ext)],
                      include_dirs = [numpy.get_include()],
                  )]

    cmdclass = {}
    if CYTHON:
        ext_modules = cythonize(ext_modules)
        cmdclass = {'build_ext' : build_ext}

    package_data = {'disvis': [join('data', '*.npy'), 
                               join('kernels', '*.cl')]}

    scripts = [join('scripts', 'disvis')]

    setup(name="disvis",
          version='1.0.1',
          description='Quantifying and visualizing the interaction space of distance-constrainted macromolecular complexes',
          author='Gydo C.P. van Zundert',
          author_email='g.c.p.vanzundert@uu.nl',
          packages=packages,
          cmdclass=cmdclass,
          ext_modules=ext_modules,
          package_data=package_data,
          scripts=scripts,
          requires=requirements,
          include_dirs=[numpy.get_include()],
         )

if __name__ == '__main__':
    main()
