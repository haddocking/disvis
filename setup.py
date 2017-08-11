"""
Quantifying and visualizing the interaction space of distance-constrainted
macromolecular complexes.
"""

import os.path
from setuptools import setup
from setuptools.extension import Extension

import numpy

packages = ['disvis', 'disvis.IO']
package_data = {'disvis': [os.path.join('data', '*.npy'), 'kernels.cl']}

ext_modules = [Extension("disvis._extensions",
                         [os.path.join("src", "_extensions.c")],
                         include_dirs=[numpy.get_include()],
                         extra_compile_args=['-ffast-math'],
                         ),
               ]

entry_points = {
    'console_scripts': [
        'disvis3 = disvis.disvis2:main',
    ]
}
requirements = ["numpy", "pyparsing"]

setup(name="disvis",
      version='3.0.0',
      description=__doc__,
      url="https://github.com/haddocking/disvis",
      author='Gydo C.P. van Zundert',
      author_email='gvanzundert51@gmail.com',
      packages=packages,
      ext_modules=ext_modules,
      package_data=package_data,
      entry_points=entry_points,
      install_requires=requirements,
      )
