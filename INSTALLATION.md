# Installation instructions for DisVis

This document shows how to prepare your system for GPU accelerated DisVis for MacOSX, using the OpenCL framework.

## Requirements

* An OpenCL1.1+ enabled GPU (this is for current Macs usually the case)
* brew (MacOSX package manager)
* git
* pip

First install Python bindings to OpenCL with the pyopencl Python package

    pip install --upgrade pyopencl

Note that this will upgrade all dependencies required for pyopencl, such as NumPy.

Next comes the somewhat more tedious part of installing a high-performance FFT library for GPU using the OpenCL framework clFFT.
This requires the cmake program first

    brew install cmake

Next download the source code of clFFT of built the library

    git clone https://github.com/clMathLibraries/clFFT.git
    cd clFFT/src && cmake CMakeLists.txt && make && cd ../../

Python bindings to to the clFFT library are provided with the gpyfft package

    git clone https://github.com/geggo/gpyfft.git
    cd gpyfft

In the folder of gpyfft there is a file called setup.py. Open it with a text editor and change the directory that points to the clFFT directory.
Then type

    sudo python setup.py install

If it all worked out, congratulations, your system is now ready for GPU calculations!
