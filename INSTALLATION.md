# Installation instructions for GPU-accelerated DisVis

This document shows how to set up GPU-accelerated DisVis using the OpenCL
framework. These instructions were tested on macOS 15 with an Apple M2 Pro GPU
and Python 3.14. Note that the speed-up depends on the GPU: a power-efficient
integrated card may not be much faster than the multi-processor CPU version
(`-p`).

## Requirements

* An OpenCL 1.1+ enabled GPU and its OpenCL driver (included in macOS)
* git
* cmake and a C++ compiler (on macOS: `brew install cmake` and the Xcode
  command line tools)
* DisVis installed in a virtual environment, as described in README.md

All commands below are typed with the virtual environment activated, so that
`$VIRTUAL_ENV` points to it.

## 1. pyopencl

Install the optional GPU requirements of DisVis (pyopencl and Mako) from the
DisVis directory

    pip install '.[gpu]'

and check that your GPU is found

    python -c "import pyopencl as cl; print([d.name for p in cl.get_platforms() for d in p.get_devices()])"

## 2. clFFT

clFFT, the OpenCL FFT library, is not maintained anymore and must be built
from source. Here it is installed into the virtual environment

    git clone https://github.com/clMathLibraries/clFFT.git
    mkdir clFFT/build && cd clFFT/build
    cmake ../src -DCMAKE_BUILD_TYPE=Release -DBUILD_CLIENT=OFF -DBUILD_TEST=OFF \
        -DBUILD_EXAMPLES=OFF -DBUILD_CALLBACK_CLIENT=OFF \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_PREFIX="$VIRTUAL_ENV"
    make -j 8 && make install
    cd ../..

On macOS, give the libraries their full path as install name, so that gpyfft
finds them at run time

    cd "$VIRTUAL_ENV/lib"
    for lib in clFFT StatTimer; do
        install_name_tool -id "$VIRTUAL_ENV/lib/lib$lib.2.dylib" lib$lib.2.*.*.dylib
        codesign -f -s - lib$lib.2.*.*.dylib
    done
    cd -

On Linux, add `$VIRTUAL_ENV/lib` (or `lib64`) to `LD_LIBRARY_PATH` instead.

## 3. gpyfft

gpyfft, the Python bindings to clFFT, is not on PyPI either

    git clone https://github.com/geggo/gpyfft.git
    cd gpyfft

Its `setup.py` contains hard-coded paths to clFFT. In the section for your
operating system (`Darwin` for macOS, `Linux` for Linux), set them to the
virtual environment

    CLFFT_DIR = os.environ['VIRTUAL_ENV']
    CLFFT_LIB_DIRS = [os.path.join(CLFFT_DIR, 'lib')]
    CLFFT_INCL_DIRS = [os.path.join(CLFFT_DIR, 'include'), ]

On Linux, also set `CL_INCL_DIRS` to the directory containing `CL/cl.h`
(e.g. `[]` if the OpenCL headers are installed system-wide). Then type

    pip install .
    cd ..

## 4. Test

From the `disvis/tests` directory, run the GPU tests

    python -m unittest test_kernels test_cpu_vs_gpu

and run DisVis with the `-g` option, e.g. on the PRE5-PUP2 test case

    cd test-cases/PRE5-PUP2-complex
    disvis O14250.pdb Q9UT97.pdb restraints.dat -a 9.72 -vs 2 -g -d gpu-results

The results should agree with the CPU results in `example-results/cpu/coarse`
to within single-precision rounding (relative differences of about 1e-5).
