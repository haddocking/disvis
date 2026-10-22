# Installation instructions for GPU-accelerated DisVis

This document shows how to set up GPU-accelerated DisVis using the OpenCL
framework. These instructions were tested with Python 3.14 on macOS 15 (Apple
M2 Pro GPU) and on Rocky Linux 9 (NVIDIA GeForce GTX 1080 Ti and RTX 6000 Ada
GPUs, NVIDIA driver 555). Note that the speed-up depends on the GPU: a power-efficient
integrated card may not be much faster than the multi-processor CPU version
(`-p`).

## Requirements

* An OpenCL 1.1+ enabled GPU and its OpenCL driver (included in macOS; on
  Linux with NVIDIA GPUs, part of the NVIDIA driver), and the OpenCL headers
  (`CL/cl.h`) and loader (`libOpenCL.so`) on Linux
* git
* cmake and a C++ compiler (on macOS: `brew install cmake` and the Xcode
  command line tools; cmake can also be installed with `pip install cmake`)
* DisVis installed in a virtual environment, as described in README.md

On a cluster, the OpenCL headers and libraries may only be installed on the
GPU nodes: build clFFT and gpyfft there.

All commands below are typed with the virtual environment activated, so that
`$VIRTUAL_ENV` points to it.

## 1. pyopencl

Install the optional GPU requirements of DisVis (pyopencl and Mako) from the
DisVis directory

```bash
pip install '.[gpu]'
```

and check that your GPU is found

```bash
python -c "import pyopencl as cl; print([d.name for p in cl.get_platforms() for d in p.get_devices()])"
```

## 2. clFFT

clFFT, the OpenCL FFT library, is not maintained anymore and must be built
from source. Here it is installed into the virtual environment

```bash
git clone https://github.com/clMathLibraries/clFFT.git
mkdir clFFT/build && cd clFFT/build
cmake ../src -DCMAKE_BUILD_TYPE=Release -DBUILD_CLIENT=OFF -DBUILD_TEST=OFF \
    -DBUILD_EXAMPLES=OFF -DBUILD_CALLBACK_CLIENT=OFF \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_PREFIX="$VIRTUAL_ENV"
make -j 8 && make install
cd ../..
```

On macOS, give the libraries their full path as install name, so that gpyfft
finds them at run time

```bash
cd "$VIRTUAL_ENV/lib"
for lib in clFFT StatTimer; do
    install_name_tool -id "$VIRTUAL_ENV/lib/lib$lib.2.dylib" lib$lib.2.*.*.dylib
    codesign -f -s - lib$lib.2.*.*.dylib
done
cd -
```

On Linux, add the library directory to `LD_LIBRARY_PATH` instead, also
whenever you run DisVis with `-g`, e.g. in your job script

```bash
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH"
```

(use `lib64` if clFFT was installed there).

## 3. gpyfft

gpyfft, the Python bindings to clFFT, is not on PyPI either

```bash
git clone https://github.com/geggo/gpyfft.git
cd gpyfft
```

Its `setup.py` contains hard-coded paths to clFFT. In the section for your
operating system (`Darwin` for macOS, `Linux` for Linux), set them to the
virtual environment, using its full path (the output of `echo $VIRTUAL_ENV`)

```python
CLFFT_DIR = r'/full/path/to/your/venv'
CLFFT_LIB_DIRS = [os.path.join(CLFFT_DIR, 'lib')]
CLFFT_INCL_DIRS = [os.path.join(CLFFT_DIR, 'include'), ]
```

Do not use `os.environ['VIRTUAL_ENV']` here: some installers, such as *uv*,
build packages in a temporary environment and change `VIRTUAL_ENV`. On some
Linux systems clFFT is installed in `lib64` instead of `lib`.

On Linux, also set `CL_INCL_DIRS` to the directory containing `CL/cl.h`
(e.g. `[]` if the OpenCL headers are installed system-wide). Then type

```bash
pip install .
cd ..
```

## 4. Test

From the DisVis directory, run the GPU tests

```bash
python -m unittest tests/test_kernels.py tests/test_cpu_vs_gpu.py
```

and run DisVis with the `-g` option, e.g. on the PRE5-PUP2 test case

```bash
cd test-cases/PRE5-PUP2-complex
disvis O14250.pdb Q9UT97.pdb restraints.dat -a 9.72 -vs 2 -g -d gpu-results
```

The results should agree with the GPU results in `example-results/gpu/coarse`
and the CPU results in `example-results/cpu/coarse` to within single-precision
rounding (relative differences of about 1e-5). Timings on several GPUs are
listed in `test-cases/README.md`.
