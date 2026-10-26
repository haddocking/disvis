# Installation instructions for GPU-accelerated DisVis

This document shows how to set up GPU-accelerated DisVis using the OpenCL
framework, with [pyopencl](https://github.com/inducer/pyopencl) and the VkFFT
library through [pyvkfft](https://github.com/vincefn/pyvkfft) for the Fourier
transforms. These instructions were tested with Python 3.14 on macOS 15 (Apple
M2 Pro GPU) and on Rocky Linux 9 (NVIDIA GeForce GTX 1080 Ti and RTX 6000 Ada
GPUs, NVIDIA driver 555). Note that the speed-up depends on the GPU: a
power-efficient integrated card may not be much faster than the multi-processor
CPU version (`-p`).

## Requirements

* An OpenCL 1.2+ enabled GPU and its OpenCL driver (included in macOS; on Linux
  with NVIDIA GPUs, part of the NVIDIA driver).
* A C++ compiler (on macOS, the Xcode command line tools), as pyvkfft is built
  from source.
* On Linux, the OpenCL headers (`CL/cl.h`) and loader (`libOpenCL.so`), e.g.
  the `opencl-headers` and `ocl-icd-opencl-dev` (Debian/Ubuntu) or
  `opencl-headers` and `ocl-icd-devel` (Rocky/RHEL) packages.
* DisVis installed in a virtual environment, as described in README.md.

On a cluster, the OpenCL headers and libraries may only be installed on the
GPU nodes: install the GPU requirements there.

## Installation

From the DisVis directory, with the virtual environment activated, install the
optional GPU requirements of DisVis (the `gpu` extra in `pyproject.toml`)

```bash
pip install '.[gpu]'
```

and check that your GPU is found

```bash
python -c "import pyopencl as cl; print([d.name for p in cl.get_platforms() for d in p.get_devices()])"
```

## Test

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
