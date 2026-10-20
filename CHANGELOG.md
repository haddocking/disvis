# Changelog

## 3.0.0

Changes compared with 2.3.0, the last Python 2 release.

### Requirements and installation

- Python 3.11 or newer and NumPy 2; Python 2 is no longer supported.
- pyFFTW is now a requirement, installed automatically; it is about twice as
  fast as the NumPy FFT.
- Cython 3 is needed to build; the generated `src/libdisvis.c` was removed
  and the build requirements are listed in `pyproject.toml`.
- New `gpu` extra (`pip install '.[gpu]'`) for pyopencl; clFFT and gpyfft
  must still be installed by hand (see `INSTALLATION.md`).
- Docker image based on Python 3.12.

### Fixes and changes

- New `--version` option, and `disvis.__version__`.
- `-p` works with the spawn start method (macOS).
- With `-p`, one progress line for all processes, updated in place.
- `disvis.log` is overwritten instead of appended to, and is also written
  with `-g`.
- `args.json` (the command-line options) is written to the output directory.
- Clear error message when `-g` is used without the GPU packages.
- GPU: OpenCL kernels are built once.
- MRC files are closed after reading.
- `scripts/analyze.py` and `scripts/analyze_interactions.py` were rewritten
  for the current code and output format.

### Results

- The CPU output is byte-identical across Python 3.11–3.14, NumPy FFT or
  pyFFTW, and macOS or Linux; the GPU output agrees with it to about 1e-5.

### Documentation and tests

- Updated `README.md` and `INSTALLATION.md` (GPU set-up on macOS and Linux).
- New CPU and GPU reference results for the test cases, and
  `test-cases/README.md` with benchmarks.
- Fixed the outdated GPU unit tests.
- PEP8 formatting and docstrings throughout the code.
- Issue and pull-request templates.
