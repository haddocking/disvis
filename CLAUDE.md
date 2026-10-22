# Agent Guidelines

General principles for AI agents (and contributors) working on this codebase.
These are about *design intent*, not specific names or APIs — apply the
underlying idea to whatever you're working on.

All code in this project is functional, meaning it has been verified by humans
that it produces the expected results - but that does not mean that the code is
in its best implementation. If you copy code from somewhere else, try to
improve it, do not assume it is already optimal.

## Overview

DisVis visualises and quantifies the accessible interaction space of
distance-restrained binary biomolecular complexes. It performs a full and
systematic 6D search (three rotational and three translational degrees of
freedom) of a scanning chain (the ligand) around a fixed chain (the receptor),
using FFT-accelerated grid correlations. For each set of restraints it reports
how many complexes are consistent with them, which restraints are likely false
positives (z-scores), and optionally which residues are likely to interact.
It is developed by BonvinLab (Utrecht University).

The search runs on the CPU (in parallel over rotations with `-p`) or on a GPU
through OpenCL (`-g`).

## Setup & Installation

DisVis requires Python 3.11 or newer and NumPy 2. It contains two compiled
extensions (Cython and C), so it must be built and installed; it is not a
pure-Python package.

```bash
# Install (builds the extensions)
pip install .

# With the GPU Python packages (pyopencl, mako)
pip install '.[gpu]'
```

GPU acceleration also needs clFFT and gpyfft, which are not on PyPI and must be
built by hand; see `INSTALLATION.md`. On Linux, `LD_LIBRARY_PATH` must include
the clFFT library directory.

After changing any code, reinstall (`pip install .`) before running it: the
installed package, not the source tree, is what gets tested. Remove the
`build/` and `disvis.egg-info/` directories left by the build.

## Commands

```bash
# Run DisVis
disvis <receptor.pdb> <ligand.pdb> <restraints.dat> [options]

# Example: coarse run of a test case on 8 cores
cd test-cases/PRE5-PUP2-complex
disvis O14250.pdb Q9UT97.pdb restraints.dat -a 9.72 -vs 2 -p 8 -d coarse

# Same on the GPU
disvis O14250.pdb Q9UT97.pdb restraints.dat -a 9.72 -vs 2 -g -d coarse

# Show all options / the version
disvis -h
disvis --version
```

Analysis scripts are in `scripts/` (`analyze.py`, `analyze_interactions.py`,
`plt-average-interactions.py`, `interactions_to_bfactor.py`).

## Architecture

- `disvis/main.py` — command-line interface: parses the arguments, input
  files and restraints, runs the search on the CPU (multiprocessing over
  rotation subsets) or GPU, and writes the output files.
- `disvis/disvis.py` — the `DisVis` class: the search itself, with a CPU
  implementation (NumPy FFT or pyFFTW) and a GPU implementation (OpenCL).
- `disvis/helpers.py` — parsing of restraints and interaction selections,
  and other small utilities.
- `disvis/rotations.py` and `disvis/data/*.npy` — rotation matrices and the
  precomputed rotational sampling sets with their weights.
- `disvis/volume.py` — the `Volume` class (3D grid with spacing and origin).
- `disvis/pdb.py`, `disvis/IO/` — PDB/mmCIF and MRC input and output.
- `disvis/kernels.py`, `disvis/kernels.cl` — OpenCL kernels for the GPU path.
- `disvis/pyclfft.py` — GPU FFTs through gpyfft/clFFT.
- `src/libdisvis.pyx` (Cython) and `src/_extensions.c` (C) — the CPU grid
  operations (restraint spaces, violations, interactions, grid rotation).

All grids are indexed as `[z, y, x]`; coordinates are `(x, y, z)`.

### Output

A run writes into its output directory (`-d`): `accessible_complexes.out`,
`violations.out`, `z-score.out`, `accessible_interaction_space.mrc`,
`disvis.log` and `args.json`, plus the interaction files with `-is` and the
occupancy maps with `-oa`.

## Testing

After making changes, reinstall and run the verification loop:

```bash
# Lint and format (ruff configuration in pyproject.toml)
ruff check .
ruff format --check .          # ruff format . to apply
cython-lint --max-line-length 79 src/libdisvis.pyx

# Unit tests: run from disvis/tests, so that the installed package is used
cd disvis/tests
python -W error -m unittest                         # all tests
python -W error -m unittest test_disvis             # CPU only
python -W error -m unittest test_kernels test_cpu_vs_gpu   # GPU
```

`disvis/elements.py` is third-party code (Christoph Gohlke): it is excluded
from formatting and from most lint rules; keep it as it is. The GPU tests need
the GPU packages and fail without them.

Codacy checks pull requests with ruff, using the configuration in
`pyproject.toml`; any new issue fails the check. Prefer fixing an issue in the
code; disable a rule in `pyproject.toml` only for a deliberate style choice,
with a comment saying why.

### Reference results

`test-cases/*/example-results/` contains CPU and GPU reference results (see
`test-cases/README.md`). The CPU output files must stay **byte-identical** to
the references (`accessible_complexes.out`, `violations.out`, `z-score.out`,
`accessible_interaction_space.mrc`); compare at least the coarse runs of both
test cases after any change to the search. GPU results agree with the CPU
references to about 1e-5 (single precision) with identical z-scores.

Fine runs and benchmarks take from minutes to hours: ask before starting them.
On a cluster, run them through the batch system, not on the login node.

If a change intentionally alters the output, say so explicitly, explain why,
and update the reference results and `test-cases/README.md`.

## Releases

The version is set in `setup.py` only (`disvis.__version__` reads it from the
installed package metadata). Pushing a git tag triggers the Docker publish
workflow (`.github/workflows/docker-publish.yml`): never push tags unless
explicitly asked.

## Pull Request Checklist

- Tests added for new code
- Documentation added (docstrings + `README.md` / `INSTALLATION.md`)
- `CHANGELOG.md` updated
- Reference results unchanged, or updated with an explanation
- No new dependencies without discussion
- Don't introduce new lint errors.
- Prefer small, incremental refactors with passing tests at each step over
  large rewrites.
