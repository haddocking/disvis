# Contributing to DisVis

Bug reports, questions and pull requests are welcome; use the issue templates
on GitHub. This file describes how to set up a development environment, check
your changes and get them merged. For using DisVis, see `README.md`; for GPU
acceleration, `INSTALLATION.md`.

## Development setup

DisVis requires Python 3.11 or newer and NumPy 2. It contains two compiled
extensions (Cython and C), so it must be built and installed; it is not a
pure-Python package.

```bash
git clone https://github.com/haddocking/disvis.git
cd disvis
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'  # with ruff and cython-lint; '.[dev,gpu]' for the GPU
```

The package metadata and dependencies are in `pyproject.toml`; `setup.py`
only defines the compiled extensions.

This is an editable install: changes to the Python code in `src/disvis/` take
effect without reinstalling. After changing the compiled extensions
(`libdisvis.pyx` or `_extensions.c`), rebuild them with `pip install -e .`.
The build leaves `build/`, `src/disvis.egg-info/` and the compiled `*.so`
files in `src/disvis/`; git ignores them.

The `gpu` extra installs the GPU requirements; pyvkfft is built from source
and needs a C++ compiler and, on Linux, the OpenCL headers (see
`INSTALLATION.md`).

## Code layout

The package is in `src/disvis/`:

- `main.py`: command-line interface. Parses the arguments, input files and
  restraints, runs the search on the CPU (multiprocessing over subsets of the
  rotations) or GPU, and writes the output files.
- `disvis.py`: the `DisVis` class, the search itself, with a CPU
  implementation (NumPy FFT or pyFFTW) and a GPU implementation (OpenCL).
- `helpers.py`: parsing of restraints, interaction selections and interaction
  files, and other small utilities.
- `rotations.py` and `data/*.npy`: rotation matrices and the precomputed
  rotational sampling sets with their weights.
- `volume.py`: the `Volume` class (3D grid with spacing and origin).
- `pdb.py`, `IO/`: PDB/mmCIF and MRC input and output.
- `kernels.py`, `kernels.cl`: OpenCL kernels for the GPU.
- `gpufft.py`: real 3D FFTs on the GPU with VkFFT, through pyvkfft.
- `libdisvis.pyx` (Cython) and `_extensions.c` (C): the CPU grid operations
  (restraint spaces, violations, interactions, grid rotation).

Outside the package: `tests/` (unit tests), `scripts/` (analysis scripts for
DisVis output) and `test-cases/` (test cases with reference results, see
`test-cases/README.md`).

All grids are indexed as `[z, y, x]`; coordinates are `(x, y, z)`.

## Code style

The code is linted and formatted with [ruff](https://docs.astral.sh/ruff/),
configured in `pyproject.toml`, and the Cython code is linted with
cython-lint:

```bash
ruff check .
ruff format .            # or ruff format --check . to only check
cython-lint --max-line-length 79 src/disvis/libdisvis.pyx
```

Every module, public class and function has a docstring (PEP 257, summary on
the first line). `src/disvis/elements.py` is third-party code (Christoph Gohlke):
it is excluded from formatting and from most lint rules; keep it as it is.

Functions are limited to a McCabe complexity of 10 (ruff rule C901); the few
older functions above it are marked with `# noqa: C901` until they are split
up. Prefer fixing an issue in the code; disable a rule in `pyproject.toml`
only for a deliberate style choice, with a comment saying why.

## Tests

Run the unit tests from the repository root; they test the installed package:

```bash
python -W error -m unittest                                            # all tests
python -W error -m unittest tests/test_disvis.py                       # CPU only
python -W error -m unittest tests/test_kernels.py tests/test_cpu_vs_gpu.py  # GPU
```

The GPU tests run on a GPU if there is one, and otherwise on any other OpenCL
device, such as pocl on the CPU (`pip install 'pyopencl[pocl]'`, Linux only).
They are skipped when the GPU packages or an OpenCL device are missing.

### Reference results

`test-cases/*/example-results/` contains CPU and GPU reference results (see
`test-cases/README.md`). The CPU output files (`accessible_complexes.out`,
`violations.out`, `z-score.out` and `accessible_interaction_space.mrc`) must
stay **byte-identical** to the references. After any change to the search,
run at least the coarse runs of both test cases, which
`tests/check_test_cases.py` runs and compares with the references:

```bash
python tests/check_test_cases.py -p 8      # coarse, on the CPU
python tests/check_test_cases.py --fine    # fine: takes from minutes to hours
python tests/check_test_cases.py --gpu     # compared with the GPU references
```

GPU results agree with the CPU references to about 1e-5 (single precision),
with identical z-scores.

If a change intentionally alters the output, explain why in the pull request,
and update the reference results and `test-cases/README.md`.

## Pull requests

Work on a branch and open a pull request into `main`; fill in the pull-request
template. Before it can be merged, a pull request needs:

- an approving review (and a review from CSBcore for the files listed in
  `.github/CODEOWNERS`),
- all review conversations resolved,
- the GitHub workflow passing (its `ci-passed` check), on a branch that is up
  to date with `main`.

The GitHub workflow (`.github/workflows/ci.yml`) runs on every pull request:
ruff and cython-lint, the unit tests on Linux, macOS and Windows with Python
3.11–3.14, the coarse test cases with `tests/check_test_cases.py`, and the GPU
unit tests on the CPU with pocl. Run the same checks locally before pushing.

If AI tools played a meaningful role, say so in the pull request (the template
has a section for it), and update `aidecl.yaml`, the project's AI declaration,
when a new tool or kind of use is added.

Checklist:

- Tests added for new code.
- Documentation added: docstrings, and `README.md` / `INSTALLATION.md` when
  the usage or installation changes.
- `CHANGELOG.md` updated.
- Reference results unchanged, or updated with an explanation.
- No new dependencies without discussion.
- No new lint errors.
- Prefer small, incremental changes, with passing tests at each step, over
  large rewrites.

## Releases

The version is set in `pyproject.toml` only; `disvis.__version__` reads it
from the installed package metadata. To release, update the version and
`CHANGELOG.md`, merge, then tag `main` as `vX.Y.Z` and create a GitHub release.
Pushing a `v*` tag runs `.github/workflows/docker-publish.yml`, which publishes
the Docker image to ghcr.io. Only CSBcore can create `v*` tags.
