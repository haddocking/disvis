# DisVis test cases

Two test cases, each with reference results and timings on different hardware.

| Test case                          | Fixed chain  | Scanning chain | Restraints                 |
| ---------------------------------- | ------------ | -------------- | -------------------------- |
| `PRE5-PUP2-complex`                | `O14250.pdb` | `Q9UT97.pdb`   | 7 cross-links              |
| `RNA-polymerase-II/A-E`            | `1wcm_A.pdb` | `1wcm_E.pdb`   | 8 cross-links (from XLdb)  |

## Running the test cases

From the test-case directory, a coarse run (9.72° rotational sampling, 7416
rotations, 2 Å grid)

```bash
disvis O14250.pdb Q9UT97.pdb restraints.dat -a 9.72 -vs 2 -p 8 -d coarse
```

and a fine run (5.27°, 53256 rotations, 1 Å grid)

```bash
disvis O14250.pdb Q9UT97.pdb restraints.dat -a 5.27 -vs 1 -p 8 -d fine
```

Use `-g` instead of `-p 8` to run on the GPU (see `INSTALLATION.md`). For
RNA polymerase II, use `1wcm_A.pdb 1wcm_E.pdb` as input.

## Reference results

`example-results/` contains, for each test case:

* `cpu/coarse` and `cpu/fine`: CPU runs with `-p 8` (macOS, Apple M2 Pro,
  Python 3.14, NumPy 2).
* `gpu/coarse` and `gpu/fine`: GPU runs with `-g` (Rocky Linux 9, NVIDIA RTX
  6000 Ada, Python 3.14), made with the clFFT library. The GPU FFTs now use
  VkFFT (pyvkfft), which gives byte-identical output for all four runs on the
  RTX 6000 Ada, the GTX 1080 Ti and the Apple M2 Pro GPU.

The CPU output files (`accessible_complexes.out`, `violations.out`,
`z-score.out` and `accessible_interaction_space.mrc`) are byte-identical across
Python 3.11–3.14, NumPy FFT or pyFFTW, and macOS (arm64), Linux (x86_64),
Docker and Apptainer. The GPU computes in single precision: the numbers of
accessible complexes agree with the CPU results to about 1e-5 (relative), and
the z-scores and flagged restraints are the same (restraints 1 and 2 for
PRE5-PUP2; 7 and 8 for RNA polymerase II). `disvis.log` records the run
time and the absolute paths of the machine that produced the results.

## Benchmarks

Run times (as reported at the end of `disvis.log`) with DisVis 3.0.0 and
Python 3.14. CPU runs use pyFFTW unless marked NumPy FFT; GPU runs use VkFFT
(pyvkfft 2025.1.1), with the earlier clFFT times in brackets. Each run was
timed once.

| Hardware                                   | Mode              | PRE5-PUP2 coarse | RNA-pol II coarse | PRE5-PUP2 fine | RNA-pol II fine |
| ------------------------------------------ | ----------------- | ---------------: | ----------------: | -------------: | --------------: |
| Apple M2 Pro (8+4 cores)                   | 1 core, NumPy FFT | 1m 2s            | 2m 56s            | –              | –               |
| Apple M2 Pro                               | 1 core            | 30s              | 1m 31s            | –              | –               |
| Apple M2 Pro                               | `-p 8`, NumPy FFT | 14s              | 32s               | 11m 25s        | 47m 28s         |
| Apple M2 Pro                               | `-p 8`            | 8s               | 26s               | 9m 22s         | 40m 49s         |
| AMD EPYC 7451 (2 × 24 cores, full node)    | `-p 8`            | 14s              | 59s               | 25m 10s        | 82m 1s          |
| AMD EPYC 7451                              | `-p 16`           | 8s               | 51s               | 22m 21s        | 42m 51s         |
| AMD EPYC 7451                              | `-p 32`           | 6s               | 1m 12s            | 10m 45s        | 24m 56s         |
| Apple M2 Pro integrated GPU                | `-g`              | 18s (18s)        | 24s (27s)         | 5m 16s (5m 14s) | 17m 53s (13m 59s) |
| NVIDIA GeForce GTX 1080 Ti                 | `-g`              | 11s (13s)        | 14s (24s)         | 2m 37s (5m 59s) | 6m 39s (22m 46s)  |
| NVIDIA RTX 6000 Ada                        | `-g`              | 5s (4s)          | 5s (5s)           | 44s (56s)       | 1m 49s (2m 31s)   |
| 2015 results (Python 2, unknown hardware)  | 16 CPUs           | –                | –                 | 27m 5s         | 73m 48s         |

Notes:

* pyFFTW, installed automatically with DisVis, is about 2× faster than the
  NumPy FFT on a single core, and 1.2–1.75× faster with `-p 8`.
* The GPU gains most on the fine runs: on the short coarse runs, fixed costs
  (reading the input, building the OpenCL kernels, writing the results)
  dominate. The RTX 6000 Ada is about 14× faster than 32 EPYC cores and 13–22×
  faster than 8 M2 Pro cores on the fine runs.
* VkFFT is 2.3–3.4× faster than clFFT on the GTX 1080 Ti fine runs and
  1.3–1.4× faster on the RTX 6000 Ada. On the M2 Pro it is as fast for
  PRE5-PUP2 and slower for RNA-pol II fine (17m 53s against 13m 59s).
* Scaling with `-p` is uneven, and short runs gain little from more
  processes. The EPYC 7451 cores are about 2.7× slower than the M2 Pro
  performance cores.
* During the M2 Pro `-p 8` RNA-pol II fine run with pyFFTW, other builds also
  used the CPU, so this time may be slightly too long.
* The 2015 results predate the 2017 fix of the ligand van der Waals radii.
