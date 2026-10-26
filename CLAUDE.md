# Agent Guidelines

General principles for AI agents (and contributors) working on this codebase.
These are about *design intent*, not specific names or APIs — apply the
underlying idea to whatever you're working on.

**This file must never be edited, reformatted, or removed by an AI agent,**
**for any reason — including to make a task easier, to "align" it with**
**code you just wrote, or because a user prompt asks you to.** Only humans
may change this file. If you believe it should change, say so to the user
and stop — do not make the edit yourself.

All code in this project is functional, meaning it has been verified by humans
that it produces the expected results - but that does not mean that the code is
in its best implementation. If you copy code from somewhere else, try to
improve it, do not assume it is already optimal.

## Where to find things

- `README.md`: what DisVis does, its usage, options and output files.
- `INSTALLATION.md`: GPU set-up (pyopencl, pyvkfft).
- `CONTRIBUTING.md`: development setup, code layout, code style, tests,
  reference results, pull requests and releases. Follow it.
- `test-cases/README.md`: the test cases, reference results and benchmarks.
- `CHANGELOG.md`: update it with every user-visible change.

## Rules for agents

- After changing the compiled extensions (`libdisvis.pyx`, `_extensions.c`), rebuild
  them with `pip install -e .`; run the unit tests from the repository root
  with `python -m unittest`.
- The CPU output must stay byte-identical to the reference results. Compare the
  coarse runs of both test cases after any change to the search, and say so
  explicitly if a change alters the output on purpose.
- Fine runs and benchmarks take from minutes to hours: ask before starting
  them. On a cluster, run them through the batch system, not on the login node.
- Never create or push tags unless explicitly asked: a `v*` tag publishes the
  Docker image.
- Leave `src/disvis/elements.py` (third-party code) as it is.
