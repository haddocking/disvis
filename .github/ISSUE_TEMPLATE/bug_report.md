---
name: Bug report
about: A bug is an error, flaw, or unintended behavior in the software that causes it to produce incorrect or unexpected results, or to behave in unintended ways.
title: ''
labels: 'bug'
assignees: ''

---
<!-- Examples of bugs include:

- Crashes or error messages
- Incorrect calculations or output
- Installation problems
- Performance issues
- Security vulnerabilities

If you're requesting new functionality or improvements to existing features, please use the Feature Request template instead. -->

## **Part of DisVis affected**
<!-- Please write down where you observed this bug, e.g. installation, the `disvis` command (CPU, multi-processor `-p`, GPU `-g`), the interaction or occupancy analysis, or one of the scripts in `scripts/` -->

## **Expected behavior**
<!-- Please explain what was the expected behaviour -->

## **Actual behavior**
<!-- Please explain what was the actual behaviour -->

## **Steps to reproduce the behavior**
<!-- Please provide a clear and concise STEP-BY-STEP description of how to reproduce the bug.

Ex:
1. Go to the PRE5-PUP2 test case directory: `cd test-cases/PRE5-PUP2-complex`
2. Run the following command: `disvis O14250.pdb Q9UT97.pdb restraints.dat -a 9.72 -vs 2 -p 4 -d results`
3. Inspect the output and error messages in `results/disvis.log`
 -->

## **What is the severity of this bug?**
<!-- Please provide an indication of the severity of this bug. -->

- [ ] 🔴 High _(code non-functional or wrong results)_
- [ ] 🟠 Medium _(partially affects functionality)_
- [ ] 🟡 Low _(very specific scenario/border case)_

## **Suggestions on how to fix it**
<!-- If you have any suggestions on how to fix the bug, please write them down here -->

## **Version**
<!-- Please add below which version/commit hash of the code you are running, and your operating system and Python version.
To get the version, run `pip show disvis`; to get the commit hash, run the following command in the repository folder: `git rev-parse HEAD` -->

## **Additional context**
<!-- Please add any other context about the bug that might be relevant here.

Ex: Your restraints file, the `args.json` and `disvis.log` files from the output directory, or any other relevant information -->
