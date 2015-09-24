# DisVis


## About DisVis

**DisVis** is a Python package and command line tool to visualize and quantify
the accessible interaction space of distance restrained binary biomolecular
complexes.  It performs a full and systematic 6 dimensional search of the three
translational and rotational degrees of freedom to determine the number of
complexes consistent with the restraints. In addition, it outputs the
percentage of restraints being violated and a density that represents the
center-of-mass position of the scanning chain corresponding to the highest
number of consistent restraints at every position in space.


## Requirements

* Python2.7
* NumPy1.8+
* C-compiler (e.g. GCC)

Optional for faster CPU version

* FFTW3
* pyFFTW

For GPU hardware acceleration the following packages are also required

* OpenCL1.1+
* [pyopencl](https://github.com/pyopencl/pyopencl)
* [clFFT](https://github.com/clMathLibraries/clFFT)
* [gpyfft](https://github.com/geggo/gpyfft)

Recommended for installation

* [git](https://git-scm.com/download)
* [pip](https://pip.pypa.io/en/latest/installing.html)
* Cython


## Installation

If the requirements are met, **DisVis** can be installed by opening a terminal
and typing

    git clone https://github.com/haddocking/disvis.git
    cd disvis
    python setup.py install

The last command might require administrator priviliges for system wide
installation. After installation, the command line tool *disvis* should be at
your disposal.

If you are starting from a clean system, then read the installation
instructions below to prepare your particular operating system. The
INSTALLATION.md file contains instructions for accessing GPU acceleration on
MacOSX.


### Linux

If Python2.7 is present on your system, use the following command on Ubuntu &
Debian systems

    sudo apt-get install git python-dev python-numpy

and for Fedora systems

    sudo yum install git python-devel numpy 

These commands will install *git*, the Python header files and NumPy. Wait
untill the installation is finished. Your system is now ready. Follow the
general instructions above to install **DisVis**.


### MacOSX

First install [*git*](https://git-scm.com/download) for MacOSX (this might
already be present on your system) by either installing it from the website or
using *brew*

    brew install git

Next, install NumPy by opening a terminal and type

    sudo easy_install numpy

Wait till the compilation and installation is finished. Your system is now
properly prepared. Follow the general instructions above to install **DisVis**.


### Windows

First install [*git*](https://git-scm.com/download) for Windows, as it also
comes with a handy *bash* shell.

For Windows it is easiest to install a Python distribution with NumPy (and many
other) packages included, such as [Anaconda](https://continuum.io/downloads).
Follow the installation instructions on their website.

Next open a *bash* shell that was shipped with *git*. Follow the general
instructions above to install **DisVis**.


## Usage

The general pattern to invoke *disvis* is

    disvis <pdb1> <pdb2> <distance-restraints-file>

where \<pdb1\> is the fixed chain, \<pdb2\> is the scanning chain and 
\<distance-restraints-file\> is a text-file
containing the distance restraints in the following format

     <chainid 1> <resid 1> <atomname 1> <chainid 2> <resid 2> <atomname 2> <mindis> <maxdis>

As an example
    
    A 18 CA F 27 CB 10.0 20.0

This puts a distance restraint between the CA-atom of residue 18 of 
chain A of pdb1 and the CB-atom of residue 27 of chain F of pdb2 that 
should be longer than or equal to 10A and smaller than or equal to 20A.
Comments can be added by starting the line with the pound sign (#) and empty
lines are ignored.


### Options

To get a help screen with available options and explanation type
            
    disvis --help

Some examples to get you started. To perform a 5.27 degree rotational search
and store the results in the directory *results/*

    disvis 1wcm_A.pdb 1wcm_E.pdb restraints.dat -a 5.27 -d results

Note that the directory is created if it doesn't exist.

To perform a 9.72 degree rotational search with 16 processors and a voxel
spacing of 2A

    disvis O14250.pdb Q9UT97.pdb restraints.dat -a 9.72 -p 16 -vs 2

Finally, to offload computations to the GPU and increase the maximum allowed
volume of clashes and decrease the minimum required volume of interaction, and
set the interaction radius to 2A

    disvis 1wcm_A.pdb 1wcm_E.pdb restraints.dat -g -cv 6000 -iv 7000 -ir 2

These examples have shown all the 9 available options.


### Output

*Disvis* outputs 5 files:

* *accessible_complexes.out*: a text file containing the number of complexes
consistent with a given number of restraints. Column 1 shows the number of
consistent restraints for each complex counted, denoted by N; column 2 shows
the number of complexes consistent with *exactly* N restraints; column 3 shows
the fraction of all complexes sampled that are consistent with *exactly* N
restraints; column 4 gives the number of complexes consistent with *at least* N
restraints, and is thus the cumulative sum of column 2; column 5 is again the
fraction of complexes consistent with *at least* N restraints, and also the
cumulative sum of column 3.

* *violations.out*: a text file showing how often a specific restraint is
violated for complexes consistent with a number of restraints.  The higher
the violation fraction of a specific restraint, the more likely it is to be a
false-positive. Column 1 shows the number of consistent restraints N, while
each following column indicates the violation fractions of a specific
restraint for complexes consistent with *at least* N restraints. Each row thus
represents the fraction of all complexes consistent with *at least* N
restraints that violated a particular restraint.

* *accessible_interaction_space.mrc*: a density file in MRC format. The density
represents the center of mass of the scanning chain conforming to the maximum
found consistent restraints at every position in space. The density can be
inspected by opening it together with the fixed chain in a molecular viewer
(UCSF Chimera is recommended for its easier manipulation of density data, but
also PyMol works).

* *z-score.out*: a text file giving the z-score for each restraint. The higher
the score, the more likely the restraint is a false-positive. Z-scores above
1.0 are explicitly mentioned in the output. The first column indicates the
restraint; column 2 gives the average violation fraction, i.e. the average of
the corresponding column in violations.out; column 3 represents the standard
deviation of the average violation fraction; and column 4 finally gives the
z-score.

* *disvis.log*: a log file showing all the parameters used, together with date
and time indications.


### Visualizing the accessible interaction space

The accessible interaction space consistent with at least N restraints can be
visualized by opening the fixed and scanning chain used in the *disvis* run
together with the *accessible_interaction_space.mrc* output file with UCSF
Chimera. The density slider in Chimera can be adjusted to change N.


Licensing
---------

If this software was useful to your research please cite us

**G.C.P. van Zundert and A.M.J.J. Bonvin.** DisVis: Quantifying and visualizing
accessible interaction space of distance-restrained biomolecular complexes.
*Bioinformatics*, Advanced Online Publication (2015).

Apache Licence 2.0


## Tested platforms

| Operating System| CPU single | CPU multi | GPU |
| --------------- | ---------- | --------- | --- |
|Linux            | Yes        | Yes       | Yes |
|MacOSX           | Yes        | Yes       | No  |
|Windows          | Yes        | Fail      | No  |

The GPU version has been tested on:
NVIDIA GeForce GTX 680 and AMD Radeon HD 7730M for Linux
