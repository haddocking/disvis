# DisVis


## About DisVis

**DisVis** is a Python package and command line tool to visualize and quantify 
the accessible interaction space of distance restrained binary biomolecular complexes.
It performs a full and systematic 6 dimensional search of the three translational
and rotational degrees of freedom to determine the number of complexes consistent
with the restraints. In addition, it outputs the percentage of restraints being violated
and a density that represents the center-of-mass position of the scanning chain corresponding 
to the highest number of consistent restraints at every position in space.


## Requirements

* Python2.7
* NumPy
* Cython

Optional for faster CPU version

* FFTW3
* pyFFTW

For GPU hardware acceleration the following packages are also required

* OpenCL1.1+
* [pyopencl](https://github.com/pyopencl/pyopencl)
* [clFFT](https://github.com/clMathLibraries/clFFT)
* [gpyfft](https://github.com/geggo/gpyfft)

Recommended for easy installation

* [git](https://git-scm.com/download)
* [pip](https://pip.pypa.io/en/latest/installing.html)


## Installation

If the requirements are met, **DisVis** can be installed by opening a terminal
and typing

    git clone https://github.com/haddocking/disvis.git
    cd disvis
    python setup.py install

The last command might require administrator priviliges for system wide installation.
After installation, the command line tool *disvis* should be at your disposal.

If you are starting from a clean system, then read the installation instructions 
below to prepare your particular operating system.


### Linux

First install git and check whether the Python header files and the Python
package manager, *pip*, are available

    sudo apt-get install git python-dev python-pip

The final step to prepare you system is installing the Python dependencies

    sudo pip install numpy cython

Wait untill the compilation and installion is finished (this might take awhile).
Your system is now ready. Follow the general instructions above to install **DisVis**.


### MacOSX (10.7+)

First install [*git*](https://git-scm.com/download) for MacOSX.
Next we will install [*pip*](https://pip.pypa.io/en/latest/installing.html), 
the official Python package manager. Follow the link, and install *pip* using
their installation instructions.

The final step to prepare you system is installing the Python dependencies.
Open a shell and type

    sudo pip install numpy cython

Wait till the compilation and installion is finished (this might take awhile).
Your system is now proporly prepared. Follow the general instructions above to install DisVis.


### Windows

First we will install [*git*](https://git-scm.com/download) for Windows, as it also comes
with a handy *bash* shell.

For Windows it easiest to install a Python distribution with NumPy and Cython
(and many other) packages available, such as [Anaconda](https://continuum.io/downloads).
Follow the installation instructions on their website.

Next open a *bash* shell that was shipped with *git*. Follow the general instructions
above to install DisVis.


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

Some examples to get you started. To perform a 5.27 degree rotational search and store the results
in the directory *results/*

    disvis 1wcm_A.pdb 1wcm_E.pdb restraints.dat -a 5.27 -d results

Note that the directory is created if it doesn't exist.

To perform a 9.72 degree rotational search with 16 processors and a voxel spacing of 2A

    disvis O14250.pdb Q9UT97.pdb restraints.dat -a 9.72 -p 16 -vs 2

Finally, to offload computations to the GPU and increase the maximum allowed volume of clashes 
and decrease the minimum required volume of interaction, and set the interaction radius to 2A.

    disvis 1wcm_A.pdb 1wcm_E.pdb restraints.dat -g -cv 6000 -iv 7000 -ir 2

These examples have shown all the 9 available options.


## Output

*disvis* outputs 4 different files:

* *accessible_complexes.out*: a text file containing the number of complexes consistent with
a number of restraints. 
* *violations.out*: a text file showing how often a specific restraint is violated for each number
of consistent restraints. This helps in identifying which restraint is most likely a false-positive
if any.
* *accessible_interaction_space.mrc*: a density file in MRC format. The density represents the
center of mass of the scanning chain conforming to the maximum found consistent restraints at
every position in space. The density can be inspected most naturally by opening it together with the
fixed chain in a molecular viewer (UCSF Chimera is recommended for its easier manipulation of density
data, but also PyMol works).
* *disvis.log*: a log file showing all the parameters used, together with date and time indications.


Licensing
---------

If this software was useful to your research please cite us

**G.C.P. van Zundert and A.M.J.J. Bonvin** (2015) DisVis: Visualizing and
quantifying the accessible interaction space of distance restrained biomolecular complexes.
*Bioinformatics* (submitted).

MIT licence
