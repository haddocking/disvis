# DisVis


## About DisVis

DisVis is a Python package and command line tool to visualize and quantify 
the accessible interaction space of distance restrained biomolecular complexes.


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


## Installation

If the requirements are met, DisVis can be installed by opening a shell
and typing

    git clone https://github.com/haddocking/disvis.git
    cd disvis
    (sudo) python setup.py install

After installation, the command line tool *disvis* should be at your disposal.

If you are starting from a clean system, then read the installation instructions 
below to prepare your particular operating system.


### Linux

First install git and check whether the python header files are available

    (sudo) apt-get install git python-dev

Next we will install [*pip*](https://pip.pypa.io/en/latest/installing.html), 
the official Python package manager. Follow the link, and install *pip* using
their installation instructions.

The final step to prepare you system is installing the Python dependencies

    (sudo) pip install numpy cython

Wait till the compilation and installion is finished (this might take awhile).
Your system is now ready to run DisVis. Follow the general instructions above to install DisVis.


### MacOSX (10.7+)

First install [*git*](https://git-scm.com/download) for MacOSX.
Next we will install [*pip*](https://pip.pypa.io/en/latest/installing.html), 
the official Python package manager. Follow the link, and install *pip* using
their installation instructions.

The final step to prepare you system is installing the Python dependencies.
Open a shell and type

    (sudo) pip install numpy cython

Wait till the compilation and installion is finished (this might take awhile).
Your system is now ready to run DisVis. Follow the general instructions above to install DisVis.


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

*disvis* outputs a file *accessible_interaction_space.mrc* and prints the 
number of accessible complexes per number of consistent distance restraints. 
The *.mrc* file can be straightforwardly opened with UCSF Chimera and PyMol.


### Options

To get a help screen with available options
            
    disvis --help



Licensing
---------

If this software was useful to your research please cite us

**G.C.P. van Zundert and A.M.J.J. Bonvin** (2015) DisVis: Visualizing and
quantifying the accessible interaction space of distance restrained biomolecular complexes.
*Bioinformatics* (submitted).

MIT licence
