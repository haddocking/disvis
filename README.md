DisVis
======


About DisVis
------------

DisVis is a Python package and program to visualize the accessible interaction space of distance constrained biomolecular complexes.


Requirements
------------

* Python2.7
* [NumPy]
* [Cython]

Optional for faster CPU version

* [FFTW3]
* [pyFFTW]

For GPU hardware acceleration the following packages are also neccesary

* OpenCL1.1 or higher distribution
* [pyopencl] [1]
* [clFFT] [2]
* [gpyfft] [3]


Installation
------------

For installation (this might require root access)

    python setup.py install

This will built the extension file in src/ and install DisVis.


Examples
--------

DisVis comes with a script to quickly perform a scan

    disvis <pdb1> <pdb2> <distance-restraints-file>

where \<pdb1\> is the fixed chain, \<pdb2\> is the scanning chain and \<distance-restraints-file\> is a text-file
containing the distance restraints in the following format

     <chainid 1> <resid 1> <atomname 1> <chainid 2> <resid 2> <atomname 2> <distance>

As an example
    
    A 18 CA A 20 CA 20.0

This puts a 20A distance restraint between the CA-atom of residue 18 of chain A of pdb1 and the CA-atom of residue 20 of chain A of pdb2.

To get a help screen with available options
            
    disvis --help

The script outputs a file *accessible_interaction_space.mrc* and prints the number of accessible complexes per number of obeying distance restraints. 
The *.mrc* file can be straightforwardly opened with UCSF Chimera and PyMol.

Licensing
---------

MIT licence

[1]: https://github.com/pyopencl/pyopencl
[2]: https://github.com/clMathLibraries/clFFT
[3]: https://github.com/geggo/gpyfft
[4]: 
