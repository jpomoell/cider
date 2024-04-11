CIDER: Coronal magnetIc fielD ExtRapolation tools
=================================================

Tools for extrapolating the solar coronal magnetic field.


Limitations and planned work
----------------------------

CIDER is currently early-level pre-release software, with several improvements to be implemented before release.
The major current limitation is only supporting uniform grids.

Planned improvements (in approximate order of implementation):

  * Non-uniform radial coordinate in PFSS and MFW models
  * Implement support for high-resolution extrapolations by adding computations on patches
  * Use SunPy facilities throughout, in particular coordinates and units 
  * Rewrite magnetogram loaders so as not to alter metadata directly
  * Parallel/multithreaded field line tracing
  * Documentation and more notebook examples
  * Migrate to a newer FISHPACK version
  * Easier installation (using meson)


Installation
------------

CIDER is easiest to install using the [Anaconda](https://www.anaconda.com/) or
the [Miniconda](https://conda.io/miniconda.html) Python distribution.

The basic environment including the basic required dependencies can be created using 
the provided specification:

    conda env create -f environment.yml

After the environment has been activated, the external FORTRAN library FISHPACK needs to be
installed. This is accomplished by the following steps:

    1. Change directory to the CIDER root directory
    2. cd external
    3. wget https://github.com/NCAR/NCAR-Classic-Libraries-for-Geophysics/raw/main/FishPack/fishpack4.1.tar.gz
    4. tar -xvzf fishpack4.1.tar.gz
    5. cd ../cider/solvers/fishpack
    6. make

If the compilation fails, the makefile (cider/solvers/fishpack/makefile) most likely will be needed to be tailored
for your system.

Finally, the package [pysmsh](https://github.com/jpomoell/pysmsh) is required to be installed.
Installing it requires only cloning the repo and making it visible to the CIDER environment.