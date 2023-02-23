Setup Instructions
==================


Clone the repository:

::

    git clone https://github.com/seanandrews/csalt.git


For now (until a pip install is ready), please work in the csalt directory:

::

    cd csalt


Configure your workspace.  For this example, we'll assume that your current 
location is /path_to_csalt/csalt and that we want to pack away all our data 
files into a new subdirectory called storage:

::

    mkdir storage


Now make sure you have the necessary Python dependencies.  For simulating 
datasets (our only task right now), the required packages are:
	- numpy
	- scipy
	- astropy
	- matplotlib
	- h5py
	- vis_sample
The first four are presumably already installed (they come default in, for 
example, any Anaconda python setup).  The latter two you might not have yet.  
In any case, you can check with 

::

    pip show <package_name>

and you can install with

::

    pip install <package_name>	

or as appropriate with a suitable alternative package manager.

Now make sure you have CASA set up properly.  This means you have an alias 
for the command `casa' that launches properly, e.g. ::

    alias casa /path_to_casa/bin/casa

(this presumably already exists).  I have tested csalt with CASA v5.7.2 and 
v5.8.0.  It will *not* work (yet) in modular CASA, since I don't have easy 
access to a functional version of that (eventually that will be the default, 
but the timing depends on the CASA team's expansion to other Linux OS 
architectures).  You also need to be able to use the h5py package inside of 
CASA, and setting that up is a little wonky.  [Note: I cannot figure out how 
to install external packages in CASA v6.4.0.  If one of you can, please let 
me know and I will test the code there!]  We'll follow the path laid out by 
the Astropy folks:

::

    casa
    <CASA>: from setuptools.command import easy_install
    <CASA>: easy_install.main(['--user', 'pip'])
    <CASA>: exit

    casa
    <CASA>: import subprocess, sys
    <CASA>: subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
                                   '--user', 'h5py'])
    <CASA>: exit

    casa
    <CASA>: import h5py

That last call is just a check; it shouldn't produce any errors.

[Caitlyn] To run csalt in CASA v6.4.0 (which has casatools compatible with the latest macOS versions) and install h5py in CASA, you need to create an initial configuration script named config.py which CASA reads on start up. This file needs to be placed in the user root .casa folder (~/.casa) prior to importing any external packages. The full list of available flags that can be specified in this config.py script can be found at https://casadocs.readthedocs.io/en/stable/api/configuration.html, but for our purposes we just need to set 

::

 user_site=True
 
and assuming that the h5py module has been correctly installed via pip outside of CASA, it will now be accessible within the CASA python environment.
