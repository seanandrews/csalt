Comparison with data
====================

Parameters needed
-----------------

Now that we know how to use csalt to configure a simulated MS 
file from the model provided, we can compare the simulated data 
with actual data to determine the accuracy of the model. In order
to compare the two, we need to provide csalt with the original MS
file as well as some additional information:

  - nu_rest: the rest wavelength of the spectral line being looked at
  - FOV: the full field of view (arcsec)
  - Npix: number of pixels per FOV (note: pixsize = FOV/(Npix-1))
  - dist: distance (pc)
  - cfg_dict: passable dictionary of kwargs
  - vra: for velocity bounds (set to None if no restrictions)
  - vcensor: for channel censoring (set to None if no restrictions)
  - chbin: binning factor (there are limited options that work within csalt for this - 2 is the default value)
  
Converting the MS file to h5 format
-----------------------------------

csalt needs the MS file to be converted into h5 format, which reduces
the size of the file and makes the comparison more efficient. To convert
the MS file to h5, we use the function found in csalt/CASA_scripts/ms_to_hdf5.py
within CASA inside the csalt directory as follows:

::

  <CASA> import 
  
  
  
(how I previously did it - copied lines from the file to debug - will need to test as running the script inside CASA. Remember you need enough memory to read it in)
