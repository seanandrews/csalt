import os
import sys
import numpy as np
import scipy.constants as sc
from parametric_disk_CSALT0 import parametric_disk as pardisk_csalt
from csalt.model import *
from csalt.helpers import *
from astropy.io import fits
import matplotlib.pyplot as plt


# set velocities
velax = np.arange(-3000, 3000, 200.)

# Calculate the cube
pars = np.array([30, 150, 1.1, 120, 0.3, 1.5, 120, -0.5, 20, 217, 
                 2.2, -1, 0, 0, 0]

fixed = 230.538e9, 10.23, 1024, 150, {}
cube = pardisk_csalt(velax, pars, fixed)

# create FITS
cube_to_fits(cube, 'cube_test.fits', RA=240., DEC=-30.)
