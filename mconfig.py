"""
    This is the main control file, used to generate synthetic data or to model
    real (or synthetic) datasets.
"""
import numpy as np


# file naming
basename = 'test'
template = 'uvtest'


# controls
gen_data = False
fit_data = False
overwrite_template = True

# auxiliary file locations
simobs_dir = '/pool/asha0/casa-release-5.7.2-4.el7/data/alma/simmos/'


# Model parameters
# ----------------
npars = 9
pars  = np.zeros(npars)
pars[0]  = 30.		# inclination angle (degrees)
pars[1]  = 140.		# position angle (degrees)
pars[2]  = 1.0		# stellar mass (Msun)
pars[3]  = 200.		# "line radius" (AU)
pars[4]  = 2.		# emission surface height at r_0 (AU)
pars[5]  = 1.		# radial power-law index of emission surface height
pars[6]  = 150.		# temperature at r_0 (K)
pars[7]  = 0.5		# radial power-law index of temperature surface
pars[8]  = 20.		# maximum brightness temperature of back side

FOV = 8.
Npix = 256
dist = 150.
rmax = 300.

chanstart_out = 3.65e3
chanwidth_out = 80.
nchan_out = 5
RMS = 5.0
 


# Simulated observations parameters
# ---------------------------------
# spectral settings
dfreq0   = 61.035e3    # in Hz
restfreq = 230.538e9          # in Hz
vtune    = 4.0e3
vsys     = 4.0e3             # in m/s
vspan    = 0.5e3             # in m/s
spec_oversample = 3     

# spatial settings
RA   = '04:30:00.00'
DEC  = '25:00:00.00'
HA   = '0.0h'
date = '2021/12/01'

# observation settings
config = '5'
ttotal = '5min'
integ  = '30s'





