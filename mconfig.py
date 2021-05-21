"""
    This is the main control file, used to generate synthetic data or to model
    real (or synthetic) datasets.
"""
import numpy as np


# file naming
basename = 'test_Sz129'
template = 'Sz129'


# controls
gen_from_scratch = False
fit_data = False
overwrite_template = True

# imaging
do_img = ['', '_noisy']
imsize = 256
cell = '0.03arcsec'
robust = 0.5
thresh = '5.0mJy'
cscales = [0, 5, 10, 15]

# auxiliary file locations
simobs_dir = '/pool/asha0/casa-release-5.7.2-4.el7/data/alma/simmos/'


# Model free parameters
# ---------------------
npars = 13
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
pars[9]  = 300.		# line-width at r_0 (m/s)
pars[10] = 4e3		# systemic velocity (m/s)
pars[11] = 0.0		# RA offset (arcsec)
pars[12] = 0.0		# DEC offset (arcsec)

# Model fixed parameters
# ----------------------
# properties for sky-projected cube
FOV  = 8.		# full field of view (arcsec)
Npix = 256		# number of pixels per FOV
dist = 150.		# distance (pc)
rmax = 300.		# maximum radius of emission (au)

# desired output LSRK velocity channels
chanstart_out = -6.40e3	# m/s
chanwidth_out = 320.	# m/s
nchan_out = 65		# 

# noise properties
RMS = 5.0		# RMS noise per channel for natural-weight image (mJy)
 


# Template observational parameters
# ---------------------------------
# spectral settings
dfreq0   = 61.035e3    	# native channel spacing (Hz)
restfreq = 230.538e9  	# spectra line rest frequency (Hz)
vtune    = 4.0e3	# LSRK velocity tuning for central channel (m/s)
vspan    = 0.5e3     	# +/- velocity width around vtune for simulation (m/s)
spec_oversample = 3   	# over-sampling factor for spectral signal processing

# spatial settings
RA   = '04:30:00.00'	# phase center RA
DEC  = '25:00:00.00'	# phase center DEC
HA   = '0.0h'		# hour angle at start of EB
date = '2021/12/01'	# UTC date for start of EB

# observation settings
config = '5'		# ALMA antenna configuration index (e.g., '5' = C43-5)
ttotal = '5min'		# total EB time
integ  = '30s'		# integration time
