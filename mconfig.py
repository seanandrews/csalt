"""
    This is the main control file, used to generate synthetic data or to model
    real (or synthetic) datasets.
"""
import numpy as np


# model name
basename = 'modeltest'

# template setup
template = 'templatetest'

# auxiliary file locations
simobs_dir = '/pool/asha0/casa-release-5.7.2-4.el7/data/alma/simmos/'


# Model free parameters
# ---------------------
pars = np.array([40, 130, 0.7, 200, 2.3, 1, 205, 0.5, 20, 348, 5.2e3, 0, 0])
npars = len(pars)
#pars[0]  = inclination angle (degrees)
#pars[1]  = position angle (degrees)
#pars[2]  = stellar mass (Msun)
#pars[3]  = "line radius" (AU)
#pars[4]  = emission surface height at r_0 (AU)
#pars[5]  = radial power-law index of emission surface height
#pars[6]  = temperature at r_0 (K)
#pars[7]  = radial power-law index of temperature surface
#pars[8]  = maximum brightness temperature of back side
#pars[9]  = line-width at r_0 (m/s)
#pars[10] = systemic velocity (m/s)
#pars[11] =  RA offset (arcsec)
#pars[12] = DEC offset (arcsec)


# Model fixed parameters
# ----------------------
# properties for sky-projected cube
FOV  = 10.		# full field of view (arcsec)
Npix = 512		# number of pixels per FOV
dist = 150.		# distance (pc)
rmax = 700.		# maximum radius of emission (au)

# desired output LSRK velocity channels
chanstart_out = -4.8e3	# m/s

# noise properties (RMS per channel in naturally weighted images (mJy))
RMS = 5.3
 


# Template observational parameters
# ---------------------------------
# spectral settings
dfreq0   = 122.0703125*1e3 	# native channel spacing (Hz)
restfreq = 230.538e9  		# rest frequency (Hz)
vtune    = 4.0e3	# LSRK velocity tuning for central channel (m/s)
vspan    = 12.5e3     	# +/- velocity width around vtune for simulation (m/s)
spec_oversample = 5   	# over-sampling factor for spectral signal processing

# spatial settings
RA   = '16:00:00.00'	# phase center RA
DEC  = '-40:00:00.00'	# phase center DEC
HA   = '0.0h'		# hour angle at start of EB
date = '2022/05/20'	# UTC date for start of EB

# observation settings
config = '5'		# antenna config ('5' = C43-5)
ttotal = '15min'	# total EB time
integ  = '30s'		# integration time

# imaging
do_img = ['', '_noisy']
imsize = 256
cell = '0.03arcsec'
robust = 0.5
thresh = str(RMS)+'mJy'
cscales = [0, 5, 10, 15]
