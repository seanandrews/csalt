"""
"""

import numpy as np

# naming
basename = 'simple2-default'
_ext = '_pure'

reduced_dir = 'data/'+basename+'/'
dataname = reduced_dir+basename


# Model parameters
incl  = 40.
PA    = 130.
mstar = 0.7
r_l   = 200.
z0    = 2.3
psi   = 1.
T0    = 205.
q     = -0.5
Tmaxb = 20.
sigV0 = 348.
Vsys  = 5.2e3
dx    = 0.
dy    = 0.
pars  = np.array([incl, PA, mstar, r_l, z0, psi, T0, q, Tmaxb,
                  sigV0, Vsys, dx, dy])


# Fixed parameters
nu_rest = 230.538e9	# spectral line rest frequency (Hz)
FOV  = [6.4]            # full field of view (arcsec)
Npix = [256]            # number of pixels per FOV
dist = 150.             # distance (pc)


# --------------------
# for reduction only (if using CASA_scripts/format_data.py; only needed for 
# _real_ observations once)
storage_dir = 'synth_storage/'
in_MS = storage_dir+basename+'/'+basename+_ext

tavg = ['']
V_bounds = [(5.2 - 10)*1e3, (5.2 + 10)*1e3]
bounds_pad = 3
# --------------------




# likelihood calculation information
chpad = 3
chbin = [2]



gen_msk = True				
gen_img = [True, True, True]		# for data, model, residual

# imaging parameters
chanstart = '0.40km/s'
chanwidth = '0.16km/s'
nchan_out = 60
imsize = 128
cell = '0.05arcsec'
scales = [0, 10, 30, 50]
gain = 0.1
niter = 50000
robust = 2.0
threshold = '4mJy'
uvtaper = ''
