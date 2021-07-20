"""
"""
import numpy as np

### data configuration and storage
# inputs 
orig_dir = 'synth_storage/simp3/'
orig_MS  = orig_dir+'simp3-nmm.pure.ms'

# outputs
basename = 'simp3-nmm'
extname  = '.pure'
outdir   = 'data/'+basename+'/'
dataname = outdir+basename+extname

# data reduction parameters
nu_rest = 230.538e9			# spectral line rest frequency (Hz)
V_sys = 5.2e3				# estimate of systemic velocity (m/s)
dVb   = 10e3				# desired +/- range around V_sys (m/s)
bounds_V = [V_sys-dVb, V_sys+dVb]	# LSRK velocity bounds to extract (m/s)
chpad = 5				# pad channels on each end of bounds_V
tavg = ''				# time-averaging interval for orig_MS
					# (no averaging == '')



gen_msk = True				
gen_img = [True, True, True]		# for data, model, residual

# model parameters
# free
incl, PA, xoff, yoff, vsys = 40., 130, 0, 0, 5.2e3
mstar, z0, psi, r_l = 0.7, 2.3, 1, 200.
Tb0, q, Tback, dV0 = 205., 0.5, 20., 348.

# fixed
FOV, Npix, dist = 10.24, 256, 150.
rmax = dist * 0.5 * FOV 		# maximum radius of emission (au)



# imaging parameters
chanstart = '0.40km/s'
chanwidth = '0.16km/s'
nchan_out = 60
imsize = 128
cell = '0.05arcsec'
scales = [0, 10, 30, 100]
gain = 0.1
niter = 50000
robust = 2.0
threshold = '7mJy'
uvtaper = ''
