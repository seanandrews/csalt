"""
"""
import numpy as np


### data configuration and storage
# file assignments
orig_dir = '/data/sandrews/LP/sa_work/Sz129/'
orig_MS  = orig_dir+'Sz129_combined_CO_selfcal.ms.contsub'
basename = 'Sz129'
extname  = ''
outdir   = 'data/'+basename+'/'
dataname = outdir+basename+extname
preserve_tmp = False

# reduction parameters
nu_rest = 230.538e9			# spectral line rest frequency (Hz)
V_sys = 4.1e3				# estimate of systemic velocity (m/s)
dVb   = 8e3				# desired +/- range around V_sys (m/s)
bounds_V = [V_sys-dVb, V_sys+dVb]	# LSRK velocity bounds to extract (m/s)
chpad = 5				# pad channels on each end of bounds_V
tavg = '30s'				# time-averaging interval for orig_MS
					# (no averaging == '')


proc_D = True
do_img = [True, True, True]		# for data, model, residual

# model parameters
incl, PA, xoff, yoff, vsys = 32., 153.+180, 0.005, 0.006, 4.1e3
mstar, z0, psi, r_l = 0.7, 2.3, 1.0, 200.
Tb0, q, Tback, dV0 = 205., 0.5, 20., 348.

FOV, Npix, dist, r0 = 5.12, 512, 160., 10.


# imaging parameters
chanstart = '-6.40km/s'
chanwidth = '0.35km/s'
nchan_out = 60
imsize = 512
cell = '0.02arcsec'
scales = [0, 10, 30, 100, 200]
gain = 0.1
niter = 50000
robust = 1.0
threshold = '3mJy'
uvtaper = '0.04arcsec'
