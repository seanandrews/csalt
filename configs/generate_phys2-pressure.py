"""
    This is the main control file, used to generate synthetic data or to model
    real (or synthetic) datasets.
"""

import numpy as np


# locators
output_dir = 'storage/'
template_dir = output_dir+'obs_templates/'
storage_dir = output_dir+'synth_storage/'
reduced_dir = output_dir+'data/'
radmc_dir = output_dir+'radmc/'
casalogs_dir = output_dir+'CASA_logs/'
simobs_dir = '/pool/asha0/casa-release-5.7.2-4.el7/data/alma/simmos/'


# naming
basename = 'phys2-pressure'
in_MS = storage_dir+basename+'/'+basename
dataname = reduced_dir+basename+'/'+basename
radmcname = radmc_dir+basename


# observation settings
template = ['lmm']
config = ['6']
ttotal = ['20min']
tinteg = ['30s']
date = ['2022/07/15']
HA_0 = ['-0.5h']
RMS = [3.7]             # desired naturally-weighted RMS in mJy/beam/channel


# spectral settings
dnu_native = [122.0703125 * 1e3]	# native channel spacing (Hz)
nu_rest = 230.538e9	# rest frequency (Hz)
V_tune  = [0.0e3]	# LSRK velocity tuning for central channel (m/s)
V_span  = [5.0e3]	# +/- velocity width around vtune for simulation (m/s)
nover = 1	# spectral over-sampling factor (for SRF convolution)


# spatial settings
RA   = '16:00:00.00'	# phase center RA
DEC  = '-40:00:00.00'	# phase center DEC
RA_pieces = [np.float(RA.split(':')[i]) for i in np.arange(3)]
RAdeg = 15 * np.sum(np.array(RA_pieces) / [1., 60., 3600.])
DEC_pieces = [np.float(DEC.split(':')[i]) for i in np.arange(3)]
DECdeg = np.sum(np.array(DEC_pieces) / [1., 60., 3600.])


# reduction settings
tavg = ['']
V_bounds = [(0. - 10)*1e3, (0. + 10)*1e3]
bounds_pad = 3


# Model parameters
r0 = 10 * 1.496e13
incl  = 40.
PA    = 130.
mstar = 0.7
r_l   = 150.

Tmid0 = 75.
Tatm0 = 125.
qmid  = -0.5
qatm  = -0.5
hs_T  = 0.250
ws_T  = 0.005

p1 = -1.0
p2 = 2.0
Sigma0_gas = 150.

xmol = 1e-5
depl = 1e-10
zrmin, zrmax = 0.23, 0.27
rmin, rmax = 0.1, 3 * r_l

xi = 0.0

Vsys  = 0.0
dx    = 0.
dy    = 0.
pars  = np.array([incl, PA, mstar, r_l, Tmid0, Tatm0, qmid, qatm, hs_T, ws_T,
                  Sigma0_gas, p1, p2, xmol, depl, zrmin, zrmax, rmin, rmax, xi, 
                  Vsys, dx, dy])


# Fixed parameters
FOV  = [6.375]		# full field of view (arcsec)
Npix = [256]  		# number of pixels per FOV
dist = 150.		# distance (pc)



# instantiate RADMC-3D parameters
grid_params = { 'spatial': {'nr': 256, 'nt': 128, 'r_min': 0.1, 'r_max': 1000 },
                'cyl': { 'nr': 2048, 'nt': 2048, 'r_min': 0.1, 'r_max': 1000,
                         'z_min': 0.001, 'z_max': 500 } }

setup_params = { 'incl_dust': 0, 'incl_lines': 1, 'nphot': 10000000, 
                 'scattering': 'Isotropic', 'camera_tracemode': 'image',
                 'molecule': 'co', 'transition': 2, 
                 'dustspec': 'DIANAstandard' }

cfg_dict = {'radmcname': radmcname, 
            'grid_params': grid_params, 'setup_params': setup_params,
            'dPdr': True, 'selfgrav': False}
