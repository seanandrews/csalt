"""
"""

import numpy as np


output_dir = 'storage/'
template_dir = output_dir+'obs_templates/'
storage_dir = output_dir+'synth_storage/'
reduced_dir = output_dir+'data/'
casalogs_dir = output_dir+'CASA_logs/'


# naming
basename = 'simple2-default'
_ext = '_pure'
_fitnote = '_noCOV'

dataname = reduced_dir+basename+'/'+basename
fitname = output_dir+'fitting/'

# Model parameters
incl  = 40.
PA    = 130.
mstar = 0.7
r_l   = 200.
z0    = 2.5
psi   = 1.
T0    = 115.
q     = -0.5
Tmaxb = 20.
sigV0 = 261. 
ltau0 = np.log10(500.)
ppp   = -1.
Vsys  = 5.2e3
dx    = 0.
dy    = 0.
pars  = np.array([incl, PA, mstar, r_l, z0, psi, T0, q, Tmaxb,
                  sigV0, ltau0, ppp, Vsys, dx, dy])


# Fixed parameters
nu_rest = 230.538e9	# spectral line rest frequency (Hz)
FOV  = [6.375]          # full field of view (arcsec)
Npix = [256]            # number of pixels per FOV
dist = 150.             # distance (pc)
cfg_dict = {}


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
vra_fit = [0, 10400]
vra_cens = None
max_steps = 3000
burnin = 0

init_incl  = [20, 60]	
init_PA    = [100, 160]	
init_mstar = [0.4, 1.0]	
init_r_l   = [150, 250]	
init_z0    = [1.0, 3.0]	
init_psi   = [0.5, 1.5]	
init_T0    = [150, 250]	
init_q     = [-1., 0.]	
init_Tmaxb = [10, 40]
init_sigV0 = [150, 500]	
init_ltau0 = [2, 3]
init_ppp   = [-2, 0]
init_Vsys  = [4.7e3, 5.7e3]
init_dx    = [-0.1, 0.1]
init_dy    = [-0.1, 0.1]
init_ = np.array([init_incl, init_PA, init_mstar, init_r_l, init_z0, init_psi,
                  init_T0, init_q, init_Tmaxb, init_sigV0, init_ltau0, init_ppp,
                  init_Vsys, init_dx, init_dy])
nwalkers = 5 * len(pars)

pt_incl, pp_incl = 'uniform', [-90., 90.]
pt_PA, pp_PA = 'uniform', [0., 360.]
pt_mstar, pp_mstar = 'uniform', [0., 5.]
pt_r_l, pp_r_l = 'uniform', [10., 0.5 * np.min(FOV) * dist]
pt_z0, pp_z0 = 'uniform', [0., 5.]
pt_psi, pp_psi = 'uniform', [0., 2.]
pt_T0, pp_T0 = 'uniform', [5., 1000.]
pt_q, pp_q = 'uniform', [-1.5, 0.]
pt_Tmaxb, pp_Tmaxb = 'uniform', [5., 100.]
pt_sigV0, pp_sigV0 = 'uniform', [50., 1000.]
pt_ltau0, pp_ltau0 = 'uniform', [1.3, 3.1]
pt_ppp, pp_ppp = 'uniform', [-2, 0]
pt_Vsys, pp_Vsys = 'uniform', [4.2e3, 6.2e3]
pt_dx, pp_dx = 'uniform', [-0.25, 0.25]
pt_dy, pp_dy = 'uniform', [-0.25, 0.25]
priors_ = {"types": [pt_incl, pt_PA, pt_mstar, pt_r_l, pt_z0, pt_psi, pt_T0,
                     pt_q, pt_Tmaxb, pt_sigV0, pt_ltau0, pt_ppp, 
                     pt_Vsys, pt_dx, pt_dy],
           "pars": [pp_incl, pp_PA, pp_mstar, pp_r_l, pp_z0, pp_psi, pp_T0,
                    pp_q, pp_Tmaxb, pp_sigV0, pp_ltau0, pp_ppp, 
                    pp_Vsys, pp_dx, pp_dy]}



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
threshold = '5mJy'
uvtaper = ''
