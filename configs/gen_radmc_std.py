"""
    This is the configuration file for generating a synthetic dataset from 
    scratch in the csalt architecture.  It is imported as a Python modeule in 
    various subroutines, and will be copied into the directory

        outputbase_dir/reduced_dir/basename/

    for future reference once the data generation is complete.
"""

import numpy as np
import scipy.constants as sc

"""
    LOCATORS:
    These set the desired locations and naming conventions of the outputs, as 
    well as the locations of necessary ancillary information.
"""
# base path
outputbase_dir = '/pool/asha0/SCIENCE/csalt/storage/'

# path to simobserve outputs and blank template .MS files
template_dir = outputbase_dir+'obs_templates/'

# path to storage space for "raw" synthetic dataset files
synthraw_dir = outputbase_dir+'synth_storage/'

# path to concatenated, "reduced" dataset files
reduced_dir = outputbase_dir+'data/'

# path to radmc model files
radmc_dir = outputbase_dir+'radmc/'

# path to hard-copies of CASA logs
casalogs_dir = outputbase_dir+'CASA_logs/'

# path to CASA/simobserve-format antenna configuration files
antcfg_dir = '/pool/asha0/casa-release-5.7.2-4.el7/data/alma/simmos/'

# path to Rich's keplerian masking script
kepmask_dir = '/home/sandrews/mypy/keplerian_mask/'

# datafile naming base
basename = 'radmc_std'

# synthetic "raw" naming base
in_MS = synthraw_dir+basename+'/'+basename

# synthetic "reduced" naming base
dataname = reduced_dir+basename+'/'+basename

# radmc output naming base
radmcname = radmc_dir+basename



"""
    SIMULATED OBSERVATION SETTINGS:

"""
# array observing settings
template = ['std'] 				# template names 
config = ['alma.cycle8.5'] 			# antenna location lists 
date = ['2023/03/23'] 				# observation dates (UTC)
HA_0 = ['-0.25h']				# HAs at observing starts
ttotal = ['30min'] 				# total on-source times
tinteg = ['30s']				# integration times per stamp

# spectral settings
dnu_native = [122070.3125] 			# native channel spacings (Hz)
nu_rest = 230.538e9                		# rest frequency (Hz)
V_tune  = [4.0e3] 				# LSRK tunings at centers (m/s)
V_span  = [15.0e3]				# +/- ranges around V_tune (m/s)
nover = 1       				# over-sampling factor (for SRF)

# spatial settings
RA = '16:00:00.00'    				# phase center RA
DEC = '-40:00:00.00'   				# phase center DEC

# noise model settings
RMS = [5.3]					# desired RMS (mJy/beam/chan)



"""
    DATA REDUCTION SETTINGS:

"""
tavg = ['']					# time-averaging intervals
V_bounds = [5.0e3-12e3, 5.0e3+12e3]



"""
    INPUT MODEL PARAMETERS:

"""
incl  = 40.
PA    = 130.
mstar = 1.0

Tmid0 = 65.	
Tatm0 = 150. 
qmid  = -0.5
qatm  = -0.5
a_z = 1.75
w_z = 0.25

Sig0  = 10.	#19.3
p1    = -1.0
p2    = np.inf
r_l   = 250.

xmol  = 1e-4
depl  = 1e-20
Tfrz  = 20.
zrmax = 2.5 
rmin  = 0.1
rmax  = 1.1 * r_l

xi    = 0.0

Vsys  = 5.0e3
dx    = 0.
dy    = 0.
pars  = np.array([incl, PA, mstar, r_l, Tmid0, Tatm0, qmid, qatm, a_z, w_z,
                  Sig0, p1, p2, xmol, depl, Tfrz, zrmax, rmin, rmax, xi,
                  Vsys, dx, dy])

# fixed inputs
FOV  = [6.375]					# full FOV (arcsec)
Npix = [256]			 		# number of pixels per FOV
						# note: pixsize = FOV/(Npix-1)
dist = 150.					# distance (pc)



# Printout the top of the CO emission layer at 1 arcsec
cs_ = np.sqrt(sc.k * Tmid0 * (1.0*dist / 10)**qmid / (2.37 * (sc.m_p + sc.m_e)))
om_ = np.sqrt(sc.G * mstar * 1.989e30 / (1.0*dist * sc.au)**3)
zCO = zrmax * (cs_ / om_) / (1.0*dist * sc.au)
print('zCO = {:1.4} (r / 1") ** {:1.3}'.format(zCO, (3 + qmid)/2))



# instantiate RADMC-3D parameters
grid_params = { 'spatial': {'nr': 300, 'nt': 300, 'r_min': 0.1, 'r_max': 300,
                            'rrefine': False },
                'cyl': { 'nr': 2048, 'nt': 2048, 'r_min': 0.1, 'r_max': 1000,
                         'z_min': 0.001, 'z_max': 500 } }

setup_params = { 'incl_dust': 0, 'incl_lines': 1, 'nphot': 10000000,
                 'scattering': 'Isotropic', 'camera_tracemode': 'image',
                 'molecule': 'co', 'transition': 2,
                 'dustspec': 'DIANAstandard' }

cfg_dict = {'radmcname': radmcname,
            'grid_params': grid_params, 'setup_params': setup_params,
            'isoz': False, 'dPdr': False, 'selfgrav': False}



"""
    IMAGING PARAMETERS:

"""
chanstart = '-5.00km/s'
chanwidth = '0.16km/s' 
nchan_out = 125
imsize = 256
cell = '0.025arcsec'
scales = [0, 10, 30, 50]
gain = 0.1
niter = 50000
robust = 0.5
threshold = '10mJy'
uvtaper = ''

# Keplerian mask
zr = 1. * zCO
r_max = 1.2 * r_l / dist
nbeams = 1.5




"""
    ADDITIONAL MISCELLANY:
"""
# process phase center into degrees
RA_pieces = [np.float(RA.split(':')[i]) for i in np.arange(3)]
RAdeg = 15 * np.sum(np.array(RA_pieces) / [1., 60., 3600.])
DEC_pieces = [np.float(DEC.split(':')[i]) for i in np.arange(3)]
DECdeg = np.sum(np.array(DEC_pieces) / [1., 60., 3600.])
