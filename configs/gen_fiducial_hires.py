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

# path to hard-copies of CASA logs
casalogs_dir = outputbase_dir+'CASA_logs/'

# path to CASA/simobserve-format antenna configuration files
antcfg_dir = '/pool/asha0/casa-release-5.7.2-4.el7/data/alma/simmos/'

# path to Rich's keplerian masking script
kepmask_dir = '/home/sandrews/mypy/keplerian_mask/'

# datafile naming base
basename = 'fiducial_hires'

# synthetic "raw" naming base
in_MS = synthraw_dir+basename+'/'+basename

# synthetic "reduced" naming base
dataname = reduced_dir+basename+'/'+basename



"""
    SIMULATED OBSERVATION SETTINGS:

"""
# array observing settings
template = ['hires']				# template names 
config = ['alma.cycle8.5'] 			# antenna location lists 
date = ['2023/03/23'] 				# observation dates (UTC)
HA_0 = ['-0.5h']				# HAs at observing starts
ttotal = ['60min'] 				# total on-source times
tinteg = ['30s']				# integration times per stamp

# spectral settings
dnu_native = [61035.15625] 			# native channel spacings (Hz)
nu_rest = 230.538e9                		# rest frequency (Hz)
V_tune  = [4.0e3] 				# LSRK tunings at centers (m/s)
V_span  = [15.0e3]				# +/- ranges around V_tune (m/s)
nover = 5       				# over-sampling factor (for SRF)

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
r_l   = 250.

T0    = 150.
q     = -0.5
Tmaxb = 20

z0    = 0.2669
psi   = (3 + q) / 2.

sigV0 = np.sqrt(2 * sc.k * T0 / (28 * (sc.m_p + sc.m_e)))	
logtau0 = np.log10(2000.)
ppp   = -1.
Vsys  = 5.0e3
dx    = 0.
dy    = 0.

# fixed inputs
FOV  = [6.375]					# full FOV (arcsec)
Npix = [256]			 		# number of pixels per FOV
						# note: pixsize = FOV/(Npix-1)
dist = 150.					# distance (pc)
cfg_dict = {}					# passable dictionary of kwargs


pars  = np.array([incl, PA, mstar, r_l, z0, psi, T0, q, Tmaxb,
                  sigV0, logtau0, ppp, Vsys, dx, dy])




"""
    IMAGING PARAMETERS:

"""
chanstart = '-5.00km/s'
chanwidth = '0.08km/s' 
nchan_out = 250
imsize = 256
cell = '0.025arcsec'
scales = [0, 10, 30, 50]
gain = 0.1
niter = 50000
robust = 0.5
threshold = '10mJy'
uvtaper = ''

# Keplerian mask
zr = z0 
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
