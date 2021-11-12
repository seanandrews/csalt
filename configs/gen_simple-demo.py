"""
    This is the configuration file for generating a synthetic dataset from 
    scratch in the csalt architecture.  It is imported as a Python modeule in 
    various subroutines, and will be copied into the directory

        outputbase_dir/reduced_dir/basename/

    for future reference once the data generation is complete.
"""

import numpy as np

"""
    LOCATORS:
    These set the desired locations and naming conventions of the outputs, as 
    well as the locations of necessary ancillary information.
"""
# base path
outputbase_dir = 'storage/'

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


# naming
basename = 'simple-demo'
in_MS = synthraw_dir+basename+'/'+basename
dataname = reduced_dir+basename+'/'+basename


# observation settings
template = ['lmm']
config = ['alma.cycle8.6']	# has a .cfg ending!
ttotal = ['2min']
tinteg = ['30s']
date = ['2022/07/11']
HA_0 = ['-0.25h']
RMS = [5.3]             # desired naturally-weighted RMS in mJy/beam/channel


# spectral settings
dnu_native = [122.0703125 * 1e3]    # native channel spacing (Hz)
nu_rest = 230.538e9	            # rest frequency (Hz)
V_tune  = [4.0e3]	# LSRK velocity tuning for central channel (m/s)
V_span  = [15.0e3]	# +/- velocity width around vtune for simulation (m/s)
nover = 3	# spectral over-sampling factor (for SRF convolution)


# spatial settings
RA   = '16:00:00.00'	# phase center RA
DEC  = '-40:00:00.00'	# phase center DEC
RA_pieces = [np.float(RA.split(':')[i]) for i in np.arange(3)]
RAdeg = 15 * np.sum(np.array(RA_pieces) / [1., 60., 3600.])
DEC_pieces = [np.float(DEC.split(':')[i]) for i in np.arange(3)]
DECdeg = np.sum(np.array(DEC_pieces) / [1., 60., 3600.])


# reduction settings
tavg = ['']
V_bounds = [(5.2 - 10)*1e3, (5.2 + 10)*1e3]
bounds_pad = 3


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
tau0  = 500.
ppp   = -1.
Vsys  = 5.2e3
dx    = 0.
dy    = 0.
pars  = np.array([incl, PA, mstar, r_l, z0, psi, T0, q, Tmaxb, 
                  sigV0, tau0, ppp, Vsys, dx, dy])


# Fixed parameters
FOV  = [6.375]		# full field of view (arcsec)
Npix = [256]  		# number of pixels per FOV
dist = 150.		# distance (pc)

cfg_dict = {}
