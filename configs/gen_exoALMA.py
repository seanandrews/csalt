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

# datafile naming base
basename = 'exoALMA'

# synthetic "raw" naming base
in_MS = synthraw_dir+basename+'/'+basename

# synthetic "reduced" naming base
dataname = reduced_dir+basename+'/'+basename



"""
    SIMULATED OBSERVATION SETTINGS:

"""
# array observing settings
template = ['exo12m-lo', 'exo12m-hi1', 
            'exo12m-hi2', 'exo12m-hi3', 
            'exo12m-hi4']			# template names 
config = ['alma.cycle8.3', 'alma.cycle8.6',
          'alma.cycle8.6', 'alma.cycle8.6',
          'alma.cycle8.6']			# antenna location lists 
date = ['2022/04/20', '2022/07/11',
        '2022/07/11', '2022/07/15', 
	'2022/07/15']				# observation dates (UTC)
HA_0 = ['-1.0h', '-2.0h', '0.0h',
        '-2.0h', '0.0h']			# HAs at observing starts
ttotal = ['60min', '60min', '60min',
          '60min', '60min']			# total on-source times
tinteg = ['30s', '30s', '30s', '30s', '30s']	# integration times per stamp

# spectral settings
dnu_native = [30517.578125, 30517.578125,
              30517.578125, 30517.578125,
              30517.578125]			# native channel spacings (Hz)
nu_rest = 345.7959899e9                		# rest frequency (Hz)
V_tune  = [0.0e3, 0.0e3, 0.0e3, 0.0e3, 0.0e3] 	# LSRK tunings at centers (m/s)
V_span  = [25.0e3, 25.0e3, 25.0e3,
           25.0e3, 25.0e3]      		# +/- ranges around V_tune (m/s)
nover = 1       				# over-sampling factor (for SRF)

# spatial settings
RA = '16:00:00.00'    				# phase center RA
DEC = '-30:00:00.00'   				# phase center DEC

# noise model settings
RMS = [5.8, 5.8, 5.8, 5.8, 5.8]			# desired RMS (mJy/beam/chan)



"""
    DATA REDUCTION SETTINGS:

"""
tavg = ['', '', '', '', '']			# time-averaging intervals
V_bounds = [0e3-15e3, 0e3+15e3]			# excised V_LSRK range (m/s)



"""
    INPUT MODEL PARAMETERS:

"""
# parametric_model inputs
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
ltau0  = np.log10(500.)
ppp   = -1.
Vsys  = 0e3
dx    = 0.
dy    = 0.
pars  = np.array([incl, PA, mstar, r_l, z0, psi, T0, q, Tmaxb, 
                  sigV0, ltau0, ppp, Vsys, dx, dy])

# fixed inputs
FOV  = [6.375, 6.375, 6.375, 6.375, 6.375]	# full FOV (arcsec)
Npix = [512, 512, 512, 512, 512] 		# number of pixels per FOV
						# note: pixsize = FOV/(Npix-1)
dist = 150.					# distance (pc)
cfg_dict = {}					# passable dictionary of kwargs




"""
    ADDITIONAL MISCELLANY:
"""
# process phase center into degrees
RA_pieces = [np.float(RA.split(':')[i]) for i in np.arange(3)]
RAdeg = 15 * np.sum(np.array(RA_pieces) / [1., 60., 3600.])
DEC_pieces = [np.float(DEC.split(':')[i]) for i in np.arange(3)]
DECdeg = np.sum(np.array(DEC_pieces) / [1., 60., 3600.])
