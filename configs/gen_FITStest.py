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

# path to Rich's keplerian masking script
kepmask_dir = '/home/sandrews/mypy/keplerian_mask/'

# datafile naming base
basename = 'FITStest'

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
            'exo12m-hi4']                       # template names 
config = ['alma.cycle8.3', 'alma.cycle8.6',
          'alma.cycle8.6', 'alma.cycle8.6',
          'alma.cycle8.6']                      # antenna location lists 
date = ['2022/04/20', '2022/07/11',
        '2022/07/11', '2022/07/15',
        '2022/07/15']                           # observation dates (UTC)
HA_0 = ['-1.0h', '-2.0h', '0.0h',
         '-2.0h', '0.0h']                       # HAs at observing starts
ttotal = ['60min', '60min', '60min',
          '60min', '60min']                     # total on-source times
tinteg = ['60s', '60s', '60s', '60s', '60s']    # integration times per stamp

# spectral settings
dnu_native = [15258.7890625, 15258.7890625,
              15258.7890625, 15258.7890625,
              15258.7890625]                    # native channel spacings (Hz)
nu_rest = 345.7959899e9                         # rest frequency (Hz)
V_tune  = [5.0e3, 5.0e3, 5.0e3, 5.0e3, 5.0e3]   # LSRK tunings at centers (m/s)
V_span  = [6.2e3, 6.2e3, 6.2e3,
           6.2e3, 6.2e3]                        # +/- ranges around V_tune (m/s)
nover = 1                                       # over-sampling factor (for SRF)

# spatial settings
RA = '16:00:00.00'                              # phase center RA
DEC = '-30:00:00.00'                            # phase center DEC

# noise model settings
RMS = [19.5, 19.5, 19.5, 19.5, 19.5]            # desired RMS (mJy/beam/chan)



"""
    DATA REDUCTION SETTINGS:

"""
tavg = ['', '', '', '', '']			# time-averaging intervals



"""
    INPUT MODEL PARAMETERS:

"""
pars  = 'testcube.fits'

# fixed inputs
FOV  = [6.375, 6.375, 6.375, 6.375, 6.375]	# full FOV (arcsec)
Npix = [512, 512, 512, 512, 512] 		# number of pixels per FOV
						# note: pixsize = FOV/(Npix-1)
dist = 150.					# distance (pc)
cfg_dict = {}					# passable dictionary of kwargs


incl = 40.
PA = 130.
zr = 0.25
r_max = 2.75
Vsys = 0.
mstar = 0.7
dx, dy = 0., 0.
nbeams = 1.5


"""
    IMAGING PARAMETERS:

"""
chanstart = '-5.7km/s'
chanwidth = '0.150km/s' 
nchan_out = 55 #23     #46     #98
imsize = 512
cell = '0.02arcsec'
scales = [0, 10, 30, 50]
gain = 0.1
niter = 50000
robust = 0.5
threshold = '6mJy'
uvtaper = ''




"""
    ADDITIONAL MISCELLANY:
"""
# process phase center into degrees
RA_pieces = [np.float(RA.split(':')[i]) for i in np.arange(3)]
RAdeg = 15 * np.sum(np.array(RA_pieces) / [1., 60., 3600.])
DEC_pieces = [np.float(DEC.split(':')[i]) for i in np.arange(3)]
DECdeg = np.sum(np.array(DEC_pieces) / [1., 60., 3600.])
