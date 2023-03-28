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
basename = 'sg_modelb'

# synthetic "raw" naming base
in_MS = synthraw_dir+basename+'/'+basename

# synthetic "reduced" naming base
dataname = reduced_dir+basename+'/'+basename

# radmc output naming base
radmcname = radmc_dir+basename+'/'



"""
    SIMULATED OBSERVATION SETTINGS:

"""
# array observing settings
template = ['sg-c4', 'sg-c7a', 'sg-c7b', 'sg-c7c', 'sg-c7d']  # template names 
config = ['alma.cycle8.4', 'alma.cycle8.7',
          'alma.cycle8.7', 'alma.cycle8.7',
          'alma.cycle8.7']                      # antenna location lists 
date = ['2023/03/20', '2023/05/20',
        '2023/05/20', '2023/05/25',
        '2023/05/25']                           # observation dates (UTC)
HA_0 = ['-0.5h', '-1.0h', '0.0h',
        '-1.0h', '0.0h']			# HAs at observing starts
ttotal = ['60min', '60min', '60min',
          '60min', '60min'] 			# total on-source times
tinteg = ['30s', '30s', '30s', '30s', '30s']	# integration times per stamp

# spectral settings
dnu_native = [30517.578125, 30517.578125,
              30517.578125, 30517.578125,
              30517.578125]		 	# native channel spacings (Hz)
nu_rest = 230.538e9                		# rest frequency (Hz)
V_tune  = [0.0e3, 0.0e3, 0.0e3, 0.0e3, 0.0e3]	# LSRK tunings at centers (m/s)
V_span  = [6.5e3, 6.5e3, 6.5e3, 6.5e3, 6.5e3]	# +/- ranges around V_tune (m/s)
nover = 1       				# over-sampling factor (for SRF)

# spatial settings
RA = '16:00:00.00'    				# phase center RA
DEC = '-30:00:00.00'   				# phase center DEC

# noise model settings
RMS = [5.4, 5.4, 5.4, 5.4, 5.4]			# desired RMS (mJy/beam/chan)



"""
    DATA REDUCTION SETTINGS:

"""
tavg = ['', '', '', '', '']			# time-averaging intervals
V_bounds = [0.0e3-6e3, 0.0e3+6e3]



"""
    INPUT MODEL PARAMETERS:

"""
incl  = 30.
PA    = 130.
mstar = 1.0

Tmid0 = 50.	#.75.	
Tatm0 = 200. 
qmid  = -0.5
qatm  = -0.5
zq    = 0.3	#0.25
deltaT = 2.0

Sig0  = 99.8
p1    = -1.0
p2    = 2.0
r_l   = 160.

xmol  = 1e-5
depl  = 1e-3
Tfrz  = 20.
Ncrit = 1.8 * 1.6e21	#1.9
rmax_abund = 5000.

xi    = 0.0

Vsys  = 0.0e3
dx    = 0.
dy    = 0.
pars  = np.array([incl, PA, mstar, r_l, Tmid0, Tatm0, qmid, qatm, zq, deltaT,
                  Sig0, p1, p2, xmol, depl, Tfrz, Ncrit, rmax_abund,
                  xi, Vsys, dx, dy])

# fixed inputs
FOV  = [10.23, 10.23, 10.23, 10.23, 10.23]
Npix = [1024, 1024, 1024, 1024, 1024]           # number of pixels per FOV
                                                # note: pixsize = FOV/(Npix-1)
dist = 150.					# distance (pc)


# instantiate RADMC-3D parameters
grid_params = { 'spatial': {'nr': 360, 'nt': 240, 'r_min': 1.0, 'r_max': 500, 
                            'rrefine': False, 'rref_i': [215], 'rref_o': [235], 
                            'nrref': [50], 'rref_scl': ['lin']},
                'cyl': { 'nr': 2048, 'nt': 2048, 'r_min': 0.1, 'r_max': 1000,
                         'z_min': 0.001, 'z_max': 500 } }

setup_params = { 'incl_dust': 0, 'incl_lines': 1, 'nphot': 10000000,
                 'scattering': 'Isotropic', 'camera_tracemode': 'image',
                 'molecule': 'co', 'transition': 2,
                 'dustspec': 'DIANAstandard' }

cfg_dict = {'radmcname': radmcname, 'grid_params': grid_params, 
            'setup_params': setup_params, 'isoz': False, 'dPdr': True, 
            'selfgrav': True, 'dens_selfgrav': True}



"""
    IMAGING PARAMETERS:

"""
chanstart = '-5.6km/s'
chanwidth = '0.08km/s'
nchan_out = 140
imsize = 512
cell = '0.02arcsec'
scales = [0, 10, 30, 50]
gain = 0.1
niter = 100000
robust = 0.5
threshold = '10mJy'
uvtaper = ''

# Keplerian mask
zr = 0.3
r_max = 1.3 * 300 / dist
nbeams = 2



"""
    ADDITIONAL MISCELLANY:
"""
# process phase center into degrees
RA_pieces = [np.float(RA.split(':')[i]) for i in np.arange(3)]
RAdeg = 15 * np.sum(np.array(RA_pieces) / [1., 60., 3600.])
DEC_pieces = [np.float(DEC.split(':')[i]) for i in np.arange(3)]
DECdeg = np.sum(np.array(DEC_pieces) / [1., 60., 3600.])
