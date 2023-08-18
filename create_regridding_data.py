import os
import sys
import numpy as np
from csalt.model import *
from csalt.helpers import *
import matplotlib as mpl
mpl.rcParams['backend'] = 'TkAgg'


# User inputs
name = 'ALMA_BLC_244kHz'
dnu_ = 244e3
SRF_ = 'ALMA'

storage_dir = 'regridding_storage/'
configs_dir = '/pool/asha0/casa-release-5.7.2-4.el7/data/alma/simmos/'

##########

# Instantiate a csalt model
cm = model('CSALT0')

# Create an empty MS from scratch
cdir = configs_dir
#cm.template_MS(storage_dir+'template_'+name+'.ms', 
#               config=[configs_dir+'alma.cycle8.4.cfg', 
#                       configs_dir+'alma.cycle8.7.cfg',
#                       configs_dir+'alma.cycle8.7.cfg',
#                       configs_dir+'alma.cycle8.7.cfg',
#                       configs_dir+'alma.cycle8.7.cfg'],
#               t_total='60min', t_integ='30s', observatory='ALMA',
#               date=['2025/03/01', '2025/05/20', '2025/05/20', 
#                     '2025/05/27', '2025/05/27'], 
#               HA_0=['-0.5h', '-1.0h', '0.0h', '-1.0h', '0.0h'],
#               restfreq=230.538e9, dnu_native=dnu_, V_span=7.5e3,
#               RA='16:00:00.00', DEC='-30:00:00.00')

cm.template_MS(storage_dir+'template_'+name+'.ms',
               config=[configs_dir+'alma.cycle8.4.cfg',
                       configs_dir+'alma.cycle8.7.cfg'],
               t_total='60min', t_integ='30s', observatory='ALMA',
               date=['2025/03/01', '2025/05/27'],
               HA_0=['-0.5h', '-1.0h'],
               restfreq=230.538e9, dnu_native=dnu_, V_span=7.5e3,
               RA='16:00:00.00', DEC='-30:00:00.00')


# Get the data dictionary from the empty MS
ddict = read_MS(storage_dir+'template_'+name+'.ms')

# Set the CSALT model parameters
pars = np.array([ 
                   45, 	# incl (deg)
                   60, 	# PA (deg), E of N to redshifted major axis
                  1.0, 	# Mstar (Msun)
                  300, 	# R_out (au)
                  0.3, 	# emission height z_0 (") at r = 1"
                  1.0, 	# phi for z(r) = z_0 * (r / 1)**phi
                  150, 	# Tb_0 at r = 10 au (K)
                 -0.5, 	# q for Tb(r) = Tb_0 * (r / 10 au)**q
                   20, 	# maximum Tb for back surface of disk (~ Tfreezeout)
                  297,	# linewidth at r = 10 au (m/s)
                  3.0, 	# log(tau_0) at 10 au
                   -1, 	# p for tau(r) = tau_0 * (r / 10 au)**p
                  0e3, 	# systemic velocity (m/s)
                    0, 	# RA offset (")
                    0	# DEC offset (")
                     ])

# Generate the SAMPLED and NOISY datacubes
if SRF_ == 'ALMA-WSU':
    x_scale = 1.2
else:
    x_scale = 1.0
noise = 7.3 * np.sqrt(30.5e3 / dnu_) / x_scale
fixed_kw = {'FOV': 5.11, 'Npix': 512, 'dist': 150,
            'Nup': 5, 'doppcorr': 'exact', 'SRF': SRF_, 'noise_inject': noise}
sampl_mdict, noisy_mdict = cm.modeldict(ddict, pars, kwargs=fixed_kw)
write_MS(sampl_mdict, outfile=storage_dir+name+'_SAMPLED.ms')
write_MS(noisy_mdict, outfile=storage_dir+name+'_NOISY.ms')
