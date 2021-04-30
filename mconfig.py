"""
    This is the main control file, used to generate synthetic data or to model
    real (or synthetic) datasets.
"""
import numpy as np


# file naming
basename = 'test'
template = 'uvtest'


# controls
gen_data = False
fit_data = False
overwrite_template = True


# Model parameters
# ----------------
npars = 13
pars  = np.zeros(npars)
pars[0]  = 30.		# inclination angle (degrees)
pars[1]  = 140.		# position angle (degrees)
pars[2]  = 1.0		# stellar mass (Msun)
pars[3]  = 200.		# "line radius" (AU)
pars[4]  = 2.		# emission surface height at r_0 (AU)
pars[5]  = 1.		# radial power-law index of emission surface height
pars[6]  = np.inf	# outer disk flaring angle
pars[7]  = 150.		# temperature at r_0 (K)
pars[8]  = 0.5		# radial power-law index of temperature surface
pars[9]  = np.inf	# outer disk temperature gradient
pars[10] = 100.		# optical depth at r_0
pars[11] = 0		# optical depth gradient
pars[12] = np.inf	# outer disk optical depth gradient

 


# Simulated observations parameters
# ---------------------------------
# spectral settings
dfreq0   = 61.035e3    # in Hz
restfreq = 230.538e9          # in Hz
vsys     = 4.0e3             # in m/s
vspan    = 10e3             # in m/s
sosampl  = 3     

# spatial settings
RA   = '04:30:00.00'
DEC  = '25:00:00.00'
HA   = '0.0h'
date = '2021/12/01'

# observation settings
config = '5'
ttotal = '20min'
integ  = '30s'





