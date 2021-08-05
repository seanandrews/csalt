"""
    This is the main control file, used to generate synthetic data or to model
    real (or synthetic) datasets.
"""
import numpy as np


# naming
template_dir = 'obs_templates/'
simobs_dir = '/pool/asha0/casa-release-5.7.2-4.el7/data/alma/simmos/'


# spectral settings
dfreq0    = 122.0703125 * 1e3	# native channel spacing (Hz)
restfreq  = 230.538e9	# rest frequency (Hz)
vtune     = 4.0e3	# LSRK velocity tuning for central channel (m/s)
vspan     = 15.0e3	# +/- velocity width around vtune for simulation (m/s)


# spatial settings
RA   = '16:00:00.00'	# phase center RA
DEC  = '-40:00:00.00'	# phase center DEC
HA   = '0.0h'		# hour angle at start of EB
date = '2022/05/20'	# UTC date for start of EB
RA_pieces = [np.float(RA.split(':')[i]) for i in np.arange(3)]
RAdeg = 15 * np.sum(np.array(RA_pieces) / [1., 60., 3600.])
DEC_pieces = [np.float(DEC.split(':')[i]) for i in np.arange(3)]
DECdeg = np.sum(np.array(DEC_pieces) / [1., 60., 3600.])


# observation settings
integ  = '6s'		# integration time
ttotal = '20min'	# total on-source EB time
config = '6'		# antenna configuration index
