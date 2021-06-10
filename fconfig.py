"""
"""
import numpy as np


### data configuration and storage
# file assignments
orig_dir = '/data/sandrews/LP/sa_work/Sz129/'
orig_MS  = orig_dir+'Sz129_combined_CO_selfcal.ms.contsub'
basename = 'Sz129'
extname  = ''
outdir   = 'data/'+basename+'/'
dataname = outdir+basename+extname
preserve_tmp = False

# reduction parameters
nu_rest = 230.538e9			# spectral line rest frequency (Hz)
V_sys = 4.1e3				# estimate of systemic velocity (m/s)
dVb   = 8e3				# desired +/- range around V_sys (m/s)
bounds_V = [V_sys-dVb, V_sys+dVb]	# LSRK velocity bounds to extract (m/s)
chpad = 5				# pad channels on each end of bounds_V
tavg = '30s'				# time-averaging interval for orig_MS
					# (no averaging == '')
