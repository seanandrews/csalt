"""
"""
import numpy as np

### Naming and input conventions
# Template(s)
template_dir = 'obs_templates'
template = ['nmm']			# always a list, even for 1 template

# Output storage
storage_dir = 'synth_storage'
basename = 'simp3'


# Synthetic data settings
restfreq  = 230.538e9	# rest frequency (Hz)
spec_over = 5		# over-sampling factor for spectral signal processing
RMS = [7.4389]		# desired naturally-weighted RMS in mJy/beam/channel


# Model parameters
# ----------------
# structure parameters (free)
pars  = np.array([40, 130, 0.7, 200, 2.3, 1, 205, 0.5, 20, 348, 5.2e3, 0, 0])
# [incl, PA, Mstar, r_l, z_0, zpsi, T0, q, Tmax_back, dV0, Vsys, dx, dy]
npars = len(pars)

# fixed setup parameters
FOV  = 10.24		# full field of view (arcsec)
Npix = 1024 		# number of pixels per FOV
dist = 150.		# distance (pc)
rmax = dist * 0.5 * FOV	# maximum radius of emission (au)
fixed = restfreq, FOV, Npix, dist, rmax
