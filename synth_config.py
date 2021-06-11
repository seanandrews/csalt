"""
    This is the main control file, used to generate synthetic data or to model
    real (or synthetic) datasets.
"""
import numpy as np


# model name
mdlname = 'simp3'
extname = ''
basename = mdlname+extname

# template setup
duration = 'short'		
spatial_res = 'medr'		
spectral_res = 'medv'		
template = duration+'_'+spatial_res+'_'+spectral_res


# observational configuration
t_key = np.array(['snap', 'short', 'long', 'deep'])
r_key = np.array(['highr', 'medr', 'lowr'])
v_key = np.array(['highv', 'medv', 'lowv'])

dt_opt = np.array(['1min', '15min', '60min', '180min'])
dr_opt = np.array(['4', '6', '8'])
dv_opt = np.array([61.03515625, 122.0703125, 244.140625])
rms_opt = np.array([[40.7445, 28.8107, 20.3723], 
                    [10.5202,  7.4389,  5.2600],
                    [ 5.2600,  3.7194,  2.6300],
                    [ 3.0369,  2.1474,  1.5185]])


# controls
overwrite_template = True

# auxiliary file locations
simobs_dir = '/pool/asha0/casa-release-5.7.2-4.el7/data/alma/simmos/'


# Model free parameters
# ---------------------
par_opt = np.array(['simp1', 'simp2', 'simp3', 'simp4', 'simp5'])
pgrid = np.array([[40, 130, 0.1,  40, 3.0, 1, 130, 0.5, 20, 277, 5.2e3, 0, 0],
                  [40, 130, 0.4, 115, 2.7, 1, 160, 0.5, 20, 307, 5.2e3, 0, 0],
                  [40, 130, 0.7, 200, 2.3, 1, 205, 0.5, 20, 348, 5.2e3, 0, 0], 
                  [40, 130, 1.0, 285, 2.0, 1, 240, 0.5, 20, 377, 5.2e3, 0, 0],
                  [40, 130, 2.0, 540, 1.5, 1, 330, 0.5, 20, 442, 5.2e3, 0, 0]])
pars = np.squeeze(pgrid[par_opt == mdlname, :])
npars = len(pars)
#pars[0]  = inclination angle (degrees)
#pars[1]  = position angle (degrees)
#pars[2]  = stellar mass (Msun)
#pars[3]  = "line radius" (AU)
#pars[4]  = emission surface height at r_0 (AU)
#pars[5]  = radial power-law index of emission surface height
#pars[6]  = temperature at r_0 (K)
#pars[7]  = radial power-law index of temperature surface
#pars[8]  = maximum brightness temperature of back side
#pars[9]  = line-width at r_0 (m/s)
#pars[10] = systemic velocity (m/s)
#pars[11] =  RA offset (arcsec)
#pars[12] = DEC offset (arcsec)


# Model fixed parameters
# ----------------------
# properties for sky-projected cube
FOV  = 10.		# full field of view (arcsec)
Npix = 512		# number of pixels per FOV
dist = 150.		# distance (pc)
rmax = 700.		# maximum radius of emission (au)

# desired output LSRK velocity channels
chanstart_out = -4.8e3	# m/s
#chanwidth_out = 320.	# m/s
#nchan_out = 65		# 

# noise properties (RMS per channel in naturally weighted images (mJy))
RMS = rms_opt[t_key==duration, v_key==spectral_res][0]	
 


# Template observational parameters
# ---------------------------------
# spectral settings
dfreq0   = dv_opt[v_key==spectral_res][0]*1e3 	# native channel spacing (Hz)
restfreq = 230.538e9  				# rest frequency (Hz)
vtune    = 4.0e3	# LSRK velocity tuning for central channel (m/s)
vspan    = 12.5e3     	# +/- velocity width around vtune for simulation (m/s)
spec_over = 5   	# over-sampling factor for spectral signal processing

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
config = dr_opt[r_key==spatial_res][0]		# antenna config ('5' = C43-5)
ttotal = dt_opt[t_key==duration][0]		# total EB time
integ  = '30s'					# integration time

# imaging
do_img = ['', '_noisy']
imsize = 256
cell = '0.03arcsec'
robust = 0.5
thresh = str(RMS)+'mJy'
cscales = [0, 5, 10, 15]
