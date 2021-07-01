"""
Generate a synthetic ALMA dataset.
"""
import os, sys, time, importlib
import numpy as np
from scipy.ndimage import convolve1d
from vis_sample import vis_sample
import const as const
from cube_parser import cube_parser


# load synthetic data inputs
inp = importlib.import_module('sconfig_'+sys.argv[-1])


# Set up the workspace if necessary
if not os.path.exists('obs_templates'):
    os.system('mkdir obs_templates')
    os.system('mkdir obs_templates/sims')
elif not os.path.exists('obs_templates/sims'):
    os.system('mkdir obs_templates/sims')

if inp.stordir[-1] != '/': inp.stordir += '/'
if not os.path.exists(inp.stordir):
    os.system('mkdir '+inp.stordir)



"""
Create an observational 'template', if necessary.
"""
### Decide whether to create a template or use an existing one
tpre = 'obs_templates/'+inp.template
if np.logical_and(os.path.exists(tpre+'.ms'), 
                  os.path.exists(tpre+'.params')):

    # load the parameters for the existing template
    tp = np.loadtxt(tpre+'.params', dtype=str).tolist()

    # if the parameters are the same, use the existing one
    ip = [inp.RA, inp.DEC, inp.date, inp.HA, inp.config, inp.ttotal, 
          inp.integ, str(inp.vtune), str(inp.vspan)]
    if tp == ip:
        print('This template already exists...using the files from %s' % \
              (time.ctime(os.path.getctime(tpre+'.ms'))))
        gen_template = False

    # otherwise, to be cautious, get rid of the old one
    else:
        gen_template = True
        print('Removing old template with same name, but different params')
        os.system('rm -rf '+tpre+'.ms '+tpre+'.params')
        os.system('rm -rf obs_templates/sims/'+inp.template+'*')

else:
    gen_template = True


### Generate a mock template (if necessary)
if gen_template: 

    # Create a template parameters file for records
    ip = [inp.RA, inp.DEC, inp.date, inp.HA, inp.config, inp.ttotal, 
          inp.integ, str(inp.vtune), str(inp.vspan)]
    f = open(tpre+'.params', 'w')
    [f.write(ip[i]+'\n') for i in range(len(ip)-1)]
    f.write(ip[-1])
    f.close()

    # Generate the (u,v) tracks and spectra on starting integration LSRK frame
    os.system('rm -rf CASA_logs/mock_obs_'+sys.argv[-1]+'.log')
    os.system('casa --nologger --logfile CASA_logs/mock_obs_'+sys.argv[-1]+\
              '.log -c CASA_scripts/mock_obs.py '+sys.argv[-1])



"""
Generate a synthetic dataset.  Initially this is spectrally over-sampled, and 
computed properly in a set of TOPO channels.  After convolution with the SRF, 
it is down-sampled and interpolated to the set of desired LSRK channels.

The outputs are saved in both a MS and easier-to-access (.NPZ) format.

The noise injection / corruption here make the very simple assumption that each 
visibility shares the same Gaussian parent distribution.  The corrupted signal 
is propagated through the spectral signal processing machinery, so that mimics 
the realistic covariances.  But this is a "hack"...not ideal.
"""
### Setups for model visibilities
# Load template
tmp = np.load('obs_templates/'+inp.template+'.npz')

# Parse template data structure
t_data, t_u, t_v, t_weights = tmp['data'], tmp['u'], tmp['v'], tmp['weights']
t_nu_TOPO, t_nu_LSRK = tmp['nu_TOPO'], tmp['nu_LSRK']
npol, nch, nvis = t_data.shape[0], t_data.shape[1], t_data.shape[2]
nstamps = t_nu_LSRK.shape[0]



# Spatial frequencies (lambda units) [nvis-length vectors]
# Note: this is an approximation that neglects the spectral dependence of the 
# baseline lengths, owing to the small bandwidth of reasonable applications.
# This is easy to change, but it increases the FT computation time by a 
# factor of roughly the number of timestamps (since there's that many more 
# unique (u,v) points to calculate.
uu = t_u * np.mean(t_nu_TOPO) / const.c_
vv = t_v * np.mean(t_nu_TOPO) / const.c_


# Upsample spectral domain (for careful windowing)
nu_TOPO = np.interp(np.arange((nch - 1) * inp.spec_over + 1),
                    np.arange(0, nch * inp.spec_over, inp.spec_over), 
                    t_nu_TOPO)
nch_up = len(nu_TOPO)
nu_LSRK = np.empty((nstamps, nch_up))
for i in range(nstamps):
    nu_LSRK[i,:] = np.interp(np.arange((nch - 1) * inp.spec_over + 1),
                             np.arange(0, nch * inp.spec_over, inp.spec_over),
                             t_nu_LSRK[i,:])

# LSRK velocities [ntimestamp x nch_up array)
v_LSRK = const.c_ * (1 - nu_LSRK / inp.restfreq)


### Configure noise
# Scale the desired output naturally-weighted RMS per channel (inp.RMS) to the
# corresponding standard deviation per visibility (per pol) for the same 
# channel spacing (sigma_out); note that inp.RMS is specified in mJy
sigma_out = 1e-3 * inp.RMS * np.sqrt(npol * nvis)

# Scale that noise per visibility (per pol) to account for the spectral 
# over-sampling and the convolution with the spectral response function
# *** I have no clue where that sqrt(pi) comes from, but it is needed to make
#     the noise distribution properly normalized in the end.  I assume it is 
#     related to the SRF kernel, but haven't tracked it down... ***
sigma_noise = sigma_out * np.sqrt(np.pi * inp.spec_over)

# Random Gaussian noise draws [npol x nchan x nvis x real/imag array]
# Note: we separate the real/imaginary components for faster convolution 
# calculations; these will be put together into complex arrays later
noise = np.squeeze(np.random.normal(0, sigma_noise, (npol, nch_up, nvis, 2)))


### Compute noisy and uncorrupted ("pure") visibilities at each timestamp
### (I think I can make this a bit more efficient, but...)
pure_vis  = np.squeeze(np.empty((npol, nch_up, nvis, 2)))
noisy_vis = np.squeeze(np.empty((npol, nch_up, nvis, 2)))
nperstamp = np.int(nvis / nstamps)
for i in range(nstamps):
    # track the steps
    print('timestamp '+str(i+1)+' / '+str(nstamps))

    # create a model cube
    cube = cube_parser(inp.pars[:inp.npars-3], FOV=inp.FOV, Npix=inp.Npix, 
                       dist=inp.dist, r_max=inp.rmax, Vsys=inp.pars[10], 
                       vel=v_LSRK[i,:], restfreq=inp.restfreq)

    # indices for this timestamp only
    ixl, ixh = i * nperstamp, (i + 1) * nperstamp

    # sample it's Fourier transform on the template (u,v) spacings
    mvis = vis_sample(imagefile=cube, uu=uu[ixl:ixh], vv=vv[ixl:ixh], 
                      mu_RA=inp.pars[11], mu_DEC=inp.pars[12], 
                      mod_interp=False).T

    # populate the results in the output array *for this timestamp only*
    pure_vis[0,:,ixl:ixh,0] = mvis.real
    pure_vis[1,:,ixl:ixh,0] = mvis.real
    pure_vis[0,:,ixl:ixh,1] = mvis.imag
    pure_vis[1,:,ixl:ixh,1] = mvis.imag
    noisy_vis[0,:,ixl:ixh,0] = mvis.real + noise[0,:,ixl:ixh,0]
    noisy_vis[1,:,ixl:ixh,0] = mvis.real + noise[1,:,ixl:ixh,0]
    noisy_vis[0,:,ixl:ixh,1] = mvis.imag + noise[0,:,ixl:ixh,1]
    noisy_vis[1,:,ixl:ixh,1] = mvis.imag + noise[1,:,ixl:ixh,1]



### Spectral signal processing
# convolution with the spectral response function
chix = np.arange(nch_up) / inp.spec_over
xch = chix - np.mean(chix)
SRF = 0.5 * np.sinc(xch) + 0.25 * np.sinc(xch - 1) + 0.25 * np.sinc(xch + 1)
pure_vis = convolve1d(pure_vis, SRF/np.sum(SRF), axis=1, mode='nearest')
noisy_vis = convolve1d(noisy_vis, SRF/np.sum(SRF), axis=1, mode='nearest')

# decimate by the over-sampling factor
pure_vis = pure_vis[:,::inp.spec_over,:,:]
noisy_vis = noisy_vis[:,::inp.spec_over,:,:]
nu_LSRK = nu_LSRK[:,::inp.spec_over]
vel_LSRK = v_LSRK[:,::inp.spec_over]


# Assign the weights
weights_out = np.sqrt(1 / sigma_out) * np.ones((npol, nvis))


### Package data (both in .npz and .ms formats)
if not os.path.exists(inp.stordir+inp.basename):
    os.system('mkdir '+inp.stordir+inp.basename)

os.system('cp sconfig_'+sys.argv[-1]+'.py '+inp.stordir+inp.basename)
os.system('rm -rf '+inp.stordir+inp.basename+'/'+inp.basename+'-'+\
          inp.template+'.npz')
np.savez(inp.stordir+inp.basename+'/'+inp.basename+'-'+inp.template+'.npz', 
         u=uu, v=vv, weights=weights_out, 
         freq_LSRK_grid=nu_LSRK, vel_LSRK_grid=vel_LSRK,
         vis_pure=pure_vis[:,:,:,0] + 1j*pure_vis[:,:,:,1],
         vis=noisy_vis[:,:,:,0] + 1j*noisy_vis[:,:,:,1])

os.system('rm -rf CASA_logs/pack_data_'+sys.argv[-1]+'.log')
os.system('casa --nologger --logfile CASA_logs/pack_data_'+sys.argv[-1]+\
          '.log -c CASA_scripts/pack_data.py '+sys.argv[-1])
