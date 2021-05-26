"""
Generate a synthetic ALMA dataset.
"""

import os, sys, time
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d
from vis_sample import vis_sample
import const as c_
import mconfig as in_
from cube_parser import cube_parser
import matplotlib.pyplot as plt



"""
First decide if a new 'template' MS needs to be created using CASA/simobserve.
    > If the same template already exists, don't make a new one.  
    > If a template of the same name but different parameters exists, either 
      copy the old one into a new name (with the date/time string appended), or 
      just delete it (if overwrite_template = True) and make a new template.

A template is created with a 'dummy' cube as the model.  The goal is to get a 
proper MS structure, with the right (u, v, oversampled channel) dimensions and 
formatting.  An actual model will use these with an appropriate cube.
"""
### Decide whether to create or use an existing observational template
tpre = 'obs_templates/'+in_.template
if np.logical_and(os.path.exists(tpre+'.ms'), 
                  os.path.exists(tpre+'.params')):
    # load the parameters for the existing template
    tp = np.loadtxt(tpre+'.params', dtype=str).tolist()

    # if the parameters are the same, use the existing one; otherwise, save
    # the old one under a previous name and proceed
    ip = [in_.RA, in_.DEC, in_.date, in_.HA, in_.config, in_.ttotal, 
          in_.integ, str(in_.vtune), str(in_.vspan), str(in_.spec_oversample)]
    if tp == ip:
        print('This template already exists...using the files from %s' % \
              (time.ctime(os.path.getctime(tpre+'.ms'))))
        gen_template = False
    else:
        gen_template = True
        if in_.overwrite_template:
            print('Removing old template with same name, but different params')
            os.system('rm -rf '+tpre+'.ms '+tpre+'.params')
            os.system('rm -rf obs_templates/sims/'+in_.template+'*')
        else:
            print('Renaming old template with same name, but different params')
            old_t = time.ctime(os.path.getctime(tpre+'.ms'))
            os.system('mv '+tpre+'.ms '+tpre+'_'+old_t+'.ms') 
            os.system('mv '+tpre+'.params '+tpre+'_'+old_t+'.params')
            os.system('mv obs_templates/sims/'+in_.template+' '+ \
                      'obs_templates/sims/'+in_.template+'_'+old_t)
else:
    gen_template = True


# Parse model coordinates into decimal degrees
RA_pieces = [np.float(in_.RA.split(':')[i]) for i in np.arange(3)]
RAdeg = 15 * np.sum(np.array(RA_pieces) / [1., 60., 3600.])
DEC_pieces = [np.float(in_.DEC.split(':')[i]) for i in np.arange(3)]
DECdeg = np.sum(np.array(DEC_pieces) / [1., 60., 3600.])


### Generate the mock observations template (if necessary)
if gen_template: 
    # Create a template parameters file for records
    ip = [in_.RA, in_.DEC, in_.date, in_.HA, in_.config, in_.ttotal, 
          in_.integ, str(in_.vtune), str(in_.vspan), str(in_.spec_oversample)]
    f = open(tpre+'.params', 'w')
    [f.write(ip[i]+'\n') for i in range(len(ip)-1)]
    f.write(ip[-1])
    f.close()

    # Set the over-sampled (LSRK) channels for calculating the template
    dv0 = (c_.c * in_.dfreq0 / in_.restfreq) / in_.spec_oversample
    nch = 2 * np.int(in_.vspan / dv0) + 1
    vel = in_.vtune + dv0 * (np.arange(nch) - np.int(nch/2) + 1)

    # Generate a dummy model .FITS cube
    cube_parser(in_.pars[:in_.npars-3], FOV=in_.FOV, Npix=in_.Npix, 
                dist=in_.dist, r_max=in_.rmax, restfreq=in_.restfreq, 
                RA=RAdeg, DEC=DECdeg, Vsys=in_.pars[10], vel=vel, 
                outfile=tpre+'.fits')

    # Generate the (u,v) tracks and spectra on starting integration LSRK frame
    os.system('casa --nologger --nologfile -c CASA_scripts/mock_obs.py')

    # Get rid of the .FITS cube (equivalent is stored in CASA .image format in
    # the corresponding obs_templates/sims/ directory ('*.skymodel')
    os.system('rm -rf '+tpre+'.fits')



"""
Generate a synthetic dataset.  Initially this is spectrally over-sampled, and 
computed properly in a set of TOPO channels.  After convolution with the SRF, 
it is down-sampled and interpolated to the set of desired LSRK channels.

The outputs are saved in both a MS and easier-to-access (.NPZ) format.

The noise injection / corruption here make the very simple assumption that each 
visibility shares the same Gaussian parent distribution.  The corrupted signal 
is propagated through the spectral signal processing machinery, so that mimics 
the realistic covariances.  But this is perhaps too simple...
"""
### Setups for model visibilities
# Load template
tmp = np.load('obs_templates/'+in_.template+'.npz')

# Parse template data structure
tmp_data, tmp_uvw, tmp_weights = tmp['data'], tmp['uvw'], tmp['weights']
if len(tmp_data.shape) == 3:
    npol, nchan, nvis = tmp_data.shape[0], tmp_data.shape[1], tmp_data.shape[2]
else:
    npol, nchan, nvis = 1, tmp_data.shape[0], tmp_data.shape[1]

# Spatial frequencies (lambda units) [nvis-length vectors]
# Note: this is an approximation that neglects the spectral dependence of the 
# baseline lengths, owing to the small bandwidth of reasonable applications.
uu = tmp_uvw[0,:] * np.mean(tmp['freq_LSRK']) / c_.c
vv = tmp_uvw[1,:] * np.mean(tmp['freq_LSRK']) / c_.c

# LSRK velocities [ntimestamp x nchan array)
v_LSRK = c_.c * (1 - tmp['freq_LSRK'] / in_.restfreq)


### Configure noise
# Scale the desired output naturally-weighted RMS per channel (in_.RMS) to the
# corresponding standard deviation per visibility (per pol) for the same 
# channel spacing (sigma_out); note that in_.RMS is specified in mJy
sigma_out = 1e-3 * in_.RMS * np.sqrt(npol * nvis)

# Scale that noise per visibility (per pol) to account for the spectral 
# over-sampling and the convolution with the spectral response function
sigma_noise = sigma_out * np.sqrt(np.pi * in_.spec_oversample)

# Random Gaussian noise draws [npol x nchan x nvis x real/imag array]
# Note: we separate the real/imaginary components for faster convolution 
# calculations; these will be put together into complex arrays later
noise = np.squeeze(np.random.normal(0, sigma_noise, (npol, nchan, nvis, 2)))


### Compute noisy and uncorrupted ("clean") visibilities at each timestamp
clean_vis = np.squeeze(np.empty((npol, nchan, nvis, 2)))
noisy_vis = np.squeeze(np.empty((npol, nchan, nvis, 2)))
nstamps = v_LSRK.shape[0]
nperstamp = np.int(nvis / nstamps)
for i in range(nstamps):
    # track the steps
    print('timestamp '+str(i+1)+' / '+str(nstamps))

    # create a model cube
    cube = cube_parser(in_.pars[:in_.npars-3], FOV=in_.FOV, Npix=in_.Npix, 
                       dist=in_.dist, r_max=in_.rmax, Vsys=in_.pars[10], 
                       vel=v_LSRK[i,:], restfreq=in_.restfreq)

    # sample it's Fourier transform on the template (u,v) spacings
    mvis = vis_sample(imagefile=cube, uu=uu, vv=vv, mu_RA=in_.pars[11], 
                      mu_DEC=in_.pars[12], mod_interp=False).T

    # populate the results in the output array *for this timestamp only*
    ixl, ixh = i * nperstamp, (i + 1) * nperstamp
    if npol == 2:
        clean_vis[0,:,ixl:ixh,0] = mvis.real[:,ixl:ixh]
        clean_vis[1,:,ixl:ixh,0] = mvis.real[:,ixl:ixh]
        clean_vis[0,:,ixl:ixh,1] = mvis.imag[:,ixl:ixh]
        clean_vis[1,:,ixl:ixh,1] = mvis.imag[:,ixl:ixh]
        noisy_vis[0,:,ixl:ixh,0] = mvis.real[:,ixl:ixh] + noise[0,:,ixl:ixh,0]
        noisy_vis[1,:,ixl:ixh,0] = mvis.real[:,ixl:ixh] + noise[1,:,ixl:ixh,0]
        noisy_vis[0,:,ixl:ixh,1] = mvis.imag[:,ixl:ixh] + noise[0,:,ixl:ixh,1]
        noisy_vis[1,:,ixl:ixh,1] = mvis.imag[:,ixl:ixh] + noise[1,:,ixl:ixh,1]
    else:
        clean_vis[:,ixl:ixh,0] = mvis.real[:,ixl:ixh]
        clean_vis[:,ixl:ixh,1] = mvis.imag[:,ixl:ixh]
        noisy_vis[:,ixl:ixh,0] = mvis.real[:,ixl:ixh] + noise[:,ixl:ixh,0]
        noisy_vis[:,ixl:ixh,1] = mvis.imag[:,ixl:ixh] + noise[:,ixl:ixh,1]


### Spectral signal processing
# convolution with the spectral response function
chix = np.arange(nchan) / in_.spec_oversample
xch = chix - np.mean(chix)
SRF = 0.5 * np.sinc(xch) + 0.25 * np.sinc(xch - 1) + 0.25 * np.sinc(xch + 1)
clean_vis_SRF = convolve1d(clean_vis, SRF/np.sum(SRF), axis=1, mode='nearest')
noisy_vis_SRF = convolve1d(noisy_vis, SRF/np.sum(SRF), axis=1, mode='nearest')

# decimate by the over-sampling factor
clean_vis_0 = clean_vis_SRF[:,::in_.spec_oversample,:,:]
noisy_vis_0 = noisy_vis_SRF[:,::in_.spec_oversample,:,:]
freq_LSRK_0 = tmp['freq_LSRK'][:,::in_.spec_oversample]

# Interpolate onto desired output channels (this is what happens when you use 
# CASA/mstransform to go from TOPO --> the specified LSRK channels when the 
# output LSRK channel spacing is <2x the TOPO channel spacing; that is what we 
# want to do with real data, so we have a good model for the covariance matrix)
#
# set output velocity and frequency grids, maintaining native spacing
dvel0 = c_.c * (in_.dfreq0 / in_.restfreq)
nch_out = 2 * np.int((in_.pars[10] - in_.chanstart_out) / dvel0)
vel_out = in_.chanstart_out + dvel0 * np.arange(nch_out)
freq_out = in_.restfreq * (1 - vel_out / c_.c)

# populate the interpolated visibility grids
clean_vis_out = np.empty((npol, nch_out, nvis, 2))
noisy_vis_out = np.empty((npol, nch_out, nvis, 2))
for i in range(nstamps):
    ixl, ixh = i * nperstamp, (i + 1) * nperstamp
    cvis_interp_stamp = interp1d(freq_LSRK_0[i,:], clean_vis_0[:,:,ixl:ixh,:], 
                                 axis=1, fill_value='extrapolate')
    clean_vis_out[:,:,ixl:ixh,:] = cvis_interp_stamp(freq_out)
    nvis_interp_stamp = interp1d(freq_LSRK_0[i,:], noisy_vis_0[:,:,ixl:ixh,:],
                                 axis=1, fill_value='extrapolate')
    noisy_vis_out[:,:,ixl:ixh,:] = nvis_interp_stamp(freq_out)

# Assign the weights
weights_out = np.sqrt(1 / sigma_out) * np.ones((npol, nvis))


### Package data (both in .npz and .ms formats)
os.system('cp mconfig.py data/mconfig_'+in_.basename+'-'+in_.template+'.py')
os.system('rm -rf data/'+in_.basename+'-'+in_.template+'.npz')
np.savez('data/'+in_.basename+'-'+in_.template+'.npz', 
         u=uu, v=vv, freq = freq_out, vel=vel_out, weights=weights_out,
         vis=clean_vis_out[:,:,:,0] + 1j*clean_vis_out[:,:,:,1],
         vis_noisy= noisy_vis_out[:,:,:,0] + 1j*noisy_vis_out[:,:,:,1])

os.system('casa --nologger --nologfile -c CASA_scripts/pack_data.py')


### Image the simulations to check that they make sense
if in_.do_img is not None:
    os.system('casa --nologger --nologfile -c CASA_scripts/image_cube.py')
