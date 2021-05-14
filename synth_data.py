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


# file parsing
tpre = 'obs_templates/'+in_.template


### Decide whether to create or use an existing observational template
if np.logical_and(os.path.exists(tpre+'.ms'), 
                  os.path.exists(tpre+'.params')):

    # load the parameters for the existing template
    tp = np.loadtxt(tpre+'.params', dtype=str).tolist()

    # if the parameters are the same, use the existing one; otherwise, save
    # the old one under a previous name and proceed
    ip = [in_.RA, in_.DEC, in_.date, in_.HA, in_.config, in_.ttotal, 
          in_.integ, str(in_.spec_oversample)]
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

# Parse model coordinates
RA_pieces = [np.float(in_.RA.split(':')[i]) for i in np.arange(3)]
RAdeg = 15 * np.sum(np.array(RA_pieces) / [1., 60., 3600.])
DEC_pieces = [np.float(in_.DEC.split(':')[i]) for i in np.arange(3)]
DECdeg = np.sum(np.array(DEC_pieces) / [1., 60., 3600.])


### Generate the mock observations template (if necessary)
if gen_template: 

    # Create a template parameters file for records
    ip = [in_.RA, in_.DEC, in_.date, in_.HA, in_.config, in_.ttotal, 
          in_.integ, in_.spec_oversample]
    f = open(tpre+'.params', 'w')
    [f.write(ip[i]+'\n') for i in range(7)]
    f.write(str(ip[-1]))
    f.close()

    # Set the over-sampled (LSRK) channels for calculating the template
    dv0 = (c_.c * in_.dfreq0 / in_.restfreq) / in_.spec_oversample
    nch = 2 * np.int(in_.vspan / dv0) + 1
    vel = in_.vtune + dv0 * (np.arange(nch) - np.int(nch/2) + 1)

    # Generate a dummy model .FITS cube
    cube_parser(in_.pars, FOV=8, Npix=256, dist=150, r_max=300, 
                restfreq=in_.restfreq, RA=RAdeg, DEC=DECdeg, Vsys=in_.vsys,
                vel=vel, outfile=tpre+'.fits')

    # Generate the (u,v) tracks and spectra on starting integration LSRK frame
    os.system('casa --nologger --nologfile -c CASA_scripts/mock_obs.py')


### Generate synthetic visibility data
# Load template
tmp = np.load('obs_templates/'+in_.template+'.npz')

# Parse template data structure
tmp_data, tmp_uvw, tmp_weights = tmp['data'], tmp['uvw'], tmp['weights']
if len(tmp_data.shape) == 3:
    npol, nchan, nvis = tmp_data.shape[0], tmp_data.shape[1], tmp_data.shape[2]
else:
    npol, nchan, nvis = 1, tmp_data.shape[0], tmp_data.shape[1]

# Spatial frequencies (lambda units)
uu = tmp_uvw[0,:] * np.mean(tmp['freq_LSRK']) / c_.c
vv = tmp_uvw[1,:] * np.mean(tmp['freq_LSRK']) / c_.c

# Get LSRK velocities
v_LSRK = c_.c * (1 - tmp['freq_LSRK'] / in_.restfreq)

# Compute visibilities at each timestamp
clean_vis = np.squeeze(np.empty((npol, nchan, nvis, 2)))
nstamps = v_LSRK.shape[0]
nperstamp = np.int(nvis / nstamps)
for i in range(nstamps):
    # track the steps
    print('timestamp '+str(i+1)+' / '+str(nstamps))

    # create a model cube
    foo = cube_parser(in_.pars, FOV=in_.FOV, Npix=in_.Npix, dist=in_.dist, 
                      r_max=in_.rmax, Vsys=in_.vsys, vel=v_LSRK[i,:], 
                      restfreq=in_.restfreq)#, RA=RAdeg, DEC=DECdeg)

    # sample it's Fourier transform on the template (u,v) spacings
    mvis = vis_sample(imagefile=foo, uu=uu, vv=vv, mu_RA=0, mu_DEC=0, 
                      mod_interp=False).T

    # populate the results in the output array *for this timestamp only*
    ixl, ixh = i * nperstamp, (i + 1) * nperstamp
    if npol == 2:
        clean_vis[0,:,ixl:ixh,0] = mvis.real[:,ixl:ixh]
        clean_vis[1,:,ixl:ixh,0] = mvis.real[:,ixl:ixh]
        clean_vis[0,:,ixl:ixh,1] = mvis.imag[:,ixl:ixh]
        clean_vis[1,:,ixl:ixh,1] = mvis.imag[:,ixl:ixh]
    else:
        clean_vis[:,ixl:ixh,0] = mvis.real[:,ixl:ixh]
        clean_vis[:,ixl:ixh,1] = mvis.imag[:,ixl:ixh]


### Spectral signal processing
# convolution with the spectral response function
chix = np.arange(nchan) / in_.spec_oversample
xch = chix - np.mean(chix)
SRF = 0.5 * np.sinc(xch) + 0.25 * np.sinc(xch - 1) + 0.25 * np.sinc(xch + 1)
clean_vis_SRF = convolve1d(clean_vis, SRF/np.sum(SRF), axis=1, mode='nearest')

# decimate by the over-sampling factor
clean_vis_0 = clean_vis_SRF[:,::in_.spec_oversample,:,:]
freq_LSRK_0 = tmp['freq_LSRK'][:,::in_.spec_oversample]

# Interpolate onto desired output channels (as with CASA/mstransform)
vel_out = in_.chanstart_out + in_.chanwidth_out * np.arange(in_.nchan_out)
freq_out = in_.restfreq * (1 - vel_out / c_.c)
clean_vis_out = np.empty((npol, in_.nchan_out, nvis, 2))
for i in range(nstamps):
    ixl, ixh = i * nperstamp, (i + 1) * nperstamp
    fvis_interp_stamp = interp1d(freq_LSRK_0[i,:], clean_vis_0[:,:,ixl:ixh,:], 
                                 axis=1, fill_value='extrapolate')
    clean_vis_out[:,:,ixl:ixh,:] = fvis_interp_stamp(freq_out)


### Package data (both in .npz and .ms formats)

#os.system('casa --nologger --nologfile -c CASA_scripts/regrid_to_LSRK.py')


