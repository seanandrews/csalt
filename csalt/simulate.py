import os, sys, importlib
import numpy as np
import warnings
import copy
import scipy.constants as sc
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d
from vis_sample import vis_sample
from csalt.data2 import *

class simulate:

    def __init__(self, prescription, path=None, quiet=True):
        if quiet:
            warnings.filterwarnings("ignore")
        self.prescription = prescription
        self.path = path


    """ Generate a cube """
    def cube(self, velax, pars, 
             restfreq=230.538e9, FOV=5.0, npix=256, dist=150):

        # Parse inputs
        if isinstance(velax, list): velax = np.array(velax)
        fixed = restfreq, FOV, npix, dist, {}

        # Locate the prescription file
        if self.path is not None:
            if self.path[-1] is not '/':
                pfile = self.path+'/parametric_disk_'+self.prescription
            else:
                pfile = self.path+'parametric_disk_'+self.prescription
        else: 
            pfile = 'parametric_disk_'+self.prescription
        if not os.path.exists(pfile+'.py'):
            print('The prescription '+pfile+'.py does not exist.  Exiting.')
            return

        # Load the appropriate precription
        pd = importlib.import_module(pfile)

        # Calculate the emission cube
        return pd.parametric_disk(velax, pars, fixed)


    """ Spectral Response Functions (SRF) """
    def get_SRF(self, srf_type, nover=1):
        
        if srf_type == 'ALMA':
            # if spectra are oversampled, use the full kernel
            if nover > 1:
                chix = np.arange(25 * nover) / nover
                xch = chix - np.mean(chix)
                srf_ = 0.5 * np.sinc(xch) + \
                       0.25 * np.sinc(xch - 1) + 0.25 * np.sinc(xch + 1)
            # otherwise use the simplification of an "in-place" convolution
            else:
                srf_ = np.array([0.00, 0.25, 0.50, 0.25, 0.00])
        else:
            print('Which SRF?')
        
        return srf_ / np.sum(srf_)


    """ Generate simulated data ('model') """
    def model(self, datadict, pars,
              restfreq=230.538e9, FOV=5.0, npix=256, dist=150,
              chpad=3, nover=None, noise_inject=None,
              doppcorr='approx', SRF='ALMA'):

        # Copy the input data format to a model
        modeldict = copy.deepcopy(datadict)

        # Loop over constituent observations to calculate modelsets
        EBlist = range(datadict['nEB'])
        for EB in EBlist:
            modeldict[str(EB)] = self.modelset(datadict[str(EB)], pars,
                                               restfreq=restfreq, FOV=FOV,
                                               npix=npix, dist=dist,
                                               chpad=chpad, nover=nover,
                                               noise_inject=noise_inject,
                                               doppcorr=doppcorr, SRF=SRF)

        return modeldict


    """ Generate simulated dataset ('modelset') """
    def modelset(self, dset, pars,
                 restfreq=230.538e9, FOV=5.0, npix=256, dist=150, chpad=3, 
                 nover=None, noise_inject=None, doppcorr='approx', SRF='ALMA'):

        # Pad the frequency grids
        dnu_LSRK = np.diff(dset.nu_LSRK, axis=1)[:,0]
        _pad = (dset.nu_LSRK[:,0])[:,None] + \
                dnu_LSRK[:,None] * np.arange(-chpad, 0, 1)[None,:]
        pad_ = (dset.nu_LSRK[:,-1])[:,None] + \
                dnu_LSRK[:,None] * np.arange(1, chpad+1, 1)[None,:]
        nuLSRK = np.concatenate((_pad, dset.nu_LSRK, pad_), axis=1)

        # Define upsampled frequency grids if desired
        if nover is not None:
            nchan = dset.nchan + 2 * chpad
            nch = (nchan - 1) * nover + 1
            nu_LSRK = np.empty((dset.nstamps, nch))
            for it in range(dset.nstamps):
                nu_LSRK[it,:] = np.interp(np.arange((nchan - 1) * nover + 1),
                                          np.arange(0, nchan * nover, nover),
                                          nuLSRK[it,:])
        else:
            nu_LSRK = 1. * nuLSRK
            nch = nu_LSRK.shape[1]
            nover = 1

        # Calculate LSRK velocities
        v_LSRK = sc.c * (1 - nu_LSRK / restfreq)


        ### - Compute the model visibilities
        mvis_ = np.squeeze(np.empty((dset.npol, nch, dset.nvis, 2)))

        # *Exact* Doppler correction calculation
        if doppcorr == 'exact':
            for itime in range(dset.nstamps):
                # track the steps
                print('timestamp '+str(itime+1)+' / '+str(dset.nstamps))

                # make a cube
                icube = self.cube(v_LSRK[itime,:], pars, restfreq=restfreq,
                                  FOV=FOV, npix=npix, dist=dist)

                # visibility indices for this timestamp only
                ixl = np.min(np.where(dset.tstamp == itime))
                ixh = np.max(np.where(dset.tstamp == itime)) + 1

                # sample the FFT on the (u, v) spacings
                mvis = vis_sample(imagefile=icube, 
                                  uu=dset.ulam[ixl:ixh], vv=dset.vlam[ixl:ixh],
                                  mu_RA=pars[-2], mu_DEC=pars[-1], 
                                  mod_interp=False).T

                # populate the results in the output array *for this stamp*
                mvis_[0,:,ixl:ixh,0] = mvis.real
                mvis_[1,:,ixl:ixh,0] = mvis.real
                mvis_[0,:,ixl:ixh,1] = mvis.imag
                mvis_[1,:,ixl:ixh,1] = mvis.imag

        else:
            # velocities at the mid-point timestamp of this EB
            mid_stamp = int(np.round(nu_LSRK.shape[0] / 2))
            v_model = v_LSRK[mid_stamp,:]

            # make a cube
            icube = self.cube(v_model, pars, restfreq=restfreq,
                              FOV=FOV, npix=npix, dist=dist)

            # sample the FFT on the (u, v) spacings
            mvis = vis_sample(imagefile=icube, uu=dset.ulam, vv=dset.vlam, 
                              mu_RA=pars[-2], mu_DEC=pars[-1], 
                              mod_interp=False).T

            # distribute to different timestamps by interpolation
            for itime in range(dset.nstamps):
                ixl = np.min(np.where(dset.tstamp == itime))
                ixh = np.max(np.where(dset.tstamp == itime)) + 1
                fint = interp1d(v_model, mvis[:,ixl:ixh], axis=0, kind='cubic',
                                fill_value='extrapolate')
                interp_vis = fint(v_LSRK[itime,:])
                mvis_[0,:,ixl:ixh,0] = interp_vis.real
                mvis_[1,:,ixl:ixh,0] = interp_vis.real
                mvis_[0,:,ixl:ixh,1] = interp_vis.imag
                mvis_[1,:,ixl:ixh,1] = interp_vis.imag

        # Convolve with the spectral response function (SRF)
        if SRF is not None:
            kernel = self.get_SRF(SRF, nover=nover)
            mvis_pure = convolve1d(mvis_, kernel, axis=1, mode='nearest')
        else:
            mvis_pure = 1. * mvis_

        # Noise injection (if desired)
        if noise_inject is not None:
            # Scale input RMS for desired noise per vis-chan-pol
            sigma_out = 1e-3 * noise_inject * np.sqrt(dset.npol * dset.nvis)

            # Scale to account for spectral oversampling and SRF
            sigma_noise = sigma_out * np.sqrt(nover * 8./3.)

            # Random Gaussian noise draws
            noise = np.random.normal(0, sigma_noise,
                                     (dset.npol, nch, dset.nvis, 2))
            noise = np.squeeze(noise)

        # Decimate, strip pads, and return a model dataset
        if noise_inject is None:
            mvis_pure = mvis_pure[:,::nover,:,:]
            mvis_pure = mvis_pure[:,chpad:-chpad,:,:]
            mvis_p = mvis_pure[:,:,:,0] + 1j * mvis_pure[:,:,:,1]
            mset = dataset(dset.um, dset.vm, mvis_p, dset.wgt, dset.nu_TOPO,
                           dset.nu_LSRK, dset.tstamp)
            return mset
