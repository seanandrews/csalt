import os, sys, importlib
import numpy as np
import warnings
import copy
import scipy.constants as sc
from vis_sample import vis_sample

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
                self.path += '/'
        else: self.path = ''
        pfile = self.path + 'parametric_disk_'+self.prescription
        if not os.path.exists(pfile+'.py'):
            print('The prescription '+pfile+'.py does not exist.  Exiting.')
            return

        # Load the appropriate precription
        pd = importlib.import_module(pfile)

        # Calculate the emission cube
        return pd.parametric_disk(velax, pars, fixed)


    """ Generate simulated data ('model') """
    def model(self, datadict, pars, 
              restfreq=230.538e9, FOV=5.0, npix=256, dist=150,
              obsID=None, chpad=3, oversample=None, noise_inject=None,
              doppcorr='approx'):

        # Copy the input data format to a model
        modeldict = copy.deepcopy(datadict)

        # Loop over constituent observations to calculate modelsets
        EBlist = range(mdl.nEB)
        for EB in EBlist:
            modeldict[str(EB)] = modelset(datadict[str(EB)], pars, 
                                          restfreq=restfreq, FOV=FOV, 
                                          npix=npix, dist=dist, chpad=chpad, 
                                          nover=nover, 
                                          noise_inject=noise_inject, 
                                          doppcorr=doppcorr)

        return modeldict


    """ Generate simulated dataset ('modelset') """
    def modelset(self, dset, pars,
                 restfreq=230.538e9, FOV=5.0, npix=256, dist=150, chpad=3, 
                 nover=None, noise_inject=None, doppcorr='approx'):

        # Pad the frequency grids
        dnu_TOPO = np.diff(dset.nu_TOPO)[0]
        _f = dset.nu_TOPO[0] + dnu_TOPO * np.arange(-chpad, 0, 1)
        f_ = dset.nu_TOPO[-1] + dnu_TOPO * np.arange(1, chpad+1, 1)
        dset.nu_TOPO = np.concatenate((_f, dset.nu_TOPO, f_))

        dnu_LSRK = np.diff(dset.nu_LSRK, axis=1)[:,0]
        _f = (dset.nu_LSRK[:,0])[:,None] + \
              dnu_LSRK[:,None] * np.arange(-chpad, 0, 1)[None,:]
        f_ = (dset.nu_LSRK[:,-1])[:,None] + \
              dnu_LSRK[:,None] * np.arange(1, chpad+1, 1)[None,:]
        dset.nu_LSRK = np.concatenate((_f, dset.nu_LSRK, f_), axis=1)

        # Define upsampled frequency grids if desired
        if nover is not None:
            nchan = dset.nchan + 2 * chpad
            nu_TOPO = np.interp(np.arange((nchan - 1) * nover + 1),
                                np.arange(0, nchan * nover, nover),
                                dset.nu_TOPO)
            nch = len(nu_TOPO)

            nu_LSRK = np.empty((dset.nstamps, nch))
            for it in range(dset.nstamps):
                nu_LSRK[it,:] = np.interp(np.arange((nchan - 1) * nover + 1),
                                np.arange(0, nchan * nover, nover),
                                dset.nu_LSRK[it,:])
        else:
            nu_TOPO = dset.nu_TOPO
            nu_LSRK = dset.nu_LSRK
            nch = len(nu_TOPO)
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
            mvis = vis_sample(imagefile=icube, uu=ulam, vv=vlam, 
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


            
            
