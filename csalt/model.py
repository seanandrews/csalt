import os
import sys
import time
import importlib
import numpy as np
import warnings
import copy
import casatools
from casatasks import (simobserve, concat)
from csalt.helpers import *
import scipy.constants as sc
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d
from scipy import linalg
from vis_sample import vis_sample
from vis_sample.classes import SkyImage


"""
    The dataset class for transporting visibility data.
"""
class dataset:

    def __init__(self, um, vm, vis, wgt, nu_TOPO, nu_LSRK, tstamp_ID):

        # Spectral frequencies in Hz units (LSRK for each timestamp)
        self.nu_TOPO = nu_TOPO
        self.nu_LSRK = nu_LSRK
        self.nchan = len(nu_TOPO)

        # Spatial frequencies in meters and lambda units
        self.um = um
        self.vm = vm
        self.ulam = self.um * np.mean(self.nu_TOPO) / sc.c
        self.vlam = self.vm * np.mean(self.nu_TOPO) / sc.c

        # Visibilities, weights, and timestamp IDs
        self.vis = vis
        self.wgt = wgt
        self.tstamp = tstamp_ID

        # Utility size trackers
        self.npol, self.nvis = vis.shape[0], vis.shape[2]
        self.nstamps = len(np.unique(tstamp_ID))


"""
    The model class that encapsulates the CSALT framework.
"""
class model:

    def __init__(self, prescription, path='', quiet=True):

        if quiet:
            warnings.filterwarnings("ignore")

        if np.logical_or((path != os.getcwd()), (path is not None)):
            sys.path.append(path)
            self.path = path
        else:
            self.path = ''
        self.prescription = prescription


    """ 
        Generate a cube 
    """
    def cube(self, velax, pars, 
             restfreq=230.538e9, FOV=5.0, Npix=256, dist=150, cfg_dict={}):

        # Parse inputs
        if isinstance(velax, list): 
            velax = np.array(velax)
        fixed = restfreq, FOV, Npix, dist, cfg_dict

        # Load the appropriate prescription
        pfile = 'parametric_disk_'+self.prescription
        if not os.path.exists(self.path+pfile+'.py'):
            print('The prescription '+pfile+'.py does not exist.  Exiting.')
            sys.exit()
        sys.path.append(self.path)
        pd = importlib.import_module(pfile)

        # Calculate the emission cube
        return pd.parametric_disk(velax, pars, fixed)


    """ 
        Spectral Response Functions (SRF) 
    """
    def SRF_kernel(self, srf_type, Nup=1):
        # Full-resolution cases for up-sampled spectra
        if Nup > 1:
            chix = np.arange(25 * Nup) / Nup
            xch = chix - np.mean(chix)
            
            # (F)XF correlators
            if srf_type in ['ALMA', 'VLA']:
                srf = 0.5 * np.sinc(xch) + \
                      0.25 * (np.sinc(xch - 1) + np.sinc(xch + 1))
            # FX correlators
            elif srf_type in ['SMA', 'NOEMA']:
                srf = (np.sinc(xch))**2
            # ALMA-WSU
            elif srf_type in ['ALMA-WSU']:
                _wsu = np.load('/pool/asha0/SCIENCE/csalt/csalt/data/'+\
                               'WSU_SRF.npz')
                wint = interp1d(_wsu['chix'], _wsu['srf'], 
                                fill_value='extrapolate', kind='cubic')
                srf = wint(xch)
            # break
            else:
                print('I do not know that SRF type.  Exiting.')
                sys.exit()

            return srf / np.sum(srf)

        # Approximations for sampled-in-place spectra
        else:
            # (F)XF correlators
            if srf_type in ['ALMA', 'VLA']:
                srf = np.array([0.00, 0.25, 0.50, 0.25, 0.00])
            # others
            else:
                srf = np.array([0.00, 0.00, 1.00, 0.00, 0.00])

            return srf
        

    """ Generate simulated data ('modeldict') """
    def modeldict(self, ddict, pars, kwargs=None):

        # Populate keywords from kwargs dictionary
        kw = {} if kwargs is None else kwargs
        if 'restfreq' not in kw:
            kw['restfreq'] = 230.538e9
        if 'FOV' not in kw:
            kw['FOV'] = 5.0
        if 'Npix' not in kw:
            kw['Npix'] = 256
        if 'dist' not in kw:
            kw['dist'] = 150.
        if 'chpad' not in kw:
            kw['chpad'] = 2
        if 'Nup' not in kw:
            kw['Nup'] = None
        if 'noise_inject' not in kw:
            kw['noise_inject'] = None
        if 'doppcorr' not in kw:
            kw['doppcorr'] = 'approx'
        if 'SRF' not in kw:
            kw['SRF'] = 'ALMA'
        if 'online_avg' not in kw:
            kw['online_avg'] = 1
        if 'cfg_dict' not in kw:
            kw['cfg_dict'] = {}

        # List of input EBs
        EBlist = range(ddict['Nobs'])

        # Copy the input data format to a model
        if kw['noise_inject'] is None:
            m_ = copy.deepcopy(ddict)
            for EB in EBlist:
                m_[str(EB)] = self.modelset(ddict[str(EB)], pars,
                                               restfreq=kw['restfreq'], 
                                               FOV=kw['FOV'], 
                                               Npix=kw['Npix'], 
                                               dist=kw['dist'], 
                                               chpad=kw['chpad'], 
                                               Nup=kw['Nup'],
                                               noise_inject=kw['noise_inject'],
                                               doppcorr=kw['doppcorr'], 
                                               SRF=kw['SRF'],
                                               online_avg=kw['online_avg'],
                                               cfg_dict=kw['cfg_dict'])
            return m_
        else:
            p_, n_ = copy.deepcopy(ddict), copy.deepcopy(ddict)
            for EB in EBlist:
                p_[str(EB)], n_[str(EB)] = self.modelset(ddict[str(EB)], pars,
                                                restfreq=kw['restfreq'],
                                                FOV=kw['FOV'],
                                                Npix=kw['Npix'],
                                                dist=kw['dist'],
                                                chpad=kw['chpad'],
                                                Nup=kw['Nup'],
                                                noise_inject=kw['noise_inject'],
                                                doppcorr=kw['doppcorr'],
                                                SRF=kw['SRF'],
                                                online_avg=kw['online_avg'],
                                                cfg_dict=kw['cfg_dict'])
            return p_, n_



    """ Generate simulated dataset ('modelset') """
    def modelset(self, dset, pars,
                 restfreq=230.538e9, FOV=5.0, Npix=256, dist=150, chpad=2, 
                 Nup=None, noise_inject=None, doppcorr='approx', SRF='ALMA',
                 online_avg=1, gcf_holder=None, corr_cache=None, 
                 return_holders=False, cfg_dict={}):

        """ Prepare the spectral grids: format = [timestamps, channels] """
        # Revert to original un-averaged grid if online binning was applied
        if online_avg > 1:
            # mean LSRK channel spacings
            _dnu = np.mean(np.diff(dset.nu_LSRK, axis=1), axis=1)

            # get the pre-averaged LSRK frequencies for each timestamp
            nu_LSRK = np.empty((dset.nstamps, online_avg * dset.nchan))
            for it in range(dset.nstamps):
                pnu = np.arange(dset.nu_LSRK[it,0], dset.nu_LSRK[it,-1], 
                                _dnu[it] / online_avg) - \
                      (online_avg - 1) * _dnu[it] / (2 * online_avg)
                while len(pnu) < nu_LSRK.shape[1]:
                    pnu = np.append(pnu, pnu[-1] + _dnu[it] / online_avg)
                nu_LSRK[it,:] = 1. * pnu   
        else:
            nu_LSRK = 1. * dset.nu_LSRK

        # Pad the LSRK frequencies
        dnu_n = (np.diff(nu_LSRK, axis=1)[:,0])[:,None]
        _pad = (nu_LSRK[:,0])[:,None] + \
                dnu_n * np.arange(-chpad, 0, 1)[None,:]
        pad_ = (nu_LSRK[:,-1])[:,None] + \
                dnu_n * np.arange(1, chpad+1, 1)[None,:]
        nu_ = np.concatenate((_pad, nu_LSRK, pad_), axis=1)

        # Upsample the LSRK frequencies (if requested)
        if Nup is not None:
            nchan = nu_.shape[1]
            nu = np.empty((dset.nstamps, (nchan - 1) * Nup + 1))
            for it in range(dset.nstamps):
                nu[it,:] = np.interp(np.arange((nchan - 1) * Nup + 1),
                                     np.arange(0, nchan * Nup, Nup), nu_[it,:])
        else:
            nu, Nup = 1. * nu_, 1
        nch = nu.shape[1]

        # Calculate LSRK velocities
        vel = sc.c * (1 - nu / restfreq)

        ### - Compute the model visibilities
        mvis_ = np.squeeze(np.ones((dset.npol, nch, dset.nvis, 2)))
        print(mvis_.shape)

        # *Exact* Doppler correction calculation
        if doppcorr == 'exact':
            for itime in range(dset.nstamps):
                # track the steps
                print('timestamp '+str(itime+1)+' / '+str(dset.nstamps))

                # make a cube
                icube = self.cube(vel[itime,:], pars, restfreq=restfreq,
                                  FOV=FOV, Npix=Npix, dist=dist, 
                                  cfg_dict=cfg_dict)

                # visibility indices for this timestamp only
                ixl = np.min(np.where(dset.tstamp == itime))
                ixh = np.max(np.where(dset.tstamp == itime)) + 1

                # sample the FFT on the (u, v) spacings
                mvis = vis_sample(imagefile=icube, 
                                  uu=dset.ulam[ixl:ixh], vv=dset.vlam[ixl:ixh],
                                  gcf_holder=gcf_holder, corr_cache=corr_cache,
                                  mu_RA=pars[-2], mu_DEC=pars[-1], 
                                  mod_interp=False).T

                # populate the results in the output array *for this stamp*
                mvis_[0,:,ixl:ixh,0] = mvis.real
                mvis_[1,:,ixl:ixh,0] = mvis.real
                mvis_[0,:,ixl:ixh,1] = mvis.imag
                mvis_[1,:,ixl:ixh,1] = mvis.imag

        elif doppcorr == 'approx':
            # velocities at the mid-point timestamp of this EB
            v_model = vel[int(np.round(nu.shape[0] / 2)),:]

            # make a cube
            icube = self.cube(v_model, pars, restfreq=restfreq,
                              FOV=FOV, Npix=Npix, dist=dist, cfg_dict=cfg_dict)

            # sample the FFT on the (u, v) spacings
            if return_holders:
                mvis, gcf, corr = vis_sample(imagefile=icube, 
                                             uu=dset.ulam, vv=dset.vlam,
                                             mu_RA=pars[-2], mu_DEC=pars[-1],
                                             return_gcf=True, 
                                             return_corr_cache=True,
                                             mod_interp=False)
                return mvis.T, gcf, corr
            else:
                mvis = vis_sample(imagefile=icube, uu=dset.ulam, vv=dset.vlam, 
                                  gcf_holder=gcf_holder, corr_cache=corr_cache,
                                  mu_RA=pars[-2], mu_DEC=pars[-1], 
                                  mod_interp=False).T

            # distribute to different timestamps by interpolation
            for itime in range(dset.nstamps):
                ixl = np.min(np.where(dset.tstamp == itime))
                ixh = np.max(np.where(dset.tstamp == itime)) + 1
                fint = interp1d(v_model, mvis[:,ixl:ixh], axis=0, kind='cubic',
                                fill_value='extrapolate')
                interp_vis = fint(vel[itime,:])
                mvis_[0,:,ixl:ixh,0] = interp_vis.real
                mvis_[1,:,ixl:ixh,0] = interp_vis.real
                mvis_[0,:,ixl:ixh,1] = interp_vis.imag
                mvis_[1,:,ixl:ixh,1] = interp_vis.imag
        elif doppcorr is None:
            print('I AM NOT DOING A DOPPLER CORRECTION!')
            # make a cube
            icube = self.cube(vel[0,:], pars, restfreq=restfreq,
                              FOV=FOV, Npix=Npix, dist=dist, cfg_dict=cfg_dict)

            # sample the FFT on the (u, v) spacings
            if return_holders:
                mvis, gcf, corr = vis_sample(imagefile=icube, 
                                             uu=dset.ulam, vv=dset.vlam,
                                             mu_RA=pars[-2], mu_DEC=pars[-1],
                                             return_gcf=True,
                                             return_corr_cache=True,
                                             mod_interp=False)
                return mvis.T, gcf, corr
            else:
                mvis = vis_sample(imagefile=icube, uu=dset.ulam, vv=dset.vlam,
                                  gcf_holder=gcf_holder, corr_cache=corr_cache,
                                  mu_RA=pars[-2], mu_DEC=pars[-1],
                                  mod_interp=False).T
                mvis_[0,:,:,0] = 1. * mvis.real
                mvis_[1,:,:,0] = 1. * mvis.real
                mvis_[0,:,:,1] = 1. * mvis.imag
                mvis_[1,:,:,1] = 1. * mvis.imag
                 
        else:
            print('You need to specify a doppcorr method.  Exiting.')
            sys.exit()

        # Convolve with the spectral response function (SRF)
        if SRF is not None:
            kernel = self.SRF_kernel(SRF, Nup=Nup)
            mvis_pure = convolve1d(mvis_, kernel, axis=1, mode='nearest')
        else:
            print('I AM NOT DOING AN SRF CONVOLUTION!')
            mvis_pure = 1. * mvis_

        # Decimate and package the pure visibility spectra
        mvis_pure = mvis_pure[:,::Nup,:,:]
        mvis_pure = mvis_pure[:,chpad:-chpad,:,:]
        # re-bin to emulate online averaging
        if online_avg > 1:
            t_ = np.transpose(mvis_pure, (1, 0, 2, 3))
            _ = [np.take(t_, np.arange(i*online_avg, 
                                       (i+1)*online_avg), 0).mean(axis=0) \
                 for i in np.arange(mvis_pure.shape[1] // online_avg)]
            mvis_pure = np.array(_).transpose((1, 0, 2, 3))
        mvis_p = mvis_pure[:,:,:,0] + 1j * mvis_pure[:,:,:,1]
        mset_p = dataset(dset.um, dset.vm, mvis_p, dset.wgt, dset.nu_TOPO,
                         dset.nu_LSRK, dset.tstamp)

        # Return the pure or pure and noisy models
        if noise_inject is None:
            return mset_p
        else:
            # Calculate noise spectra
            noise = self.calc_noise(noise_inject, dset, 
                                    nchan=nch, Nup=Nup, SRF=SRF)

            # SRF convolution of noisy data
            if SRF is not None:
                mvis_noisy = convolve1d(mvis_ + noise, kernel, axis=1, 
                                        mode='nearest')
            else:
                mvis_noisy = mvis_ + noise

            # Decimate and package the pure visibility spectra
            mvis_noisy = mvis_noisy[:,::Nup,:,:]
            mvis_noisy = mvis_noisy[:,chpad:-chpad,:,:]
            # re-bin to emulate online averaging
            if online_avg > 1:
                t_ = np.transpose(mvis_noisy, (1, 0, 2, 3))
                _ = [np.take(t_, np.arange(i*online_avg, 
                                           (i+1)*online_avg), 0).mean(axis=0) \
                     for i in np.arange(mvis_noisy.shape[1] // online_avg)]
                mvis_noisy = np.array(_).transpose((1, 0, 2, 3))
            mvis_n = mvis_noisy[:,:,:,0] + 1j * mvis_noisy[:,:,:,1]
            mset_n = dataset(dset.um, dset.vm, mvis_n, dset.wgt, dset.nu_TOPO,
                             dset.nu_LSRK, dset.tstamp)

            return mset_p, mset_n


    """
        A noise calculator
    """
    def calc_noise(self, noise_inject, dataset, nchan=1, Nup=None, SRF='ALMA'):

        # Scale input RMS for desired noise per vis-chan-pol
        sigma_out = noise_inject * np.sqrt(dataset.npol * dataset.nvis)

        # Scale to account for spectral up-sampling and SRF (TEMPORARY)
        if Nup is None: Nup = 1
        if SRF in ['ALMA', 'VLA']:
            fcov = 8./3.
        else:
            fcov = 1.
        sigma_noise = sigma_out * np.sqrt(Nup * fcov)

        # Random Gaussian noise draws
        noise = np.random.normal(0, sigma_noise,
                                 (dataset.npol, nchan, dataset.nvis, 2))
        
        return np.squeeze(noise)



    """
        Create a blank MS template 
    """
    def template_MS(self, msfile, config='', t_total='1min', 
                    sim_save=False, RA='16:00:00.00', DEC='-30:00:00.00', 
                    restfreq=230.538e9, dnu_native=122e3, V_span=10e3, 
                    V_tune=0.0e3, t_integ='6s', HA_0='0h', date='2023/03/20',
                    observatory='ALMA', force_to_LSRK=False):

        # Load the measures tools
        me = casatools.measures()

        # Parse / determine the executions
        if np.isscalar(config): config = np.array([config])
        Nobs = len(config)

        # things to format check:
                # RA, DEC have full hms/dms string formatting
                # HA_0 has proper string formatting
                # date has proper string formatting
                # msfile has proper '.ms' ending

        # If only scalars specified for keywords, copy them for each execution
        if np.isscalar(t_total): t_total = np.repeat(t_total, Nobs)
        if np.isscalar(dnu_native): dnu_native = np.repeat(dnu_native, Nobs)
        if np.isscalar(V_span): V_span = np.repeat(V_span, Nobs)
        if np.isscalar(V_tune): V_tune = np.repeat(V_tune, Nobs)
        if np.isscalar(HA_0): HA_0 = np.repeat(HA_0, Nobs)
        if np.isscalar(date): date = np.repeat(date, Nobs)
        if np.isscalar(t_integ): t_integ = np.repeat(t_integ, Nobs)

        # Move to simulation space
        cwd = os.getcwd()
        out_path, out_name = os.path.split(msfile)
        if out_path != '':
            if not os.path.exists(out_path): os.system('mkdir '+out_path)
            os.chdir(out_path)

        # Loop over execution blocks
        obs_files = []
        for i in range(Nobs):

            # Calculate the number of native channels
            nch = 2 * int(V_span[i] / (sc.c * dnu_native[i] / restfreq)) + 1

            # Calculate the LST starting time of the execution block
            h, m, s = RA.split(':')
            LST_h = int(h) + int(m)/60 + float(s)/3600 + float(HA_0[i][:-1])
            LST_0 = str(datetime.timedelta(hours=LST_h))
            if (LST_h < 10.): LST_0 = '0' + LST_0

            # Get the observatory longitude
            obs_long = np.degrees(me.observatory(observatory)['m0']['value'])

            # Calculate the UT starting time of the execution block
            UT_0 = LST_to_UTC(date[i], LST_0, obs_long)

            # Calculate the TOPO tuning frequency
            nu_tune_0 = doppler_set(restfreq, V_tune[i], UT_0, RA, DEC,
                                    observatory=observatory)

            # Generate a dummy (empty) cube
            ia = casatools.image()
            dummy = ia.makearray(v=0.001, shape=[64, 64, 4, nch])
            ia.fromarray(outfile='dummy.image', pixels=dummy, overwrite=True)
            ia.done()

            # Compute the midpoint HA
            if (t_total[i][-1] == 'h'):
                tdur = float(t_total[i][:-1])
            elif (t_total[i][-3:] == 'min'):
                tdur = float(t_total[i][:-3]) / 60
            elif (t_total[i][-1] == 's'):
                tdur = float(t_total[i][:-1]) / 3600
            HA_mid = str(float(HA_0[i][:-1]) + 0.5 * tdur) +'h'

            # Generate the template sub-MS file
            simobserve(project=out_name[:-3]+'_'+str(i)+'.sim',
                       skymodel='dummy.image',
                       antennalist=config[i],
                       totaltime=t_total[i],
                       integration=t_integ[i],
                       thermalnoise='',
                       hourangle=HA_mid,
                       indirection='J2000 '+RA+' '+DEC,
                       refdate=date[i],
                       incell='0.01arcsec',
                       mapsize='5arcsec',
                       incenter=str(nu_tune_0 / 1e9)+'GHz',
                       inwidth=str(dnu_native[i] * 1e-3)+'kHz',
                       outframe='TOPO')

            # Pull the sub-MS file out of the simulation directory
            cfg_dir, cfg_file = os.path.split(config[i])
            sim_MS = out_name[:-3]+'_'+str(i)+'.sim/'+out_name[:-3]+ \
                     '_'+str(i)+'.sim.'+cfg_file[:-4]+'.ms'
            os.system('rm -rf '+out_name[:-3]+'_'+str(i)+'.ms*')
            os.system('mv '+sim_MS+' '+out_name[:-3]+'_'+str(i)+'.ms')

            # Delete the simulation directory if requested
            if not sim_save:
                os.system('rm -rf '+out_name[:-3]+'_'+str(i)+'.sim')

            # Delete the dummy (empty) cube
            os.system('rm -rf dummy.image')

            # Update the file list
            obs_files += [out_name[:-3]+'_'+str(i)+'.ms']

        # Concatenate the sub-MS files into a single MS
        os.system('rm -rf '+out_name[:-3]+'.ms*')
        if Nobs > 1:
            concat(vis=obs_files,
                   concatvis=out_name[:-3]+'.ms',
                   dirtol='0.1arcsec',
                   copypointing=False)
        else:
            os.system('mv '+out_name[:-3]+'_0.ms '+out_name[:-3]+'.ms')

        # Clean up
        os.system('rm -rf '+out_name[:-3]+'_*.ms*')
        os.chdir(cwd)

        return None


    """
        Function to parse and package visibility data for inference
    """
    def fitdata(self, msfile,
                vra=None, vcensor=None, restfreq=230.538e9, chbin=1, 
                well_cond=300):

        # Load the data from the MS file into a dictionary
        data_dict = read_MS(msfile)

        # If chbin is a scalar, distribute it over the Nobs executions
        if np.isscalar(chbin):
            chbin = chbin * np.ones(data_dict['Nobs'], dtype=int)
        else:
            if isinstance(chbin, list):
                chbin = np.asarray(chbin)

        # If vra is a list, make it an array
        if isinstance(vra, list):
            vra = np.asarray(vra)

        # Assign an output dictionary
        out_dict = {'Nobs': data_dict['Nobs'], 'chbin': chbin}

        # Force chbin <= 2
        if np.any(chbin > 2):
            print('Forcing chbin --> 2; do not over-bin your data!')
        chbin[chbin > 2] = 2

        # Loop over executions
        for i in range(data_dict['Nobs']):

            # Pull the dataset object for this execution
            data = data_dict[str(i)]

            # If necessary, distribute weights across the spectrum
            if not data.wgt.shape == data.vis.shape:
                data.wgt = np.tile(data.wgt, (data.nchan, 1, 1))
                data.wgt = np.rollaxis(data.wgt, 1, 0)

            # Convert the LSRK frequency grid to velocities
            v_LSRK = sc.c * (1 - data.nu_LSRK / restfreq)

            # Fix direction of desired velocity bounds
            if vra is None: vra = np.array([-1e5, 1e5])
            dv, dvra = np.diff(v_LSRK, axis=1), np.diff(vra)
            if np.logical_or(np.logical_and(np.all(dv < 0), np.all(dvra > 0)),
                             np.logical_and(np.all(dv < 0), np.all(dvra < 0))):
                vra_ = vra[::-1]
            else:
                vra_ = 1. * vra
            sgn_v = np.sign(np.diff(vra_)[0])

            # Find where to clip to lie within the desired velocity bounds
            midstamp = int(data.nstamps / 2)
            ixl = np.abs(v_LSRK[midstamp,:] - vra_[0]).argmin()
            ixh = np.abs(v_LSRK[midstamp,:] - vra_[1]).argmin()

            # Adjust indices to ensure they are evenly divisible by chbin
            if np.logical_and((chbin[i] > 1), ((ixh - ixl) % chbin[i] != 0)):
                # bounded at upper edge only
                if np.logical_and((ixh == (data.nchan - 1)), (ixl > 0)):
                    ixl -= 1
                # bounded at lower edge only
                elif np.logical_and((ixh < (data.nchan - 1)), (ixl == 0)):
                    ixh += 1
                # bounded at both edges
                elif np.logical_and((ixh == (data.nchan - 1)), (ixl == 0)):
                    ixh -= 1
                # unbounded on either side
                else:
                    ixh += 1

            # Clip the data to cover only the frequencies of interest
            inu_TOPO = data.nu_TOPO[ixl:ixh]
            inu_LSRK = data.nu_LSRK[:,ixl:ixh]
            iv_LSRK = v_LSRK[:,ixl:ixh]
            inchan = inu_LSRK.shape[1]
            ivis = data.vis[:,ixl:ixh,:]
            iwgt = data.wgt[:,ixl:ixh,:]

            # Binning operations
            binned = True if chbin[i] > 1 else False
            if binned:
                bnchan = int(inchan / chbin[i])
                bshape = (data.npol, -1, chbin[i], data.nvis)
                wt = iwgt.reshape(bshape)
                bvis = np.average(ivis.reshape(bshape), weights=wt, axis=2)
                bwgt = np.sum(wt, axis=2)

            # Channel censoring
            if vcensor is not None:
                cens_chans = np.ones(inchan, dtype='bool')
                for j in range(len(vcensor)):
                    if sgn_v < 0:
                        vcens = (vcensor[j])[::-1]
                    else:
                        vcens = vcensor[j]
                    cixl = np.abs(iv_LSRK[midstamp,:] - vcens[0]).argmin()
                    cixh = np.abs(iv_LSRK[midstamp,:] - vcens[1]).argmin()
                    cens_chans[cixl:cixh+1] = False
                iwgt[:,cens_chans == False,:] = 0

                if binned:
                    bcens_chans = np.all(cens_chans.reshape((-1, chbin[i])),
                                         axis=1)
                    bwgt[:,cens_chans == False,:] = 0

            # Pre-calculate the spectral covariance matrix 
            # (** note: this assumes the Hanning kernel for ALMA **)
            if binned:
                scov = (5/16) * np.eye(bnchan) \
                       + (3/32) * (np.eye(bnchan, k=-1) + np.eye(bnchan, k=1))
            else:
                scov = (3/8) * np.eye(inchan) \
                       + (1/4) * (np.eye(inchan, k=-1) + np.eye(inchan, k=1)) \
                       + (1/16) * (np.eye(inchan, k=-2) + np.eye(inchan, k=2))

            # If well-conditioned (usually for binned), do direct inversion
            if np.linalg.cond(scov) <= well_cond:
                print('EB '+str(i)+' SCOV inverted with direct calculation.')
                scov_inv = linalg.inv(scov)

            # See if you can use Cholesky factorization
            else:
                chol = linalg.cholesky(scov)
                if np.linalg.cond(chol) <= well_cond:
                    print('EB '+str(i)+' SCOV inverted with Cholesky'
                          + ' factorization')
                    scov_inv = np.dot(linalg.inv(chol), linalg.inv(chol.T))

                # Otherwise use SVD
                else:
                    print('EB '+str(i)+' SCOV inverted with singular value'
                          + ' decomposition')
                    uu, ss, vv = linalg.svd(scov)
                    scov_inv = np.dot(vv.T, np.dot(np.diag(ss**-1), uu.T))

            # Pre-calculate the log-likelihood normalization term
            dterm = np.empty((data.npol, data.nvis))
            for ii in range(data.nvis):
                for jj in range(data.npol):
                    _wgt = bwgt[jj,:,ii] if binned else iwgt[jj,:,ii]
                    sgn, lndet = np.linalg.slogdet(scov / _wgt)
                    dterm[jj,ii] = sgn * lndet
            _ = np.prod(bvis.shape) if binned else np.prod(ivis.shape)
            lnL0 = -0.5 * (_ * np.log(2 * np.pi) + np.sum(dterm))

            # Package the output data into the dictionary
            if binned:
                odata = dataset(data.um, data.vm, bvis, bwgt, inu_TOPO,
                                inu_LSRK, data.tstamp)
            else:
                odata = dataset(data.um, data.vm, ivis, iwgt, inu_TOPO,
                                inu_LSRK, data.tstamp)
            out_dict[str(i)] = odata

            # Package additional information into the dictionary
            out_dict['invcov_'+str(i)] = scov_inv
            out_dict['lnL0_'+str(i)] = lnL0
            out_dict['gcf_'+str(i)] = None
            out_dict['corr_'+str(i)] = None

        # Return the output dictionary
        return out_dict


    """
        Sample the posteriors.
    """
    def sample_posteriors(self, msfile, vra=None, vcensor=None, kwargs=None,
                          restfreq=230.538e9, chbin=1, well_cond=300,
                          Nwalk=75, Ninits=20, Nsteps=1000, 
                          outpost='stdout.h5', append=False, Nthreads=6):

        import emcee
        from multiprocessing import Pool
        if Nthreads > 1:
            os.environ["OMP_NUM_THREADS"] = "1"

        # Parse the data into proper format
        infdata = self.fitdata(msfile, vra=vra, vcensor=vcensor, 
                               restfreq=restfreq, chbin=chbin, 
                               well_cond=well_cond)

        # Initialize the parameters using random draws from the priors
        priors = importlib.import_module('priors_'+self.prescription)
        Ndim = len(priors.pri_pars)
        p0 = np.empty((Nwalk, Ndim))
        for ix in range(Ndim):
            if ix == 9:
                p0[:,ix] = np.sqrt(2 * sc.k * p0[:,6] / (28 * (sc.m_p+sc.m_e)))
            else:
                _ = [str(priors.pri_pars[ix][ip])+', '
                     for ip in range(len(priors.pri_pars[ix]))]
                cmd = 'np.random.'+priors.pri_types[ix]+ \
                      '('+"".join(_)+str(Nwalk)+')'
                p0[:,ix] = eval(cmd)

        # Acquire and store the GCF and CORR caches for iterative sampling
        for i in range(infdata['Nobs']):
            _, gcf, corr = self.modelset(infdata[str(i)], p0[0], 
                                         restfreq=restfreq, 
                                         FOV=kwargs['FOV'],
                                         Npix=kwargs['Npix'], 
                                         dist=kwargs['dist'],
                                         return_holders=True)
            infdata['gcf_'+str(i)] = gcf
            infdata['corr_'+str(i)] = corr

        # Declare the data and kwargs as globals (for speed in pickling)
        global fdata
        global kw
        fdata = copy.deepcopy(infdata)
        kw = copy.deepcopy(kwargs)

        # Populate keywords from kwargs dictionary
        if 'restfreq' not in kw:
            kw['restfreq'] = restfreq
        if 'FOV' not in kw:
            kw['FOV'] = 5.0
        if 'Npix' not in kw:
            kw['Npix'] = 256
        if 'dist' not in kw:
            kw['dist'] = 150.
        if 'chpad' not in kw:
            kw['chpad'] = 2
        if 'Nup' not in kw:
            kw['Nup'] = None
        if 'noise_inject' not in kw:
            kw['noise_inject'] = None
        if 'doppcorr' not in kw:
            kw['doppcorr'] = 'approx'
        if 'SRF' not in kw:
            kw['SRF'] = 'ALMA'

        if not append:
            # Initialize the MCMC walkers
            with Pool(processes=Nthreads) as pool:
                isamp = emcee.EnsembleSampler(Nwalk, Ndim, self.log_posterior,
                                              pool=pool)
                isamp.run_mcmc(p0, Ninits, progress=True)
            isamples = isamp.get_chain()   # [Ninits, Nwalk, Ndim]-shaped
            lop0 = np.quantile(isamples[-1, :, :], 0.25, axis=0)
            hip0 = np.quantile(isamples[-1, :, :], 0.75, axis=0)
            p00 = [np.random.uniform(lop0, hip0, Ndim) for iw in range(Nwalk)]

            # Prepare the backend
            os.system('rm -rf '+outpost)
            backend = emcee.backends.HDFBackend(outpost)
            backend.reset(Nwalk, Ndim)

            # Sample the posterior distribution
            with Pool(processes=Nthreads) as pool:
                samp = emcee.EnsembleSampler(Nwalk, Ndim, self.log_posterior,
                                             pool=pool, backend=backend)
                t0 = time.time()
                samp.run_mcmc(p00, Nsteps, progress=True)
            t1 = time.time()
            print('backend run in ', t1-t0)

        else:
            # Load the old backend
            new_backend = emcee.backends.HDFBackend(outpost)
            
            # Continue sampling the posterior distribution
            with Pool(processes=Nthreads) as pool:
                samp = emcee.EnsembleSampler(Nwalk, Ndim, self.log_posterior,
                                             pool=pool, backend=new_backend)
                t0 = time.time()
                samp.run_mcmc(None, Nsteps, progress=True)
            t1 = time.time()

        print('\n\n    This run took %.2f hours' % ((t1 - t0) / 3600))

        # Release the globals
        del fdata
        del kw

        return samp


    """
        Function to calculate a log-posterior sample.
    """
    def log_posterior(self, theta):

        # Calculate log-prior
        priors = importlib.import_module('priors_'+self.prescription)
        lnT = np.sum(priors.logprior(theta)) * fdata['Nobs']
        if lnT == -np.inf:
            return -np.inf, -np.inf

        # Compute log-likelihood
        lnL = self.log_likelihood(theta, fdata=fdata, kwargs=kw)

        # return the log-posterior and the log-prior
        return lnL + lnT, lnT


    """
        Function to calculate a log-likelihood.
    """
    def log_likelihood(self, theta, fdata=None, kwargs=None):

        # Loop over observations to get likelihood
        logL = 0
        for i in range(fdata['Nobs']):

            # Get the data 
            _data = fdata[str(i)]

            # Calculate the model
            _mdl = self.modelset(_data, theta, restfreq=kwargs['restfreq'],
                                 FOV=kwargs['FOV'], Npix=kwargs['Npix'], 
                                 dist=kwargs['dist'], chpad=kwargs['chpad'],
                                 doppcorr=kwargs['doppcorr'], 
                                 SRF=kwargs['SRF'], 
                                 gcf_holder=fdata['gcf_'+str(i)],
                                 corr_cache=fdata['corr_'+str(i)])

            # Spectrally bin the model visibilities if necessary
            # **technically wrong, since the weights are copied like this; 
            # **ideally would propagate the unbinned weights?
            if fdata['chbin'][i] > 1:
                oshp = (_mdl.npol, -1, fdata['chbin'][i], _mdl.nvis)
                wt = np.rollaxis(np.tile(_data.wgt, (2, 1, 1, 1)), 0, 3)
                mvis = np.average(_mdl.vis.reshape(oshp),
                                  weights=wt.reshape(oshp), axis=2)
            else:
                mvis = 1. * _mdl.vis

            # Compute the residual and variance matrices(stack both pols)
            resid = np.hstack(np.absolute(_data.vis - mvis))
            var = np.hstack(_data.wgt)

            # Compute the log-likelihood (** still needs constant term)
            Cinv = fdata['invcov_'+str(i)]
            logL += -0.5 * np.tensordot(resid, np.dot(Cinv, var * resid))

        return logL
