import os, sys, importlib
import numpy as np
import h5py
import scipy.constants as sc

# General visibility dataset object
class dataset:

   def __init__(self, um, vm, vis, wgt, nu_TOPO, nu_LSRK, tstamp_ID):

        # spatial frequencies in meters units
        self.um = um
        self.vm = vm

        # spectral frequencies in Hz units (LSRK for each timestamp)
        self.nu_TOPO = nu_TOPO
        self.nu_LSRK = nu_LSRK

        # data visibilities, weights, and timestamp IDs
        self.vis = vis
        self.npol, self.nchan, self.nvis = vis.shape
        self.wgt = wgt

        self.tstamp = tstamp_ID
        self.nstamps = len(np.unique(tstamp_ID))




# Inference-specific dataset object
class inf_dataset:

    def __init__(self, um, vm, vis_bin, wgt_bin, nu_TOPO, nu_LSRK, tstamp_ID, 
                 iwgt, M_bin, invM_bin, lnL0):

        # spatial frequencies in lambda units
        self.um = um
        self.vm = vm

        # timestamps (indices)
        self.tstamp = tstamp_ID
        self.nstamps = len(np.unique(tstamp_ID))

        # **unbinned** frequencies in Hz units (LSRK for each timestamp)
        self.nu_TOPO = nu_TOPO
        self.nu_LSRK = nu_LSRK

        # **binned** data visibilities, weights
        self.vis = vis_bin
        self.wgt = wgt_bin
        self.iwgt = iwgt
        self.npol, self.nchan, self.nvis = self.vis.shape
        self.chbin = np.int(len(self.nu_TOPO) / self.vis.shape[1])

        # likelihood-related quantities
        self.cov = M_bin
        self.inv_cov = invM_bin
        self.lnL0 = lnL0




# Data parsing to generate inputs for likelihood function
def fitdata(datafile, vra=None, vcensor=None, nu_rest=230.538e9, chbin=2):

    # Load the data from the HDF5 file
    f = h5py.File(datafile+'.h5', "r")

    # Load the relevant attributes
    nobs = f.attrs['nobs']
    f.close()

    # Initialize output data dictionary
    out_dict = {'nobs': nobs}

    # If chbin is a scalar, distribute it over the nobs elements
    if np.isscalar(chbin):
        chbin = chbin * np.ones(nobs, dtype=np.int)
    else:
        chbin = np.asarray(chbin)

    # Loop through each EB
    for i in range(nobs):

        # load the data into a dataset object
        idata = HDF_to_dataset(datafile, grp='EB'+str(i)+'/')

        # if necessary, distribute weights across spectrum
        if not idata.wgt.shape == idata.vis.shape:
            idata.wgt = np.tile(idata.wgt, (idata.nchan, 1, 1))
            idata.wgt = np.rollaxis(idata.wgt, 1, 0)

        # convert the LSRK frequency grid to a velocity grid
        v_LSRK = sc.c * (1 - idata.nu_LSRK / nu_rest)

        # fix direction of desired velocity bounds, based on data format
        if vra is None: vra = [-1e5, 1e5]
        dvi, dvra = np.diff(v_LSRK, axis=1), np.diff(vra)
        if np.logical_or(np.logical_and(np.all(dvi<0), np.all(dvra>0)),
                         np.logical_and(np.all(dvi>0), np.all(dvra<0))): 
            vra = vra[::-1]
        sgn_v = np.sign(np.diff(vra)[0])

        # find where to clip to lie within the desired velocity bounds
        midstamp = np.int(idata.nstamps / 2)
        ixl = np.abs(v_LSRK[midstamp,:] - vra[0]).argmin()
        ixh = np.abs(v_LSRK[midstamp,:] - vra[1]).argmin()

        # reconcile channel set to be evenly divisible by binning factor
        if ((ixh - ixl + (ixh - ixl) % chbin[i]) < idata.nchan):
            for j in range((ixh - ixl) % chbin[i]):
                if not (ixh == idata.nchan-1):
                    ixh += 1
                elif not (ixl == 0):
                    ixl -= 1
                else:
                    if j % 2 == 0: 
                        ixh -= 1
                    else:
                        ixl += 1

        # clip the data to cover only the frequencies of interest
        inu_TOPO = idata.nu_TOPO[ixl:ixh]
        inu_LSRK = idata.nu_LSRK[:,ixl:ixh]
        iv_LSRK = v_LSRK[:,ixl:ixh]
        inchan = inu_LSRK.shape[1]
        ivis = idata.vis[:,ixl:ixh,:]
        iwgt = idata.wgt[:,ixl:ixh,:]

        print(iv_LSRK.min(), iv_LSRK.max())

        # spectral binning
        bnchan = np.int(inchan / chbin[i])
        wt = iwgt.reshape((idata.npol, -1, chbin[i], idata.nvis))
        bvis = np.average(ivis.reshape((idata.npol, -1, chbin[i], idata.nvis)), 
                          weights=wt, axis=2)
        bwgt = np.sum(wt, axis=2)

        # channel censoring
        if vcensor is not None:
            # determine number of censoring zones
            ncens = len(vcensor)

            # approximate velocities of binned channels
            v_ = iv_LSRK[midstamp,:]
            v_bin = np.average(v_.reshape(-1, chbin[i]), axis=1)

            # identify which (binned) channels are censored (==False)
            cens_chans = np.ones(inchan, dtype='bool')
            for j in range(ncens):
                if sgn_v < 0:
                    vcens = (vcensor[j])[::-1]
                else: vcens = vcensor[j]
                ixl = np.abs(iv_LSRK[midstamp,:] - vcens[0]).argmin()
                ixh = np.abs(iv_LSRK[midstamp,:] - vcens[1]).argmin()
                cens_chans[ixl:ixh+1] = False
            cens_chans = np.all(cens_chans.reshape((-1, chbin[i])), axis=1)

            # set weights --> 0 in censored channels
            bwgt[:,cens_chans == False,:] = 0
           
        # pre-calculate the spectral covariance matrix and its inverse
        if chbin[i] == 2:
            di, odi = (5./16), (3./32)
        elif chbin[i] == 3:
            di, odi = (1./4), (1./24)
        elif chbin[i] == 4:
            di, odi = (13./64), (3./128)
        else:
            di, odi = 1, 0      # this is wrong, but maybe useful to test
        scov = di * np.eye(bnchan) + \
               odi * (np.eye(bnchan, k=-1) + np.eye(bnchan, k=1))
        scov_inv = np.linalg.inv(scov)

        # pre-calculate the log-likelihood normalization
        dterm = np.empty((idata.npol, idata.nvis))
        for ii in range(idata.nvis):
            for jj in range(idata.npol):
                sgn, lndet = np.linalg.slogdet(scov / bwgt[jj,:,ii])
                dterm[jj,ii] = sgn * lndet
        lnL0 = -0.5*(np.prod(bvis.shape) * np.log(2*np.pi) + np.sum(dterm))

        # package the data and add to the output dictionary
        out_dict[str(i)] = inf_dataset(idata.um, idata.vm, bvis, bwgt, 
                                       inu_TOPO, inu_LSRK, idata.tstamp, iwgt,
                                       scov, scov_inv, lnL0)

    # return the output dictionary
    return out_dict




def HDF_to_dataset(HDF_in, grp=''):

    # Open the HDF file
    _ = h5py.File(HDF_in+'.h5', "r")

    # Load the inputs into numpy arrays (convert visibilities to complex)
    _um, _vm = np.asarray(_[grp+'um']), np.asarray(_[grp+'vm'])
    _vis = np.asarray(_[grp+'vis_real']) + 1j * np.asarray(_[grp+'vis_imag'])
    _wgts, _stmp = np.asarray(_[grp+'weights']), np.asarray(_[grp+'tstamp_ID'])
    _TOPO, _LSRK = np.asarray(_[grp+'nu_TOPO']), np.asarray(_[grp+'nu_LSRK'])
    _.close()

    # Return a dataset object
    return dataset(_um, _vm, _vis, _wgts, _TOPO, _LSRK, _stmp)




def dataset_to_HDF(dataset_in, HDF_out, append=False, groupname=None):

    # Check output kwargs
    if append:
        if not os.path.exists(HDF_out+'.h5'):
            print('The HDF5 file you are trying to append does not exist.')
            return
        elif groupname == None:
            print('You need to specify a groupname to append in the HDF file.')
            return
        else:
            if groupname[-1] != '/': groupname += '/'

        outp = h5py.File(HDF_out+'.h5', "a")
    else:
        groupname = ''
        os.system('rm -rf '+HDF_out+'.h5')
        outp = h5py.File(HDF_out+'.h5', "w")

    # Populate the file
    outp.create_dataset(groupname+'um', data=dataset_in.um)
    outp.create_dataset(groupname+'vm', data=dataset_in.vm)
    outp.create_dataset(groupname+'vis_real', data=dataset_in.vis.real)
    outp.create_dataset(groupname+'vis_imag', data=dataset_in.vis.imag)
    outp.create_dataset(groupname+'weights', data=dataset_in.wgt)
    outp.create_dataset(groupname+'nu_TOPO', data=dataset_in.nu_TOPO)
    outp.create_dataset(groupname+'nu_LSRK', data=dataset_in.nu_LSRK)
    outp.create_dataset(groupname+'tstamp_ID', data=dataset_in.tstamp)
    outp.close()

    return





def format_data(cfg_file):

    # Format the data (+ time-average if desired) 
    os.system('rm -rf '+inp.casalogs_dir+'format_data.'+cfg_file+'.log')
    os.system('casa --nologger --logfile '+inp.casalogs_dir+ \
              'format_data.'+cfg_file+'.log '+ \
              '-c csalt/CASA_scripts/format_data.py configs/mconfig_'+cfg_file)
