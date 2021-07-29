import os, sys, importlib
import numpy as np
import const as const

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

    def __init__(self, uu, vv, v_LSRK, tstamp_ID, vis_bin, wgt_bin, 
                 M_bin, invM_bin, lnL0):

        # spatial frequencies in lambda units
        self.uu = uu
        self.vv = vv

        # timestamps (indices)
        self.tstamp = tstamp_ID
        self.nstamps = len(np.unique(tstamp_ID))

        # **unbinned** LSRK velocity grid, model input velocities
        self.v_LSRK = v_LSRK
        self.v_calc = v_LSRK[np.int(self.nstamps / 2),:]

        # **binned** data visibilities, weights
        self.vis = vis_bin
        self.wgt = wgt_bin
        self.npol, self.nchan, self.nvis = self.vis.shape
        self.chbin = np.int(len(self.v_calc) / self.vis.shape[1])

        # likelihood-related quantities
        self.cov = M_bin
        self.inv_cov = invM_bin
        self.lnL0 = lnL0




# Data parsing to generate inputs for likelihood function
def fitdata(file_prefix, vra=None, vcensor=None):

    # Load the configuration file
    inp = importlib.import_module('mconfig_'+file_prefix)

    # Load the metadata and initialize the output dictionary
    data_dict = np.load(inp.dataname+'.npy', allow_pickle=True).item()
    nobs = data_dict['nobs']
    out_dict = {'nobs': nobs}

    # Loop through each EB
    for i in range(nobs):

        # load the data into a dataset object
        d_ = np.load(inp.dataname+'_EB'+str(i)+'.npz')
        idata = dataset(d_['um'], d_['vm'], d_['data'], d_['weights'],
                        d_['nu_TOPO'], d_['nu_LSRK'], d_['tstamp_ID'])

        # if necessary, distribute weights across spectrum
        if not idata.wgt.shape == idata.vis.shape:
            idata.wgt = np.tile(idata.wgt, (idata.nchan, 1, 1))
            idata.wgt = np.rollaxis(idata.wgt, 1, 0)

        # convert the LSRK frequency grid to a velocity grid
        v_LSRK = const.c_ * (1 - idata.nu_LSRK / inp.restfreq)

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
        if ((ixh - ixl + (ixh - ixl) % inp.chbin[i]) < idata.nchan):
            for j in range((ixh - ixl) % inp.chbin[i]):
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
        iv_LSRK = v_LSRK[:,ixl:ixh]
        inu_LSRK = idata.nu_LSRK[:,ixl:ixh]
        inchan = inu_LSRK.shape[1]
        ivis = idata.vis[:,ixl:ixh,:]
        iwgt = idata.wgt[:,ixl:ixh,:]

        # convert spatial frequencies to lambda units
        iu = idata.um * np.mean(inu_LSRK[np.int(idata.nstamps/2),:]) / const.c_
        iv = idata.vm * np.mean(inu_LSRK[np.int(idata.nstamps/2),:]) / const.c_

        # spectral binning
        bnchan = np.int(inchan / inp.chbin[i])
        wt = iwgt.reshape((idata.npol, -1, inp.chbin[i], idata.nvis))
        bvis = np.average(ivis.reshape((idata.npol, -1, inp.chbin[i], 
                                        idata.nvis)), weights=wt, axis=2)
        bwgt = np.sum(wt, axis=2)

        # channel censoring
        if vcensor is not None:
            # determine number of censoring zones
            ncens = len(vcensor)

            # approximate velocities of binned channels
            v_ = iv_LSRK[midstamp,:]
            v_bin = np.average(v_.reshape(-1, inp.chbin[i]), axis=1)

            # identify which (binned) channels are censored (==False)
            cens_chans = np.ones(inchan, dtype='bool')
            for j in range(ncens):
                if sgn_v < 0:
                    vcens = (vcensor[j])[::-1]
                else: vcens = vcensor[j]
                ixl = np.abs(iv_LSRK[midstamp,:] - vcens[0]).argmin()
                ixh = np.abs(iv_LSRK[midstamp,:] - vcens[1]).argmin()
                cens_chans[ixl:ixh+1] = False
            cens_chans = np.all(cens_chans.reshape((-1, inp.chbin[i])), axis=1)

            # set weights --> 0 in censored channels
            bwgt[:,cens_chans == False,:] = 0
           
        # pre-calculate the spectral covariance matrix and its inverse
        if inp.chbin[i] == 2:
            di, odi = (5./16), (3./32)
        elif inp.chbin[i] == 3:
            di, odi = (1./4), (1./24)
        elif inp.chbin[i] == 4:
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
        out_dict[str(i)] = inf_dataset(iu, iv, iv_LSRK, idata.tstamp, 
                                       bvis, bwgt, scov, scov_inv, lnL0)

    # return the output dictionary
    return out_dict
