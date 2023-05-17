import os
import sys
import importlib
import warnings
import numpy as np
from scipy import linalg
import scipy.constants as sc
from csalt.data2 import *


"""
The infer class for modeling visibility spectra.
"""
class infer:

    def __init__(self, prescription, path=None, quiet=True):

        if quiet:
            warnings.filterwarnings("ignore")

        self.prescription = prescription
        self.path = path


    """ Parse and package data for inference """
    def fitdata(self, msfile, 
                vra=None, vcensor=None, nu_rest=230.538e9, chbin=1, 
                well_cond=300):

        # Load the data from the MS file into a dictionary
        data_dict = read_MS(msfile)

        # If chbin is a scalar, distribute it over the Nobs executions
        if np.isscalar(chbin):
            chbin = chbin * np.ones(data_dict['Nobs'], dtype=int)
        else:
            if isinstance(chbin, list): 
                chbin = np.asarray(chbin)

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
            v_LSRK = sc.c * (1 - data.nu_LSRK / nu_rest)

            # Fix direction of desired velocity bounds
            if vra is None: vra = [-1e5, 1e5]
            dv, dvra = np.diff(v_LSRK, axis=1), np.diff(vra)
            if np.logical_or(np.logical_and(np.all(dv < 0), np.all(dvra > 0)),
                             np.logical_and(np.all(dv < 0), np.all(dvra < 0))):
                vra = vra[::-1]
            sgn_v = np.sign(np.diff(vra)[0])

            # Find where to clip to lie within the desired velocity bounds
            midstamp = int(data.nstamps / 2)
            ixl = np.abs(v_LSRK[midstamp,:] - vra[0]).argmin()
            ixh = np.abs(v_LSRK[midstamp,:] - vra[1]).argmin()

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
            inchan = data.nu_LSRK.shape[1]
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
                print('SCOV inverted with direct calculation.')
                scov_inv = linalg.inv(scov)

            # See if you can use Cholesky factorization
            else:
                chol = linalg.cholesky(scov)
                if np.linalg.cond(chol) <= well_cond:
                    print('SCOV inverted with Cholesky factorization')
                    scov_inv = np.dot(linalg.inv(chol), linalg.inv(chol.T))
                
                # Otherwise use SVD
                else:
                    print('SCOV inverted with singular value decomposition')
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

        # Return the output dictionary
        return out_dict
