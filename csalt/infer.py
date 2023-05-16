import os
import sys
import importlib
import warnings
import numpy as np
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
    def fitdata(msfile, vra=None, vcensor=None, nu_rest=230.538e9, chbin=1):

        # Load the data from the MS file
        data_dict = read_MS(msfile)

        # If chbin is a scalar, distribute it over the Nobs executions
        if np.isscalar(chbin):
            chbin = chbin * np.ones(data_dict['Nobs'], dtype=np.int)
        else:
            if isinstance(chbin, list): 
                chbin = np.asarray(chbin)

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

            # reconcile channel set to be evenly divisible by binning factor
            if ((ixh - ixl + (ixh - ixl) % chbin[i]) < data.nchan):
                for j in range((ixh - ixl) % chbin[i]):
                    if not (ixh == data.nchan - 1):
                        ixh += 1
                    elif not (ixl == 0):
                        ixl -= 1
                    else:
                        if j % 2 == 0:
                            ixh -= 1
                        else:
                            ixl += 1

            # clip the data to cover only the frequencies of interest
            inu_TOPO = data.nu_TOPO[ixl:ixh]
            inu_LSRK = data.nu_LSRK[:,ixl:ixh]
            iv_LSRK = v_LSRK[:,ixl:ixh]
            inchan = nu_LSRK.shape[1]
            ivis = data.vis[:,ixl:ixh,:]
            iwgt = data.wgt[:,ixl:ixh,:]



            
        


