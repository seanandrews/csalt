import os, sys, importlib
import numpy as np
import const as const

# General visibility dataset class
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


# Fitting class
class fitdata:

    def __init__(self, file_prefix, vra, vcensor):

        # Load the configuration file
        inp = importlib.import_module('mconfig_'+file_prefix)

        # Load the metadata dictionary
        data_dict = np.load(inp.dataname+'.npy', allow_pickle=True).item()
        self.nobs = data_dict['nobs']
        self.chbin = inp.chbin

        # Loop through each EB
        for i in range(self.nobs):

            # load the data into a dataset object
            d_ = np.load(inp.dataname+'_EB'+str(i)+'.npz')
            idata = dataset(d_['um'], d_['vm'], d_['data'], d_['weights'],
                            d_['nu_TOPO'], d_['nu_LSRK'], d_['tstamp_ID'])

            # convert the LSRK frequency grid to a velocity grid
            v_LSRK = const.c_ * (1 - idata.nu_LSRK / inp.restfreq)

            # fix direction of desired velocity bounds, based on data format
            dvi, dvra = np.diff(v_LSRK, axis=1), np.diff(vra)
            if np.logical_or(np.logical_and(np.all(dvi<0), np.all(dvra>0)),
                             np.logical_and(np.all(dvi>0), np.all(dvra<0))): 
                vra = vra[::-1]
            sgn_v = np.sign(np.diff(vra)[0])

            # clip the data to lie within the desired velocity bounds
            if (sgn_v < 0):
                
            print(sgn_v)
            sys.exit() 
