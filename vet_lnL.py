import os, sys, time, importlib
import numpy as np
from csalt_data import *
from csalt_models import *


# configuration file
file_prefix = 'simp3-nmm'
inp = importlib.import_module('mconfig_'+file_prefix)

# package data for inference purposes
data_ = fitdata('simp3-nmm', vra=[3000, 8000])

# set initial parameter guesses
theta = np.array([inp.incl, inp.PA, inp.mstar, inp.r_l, inp.z0, inp.psi,
                  inp.Tb0, inp.q, inp.Tback, inp.dV0,
                  inp.vsys, inp.xoff, inp.yoff])

# set fixed parameters
theta_fixed = inp.restfreq, inp.FOV, inp.Npix, inp.dist, inp.rmax

# prior setups
pri_type = np.array(['uniform', 'uniform', 'uniform', 'uniform', 'uniform',
                     'uniform', 'uniform', 'uniform', 'uniform', 'uniform',
                     'uniform', 'uniform', 'uniform'])
pri_pars = [ [0., 90.], [0., 360.], [0., 5.], [10., 0.5 * inp.dist * inp.FOV],
             [0., 10.], [0., 1.5], [5., 1000.], [0., 2.], [5., 50.], 
             [0., 1000.], [4200, 6200], [-0.2, 0.2], [-0.2, 0.2] ]


# acquire gcfs and corr caches from preliminary model calculations
for i in range(data_['nobs']):

    # initial model calculations
    _mvis, gcf, corr = vismodel_def(theta, theta_fixed, data_[str(i)],
                                    return_holders=True)

    # add gcf, corr caches into data dictionary, indexed by EB
    data_['gcf'+str(i)] = gcf
    data_['corr'+str(i)] = corr



# prior functions
def lnp_uniform(theta, ppars):
    if ((theta >= ppars[0]) and (theta <= ppars[1])):
        return 0
    else:
        return -np.inf

def lnp_normal(theta, ppars):
    return -0.5 * (theta - ppars[0])**2 / ppars[1]**2


# log-prior calculator
def lnprior(theta):

    # initialize
    ptheta = np.empty_like(theta)

    # loop through parameters
    for ip in range(len(theta)):
        if pri_type[ip] == 'uniform':
            ptheta[ip] = lnp_uniform(theta[ip], pri_pars[ip])
        elif pri_type[ip] == 'normal':
            ptheta[ip] = lnp_normal(theta[ip], pri_pars[ip])
        else:
            ptheta[ip] = -np.inf

    return ptheta

    
# log-posterior calculator
def lnprob(theta):

    # compute the log-prior and return if problematic
    lnT = np.sum(lnprior(theta)) * data_['nobs']
    if lnT == -np.inf:
        return -np.inf, -np.inf

    # loop through observations to compute the log-likelihood
    lnpost = 0
    for i in range(data_['nobs']):

        # get the inference dataset
        dat = data_[str(i)]

        # calculate model visibilities
        mvis = vismodel_iter(theta, theta_fixed, dat, 
                             data_['gcf'+str(i)], data_['corr'+str(i)]) 

        # spectrally bin the model
        wt = dat.iwgt.reshape((dat.npol, -1, dat.chbin, dat.nvis))
        mvis_b = np.average(mvis.reshape((dat.npol, -1, dat.chbin, dat.nvis)),
                            weights=wt, axis=2)

        # compute the residuals (stack both pols)
        resid = np.hstack(np.absolute(dat.vis - mvis_b))
        var = np.hstack(dat.wgt)

        # compute the log-likelihood
        lnL = -0.5 * np.tensordot(resid, np.dot(dat.inv_cov, var * resid))

    # return the log-posterior and log-prior (NEED TO NORMALIZE LNL STILL)
    return lnL + lnT, lnT


print(lnprob(theta))
