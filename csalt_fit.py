import os, sys, time, importlib
import numpy as np
from csalt_data import *
from csalt_models import *
import emcee
from multiprocessing import Pool
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append('configs_modeling/')


### Ingest the configuration file (in configs_modeling/). 
if len(sys.argv) == 1:
    print('\nSpecify a configuration filename in configs_modeling/ as an '+ \
          'argument: e.g., \n')
    print('        python csalt_fit.py <cfg_file>\n')
    sys.exit()
else:
    # Read user-supplied argument
    cfg_file = sys.argv[1]

    # Strip the '.py' if they included it
    if cfg_file[-3:] == '.py':
        cfg_file = cfg_file[:-3]

    # Load the configuration file contents
    try:
        inp = importlib.import_module('mconfig_'+cfg_file)
    except:
        print('\nThere is a problem with the configuration file:')
        print('trying to use configs_modeling/mconfig_'+cfg_file+'.py\n')
        sys.exit()


# package data for inference purposes
data_ = fitdata(inp, vra=inp.vra_fit, vcensor=inp.vra_cens)


# set initial parameter guesses
p_lo, p_hi = inp.init_[:,0], inp.init_[:,1]
ndim, nwalk = len(p_lo), inp.nwalkers
p0 = [np.random.uniform(p_lo, p_hi, ndim) for iw in range(nwalk)]


# acquire gcfs and corr caches from preliminary model calculations
for EB in range(data_['nobs']):

    # set fixed parameters
    fixed = inp.nu_rest, inp.FOV[EB], inp.Npix[EB], inp.dist

    # initial model calculations
    _mvis, gcf, corr = vismodel_def(p0[0], fixed, data_[str(EB)],
                                    return_holders=True)

    # add gcf, corr caches into data dictionary, indexed by EB
    data_['gcf'+str(EB)] = gcf
    data_['corr'+str(EB)] = corr



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
        pri_type, pri_pars = inp.priors_['types'][ip], inp.priors_['pars'][ip]
        if pri_type == 'uniform':
            ptheta[ip] = lnp_uniform(theta[ip], pri_pars)
        elif pri_type == 'normal':
            ptheta[ip] = lnp_normal(theta[ip], pri_pars)
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
    for EB in range(data_['nobs']):

        # get the inference dataset
        dat = data_[str(EB)]

        # calculate model visibilities
        fixed = inp.nu_rest, inp.FOV[EB], inp.Npix[EB], inp.dist       
        mvis = vismodel_iter(theta, fixed, dat, 
                             data_['gcf'+str(EB)], data_['corr'+str(EB)]) 

        # spectrally bin the model
        wt = dat.iwgt.reshape((dat.npol, -1, dat.chbin, dat.nvis))
        mvis_b = np.average(mvis.reshape((dat.npol, -1, dat.chbin, dat.nvis)),
                            weights=wt, axis=2)

        # compute the residuals (stack both pols)
        resid = np.hstack(np.absolute(dat.vis - mvis_b))
        var = np.hstack(dat.wgt)

        # compute the log-likelihood
        lnL = -0.5 * np.tensordot(resid, np.dot(dat.inv_cov, var * resid))

    # return the log-posterior and log-prior
    return lnL + dat.lnL0 + lnT, lnT




if not os.path.exists('fitting'):
    os.mkdir('fitting')
    os.mkdir('fitting/posteriors/')

# Configure emcee backend
post_filename = 'fitting/posteriors/'+inp.basename+inp._ext+inp._fitnote+'.h5'
os.system('rm -rf '+post_filename)
backend = emcee.backends.HDFBackend(post_filename)
backend.reset(nwalk, ndim)

# run the sampler
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalk, ndim, lnprob, pool=pool, 
                                    backend=backend)
    t0 = time.time()
    sampler.run_mcmc(p0, inp.max_steps, progress=True)
t1 = time.time()

print(' ')
print(' ')
print('This run took %.2f hours' % ((t1 - t0) / 3600))
