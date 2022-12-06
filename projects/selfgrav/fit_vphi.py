import os, sys, time, importlib
sys.path.append('../../')
sys.path.append('../../configs/')
import numpy as np
import scipy.constants as sc
import emcee
import corner
from model_vexact import model_vphi
from multiprocessing import Pool
os.environ["OMP_NUM_THREADS"] = "1"



### USER INPUTS
# naming / labeling
mdl = 'taper2hi'
msetup = '_raw'
post_file = 'posteriors/'+mdl+msetup+'_offsetSig.posteriors.h5'

# run emcee?
do_fit = True
append_fit = False
fit_r = [0, 350]



### SETUPS
# load the "data" object
dat = np.load('data/surf_vphi_'+mdl+msetup+'.npz')

# extract the inferred velocity profile
re, ve, e_ve = dat['re'], dat['ve'], dat['eve']

# assign the subset to fit (and not)
ok = np.logical_and(re >= fit_r[0], re <= fit_r[1])
f_r, f_v, f_e = re[ok], ve[ok], e_ve[ok]
n_r, n_v, n_e = re[~ok], ve[~ok], e_ve[~ok]



### POSTERIOR
# log-likelihood function
def lnlikelihood(pars, r, v, err):

    # model calculation
    mdl = model_vphi(r, pars)   

    # variance
    sigma2 = err**2 + mdl**2 * np.exp(2 * pars[-1])

    # return log-likelihood
    return -0.5 * np.sum((v - mdl)** 2 / sigma2 + np.log(sigma2))

# log-prior function
def lnprior(pars):

    # uniform for Mstar and log(f)
    if 0.0 < pars[0] < 2.0 and 0.0 < pars[1] < 1.0 and -10.0 < pars[2] < 1.0:
        return 0.0
    return -np.inf

# log-posterior function
def lnprob(pars, r, v, err):

    # check prior
    lp = lnprior(pars)
    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlikelihood(pars, r, v, err)
    


### SAMPLE POSTERIORS (if requested)
if do_fit:

    # definitions
    ndim, nwalk, nstep, ninit, nthread = 3, 64, 2000, 200, 6

    # new run
    if not append_fit:

        # initial initializing ;)
        p0 = np.array([0.5, 0.01, -1.0]) + 1e-1 * np.random.randn(nwalk, ndim)

        # initial sampling
        with Pool(processes=nthread) as pool:
            isampler = emcee.EnsembleSampler(nwalk, ndim, lnprob, pool=pool,
                                             args=(f_r, f_v, f_e))
            isampler.run_mcmc(p0, ninit, progress=True)

        # reset initialization for stray walkers!
        isamples = isampler.get_chain()
        lop0 = np.quantile(isamples[-1,:,:], 0.25, axis=0)
        hip0 = np.quantile(isamples[-1,:,:], 0.75, axis=0)
        p00 = [np.random.uniform(lop0, hip0, ndim) for iw in range(nwalk)]
        print('\nChains now properly initialized...\n')

        # prepare the backend file for the full run
        os.system('rm -rf '+post_file)
        backend = emcee.backends.HDFBackend(post_file)
        backend.reset(nwalk, ndim)

        # run the MCMC
        with Pool(processes=nthread) as pool:
            sampler = emcee.EnsembleSampler(nwalk, ndim, lnprob, pool=pool,
                                            args=(f_r, f_v, f_e), 
                                            backend=backend)
            sampler.run_mcmc(p00, nstep, progress=True)

    # append to an existing run
    else:

        # configure backend properly
        new_backend = emcee.backends.HDFBackend(post_file)
        print('Initial size: {0}'.format(new_backend.iteration))

        # run the MCMC
        with Pool(processes=nthread) as pool:
            new_sampler = emcee.EnsembleSampler(nwalk, ndim, lnprob, pool=pool,
                                                args=(f_r, f_v, f_e),
                                                backend=new_backend)
            new_sampler.run_mcmc(None, nstep, progress=True)
        print('Final size: {0}'.format(new_backend.iteration))
