import os, sys, time
import numpy as np
from csalt.data import *
from csalt.models import *
from priors import *
import emcee
from schwimmbad import MPIPool
os.environ["OMP_NUM_THREADS"] = "1"


# I/O
datafile = 'storage/data/fiducial_snap/fiducial_snap_noisy.DATA'
post_dir = ''
postfile = 'mpi_test'
append = True
initialize = False

# data selections 
vra_fit = [3e3, 7e3]	# (LSRK velocities from -1 to +11 km/s)
vcensor = None

# inference setups
nwalk = 75
ninits = 50
nsteps = 100
prune_lo, prune_hi = 0.25, 0.75

# fixed parameters
nu_rest = 230.538e9	# rest frequency of line (Hz)
FOV = 6.375		# field of view (arcsec)
Npix = 256		# pixels per side
dist = 150.		# distance (pc)
cfg_dict = {}		# empty dict for CSALT models
fixed = nu_rest, FOV, Npix, dist, cfg_dict






### Log-posterior function
def lnprob(theta):	

    # compute the log-prior and return if problematic
    lnT = np.sum(logprior(theta)) * data_['nobs']
    if lnT == -np.inf:
        return -np.inf, -np.inf

    # loop through observations to compute the log-likelihood
    lnL = 0
    for EB in range(data_['nobs']):

        # get the inference dataset
        dat = data_[str(EB)]

        # calculate model visibilities
        mvis = vismodel_iter(theta, fixed_, dat,
                             data_['gcf'+str(EB)], data_['corr'+str(EB)])

        # spectrally bin the model
        wt = dat.iwgt.reshape((dat.npol, -1, dat.chbin, dat.nvis))
        mvis_b = np.average(mvis.reshape((dat.npol, -1, dat.chbin,
                                          dat.nvis)), weights=wt, axis=2)

        # compute the residuals (stack both pols)
        resid = np.hstack(np.absolute(dat.vis - mvis_b))
        var = np.hstack(dat.wgt)

        # compute the log-likelihood
        lnL += -0.5 * np.tensordot(resid, np.dot(dat.inv_cov, var * resid))

    # return the log-posterior and log-prior
    return lnL + dat.lnL0 + lnT, lnT



# load the data
global data_
data_ = fitdata(datafile, vra=vra_fit, vcensor=None, nu_rest=nu_rest, chbin=2)

# assign the fixed parameters
global fixed_
fixed_ = fixed



### Initialize the walkers
ndim = len(pri_pars)

# start with random draws from the priors
if initialize:
    p0 = np.empty((nwalk, ndim))
    for ix in range(ndim):
        _ = [str(pri_pars[ix][ip])+', ' for ip in range(len(pri_pars[ix]))]
        cmd = 'np.random.'+pri_types[ix]+'('+"".join(_)+str(nwalk)+')'
        p0[:,ix] = eval(cmd)
    nst = ninits

# or prune an initialization run
else:
    reader = emcee.backends.HDFBackend(post_dir+postfile+'.init.h5')
    isamples = reader.get_chain(discard=0, flat=False)
    lop0 = np.quantile(isamples[-1, :, :], prune_lo, axis=0)
    hip0 = np.quantile(isamples[-1, :, :], prune_hi, axis=0)
    p0 = [np.random.uniform(lop0, hip0, ndim) for iw in range(nwalk)]
    nst = nsteps



### Acquire gcfs and corr caches from preliminary model calculations
for EB in range(data_['nobs']):

    # initial model calculations
    _mvis, gcf, corr = vismodel_def(p0[0], fixed_, data_[str(EB)],
                                    return_holders=True)

    # add gcf, corr caches into data dictionary, indexed by EB
    data_['gcf'+str(EB)] = gcf
    data_['corr'+str(EB)] = corr




### Sample the posteriors
# a new run / posteriors backend
if not append:
    # remove any old posteriors backend file
    if initialize:
        post_file = post_dir+postfile+'.init.h5'
    else:
        post_file = post_dir+postfile+'.h5'
    if os.path.exists(post_file):
        os.system('rm -rf '+post_file)

    # MCMC
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        # configure backend for recording posterior samples
        backend = emcee.backends.HDFBackend(post_file)
        backend.reset(nwalk, ndim)
        
        # configure sampler and run
        sampler = emcee.EnsembleSampler(nwalk, ndim, lnprob, 
                                        pool=pool, backend=backend)
        t0 = time.time()
        sampler.run_mcmc(p0, nst)
    t1 = time.time()
# appending to an existing run / posteriors backend
else:
    # proceed to actual MCMC directly
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        # prepare to append to backend
        new_backend = emcee.backends.HDFBackend(post_dir+postfile+'.h5')
        print("Initial size: {0}".format(new_backend.iteration))

        # configure sampler and run
        new_sampler = emcee.EnsembleSampler(nwalk, ndim, lnprob,
                                            pool=pool, backend=new_backend)
        t0 = time.time()
        new_sampler.run_mcmc(None, nst)
    t1 = time.time()


print(' ')
print(' ')
print('MPI run took %.2f minutes' % ((t1 - t0) / 60))
print(' ')
print(' ')

#mpi_time = !mpiexec -n {ncpu} python script.py
