import os, sys, time, importlib
import numpy as np
from csalt.data import *
from csalt.models import *
import emcee
import corner
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from multiprocessing import Pool
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append('configs/')


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
        pri_type = inp.priors_['types'][ip]
        pri_pars = inp.priors_['pars'][ip]
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
        fixed = inp.nu_rest, inp.FOV[EB], inp.Npix[EB], inp.dist, \
                inp.cfg_dict
        mvis = vismodel_iter(theta, fixed, dat,
                             data_['gcf'+str(EB)], data_['corr'+str(EB)])

        # spectrally bin the model
        wt = dat.iwgt.reshape((dat.npol, -1, dat.chbin, dat.nvis))
        mvis_b = np.average(mvis.reshape((dat.npol, -1, dat.chbin,
                                          dat.nvis)), weights=wt, axis=2)

        # compute the residuals (stack both pols)
        resid = np.hstack(np.absolute(dat.vis - mvis_b))
        var = np.hstack(dat.wgt)

        # compute the log-likelihood
        lnL = -0.5 * np.tensordot(resid, np.dot(dat.inv_cov, var * resid))

    # return the log-posterior and log-prior
    return lnL + dat.lnL0 + lnT, lnT




def run_emcee(cfg_file, nsteps=1000, append=False):

    # Load the configuration file contents
    try:
        global inp
        inp = importlib.import_module('mconfig_'+cfg_file)
    except:
        print('\nThere is a problem with the configuration file:')
        print('trying to use configs/mconfig_'+cfg_file+'.py\n')
        sys.exit()


    # package data for inference purposes
    global data_
    data_ = fitdata(inp, vra=inp.vra_fit, vcensor=inp.vra_cens)


    # initialize parameters
    p_lo, p_hi = inp.init_[:,0], inp.init_[:,1]
    ndim, nwalk = len(p_lo), inp.nwalkers
    p0 = [np.random.uniform(p_lo, p_hi, ndim) for iw in range(nwalk)]


    # acquire gcfs and corr caches from preliminary model calculations
    for EB in range(data_['nobs']):

        # set fixed parameters
        fixed = inp.nu_rest, inp.FOV[EB], inp.Npix[EB], inp.dist, inp.cfg_dict

        # initial model calculations
        _mvis, gcf, corr = vismodel_def(p0[0], fixed, data_[str(EB)],
                                        return_holders=True)

        # add gcf, corr caches into data dictionary, indexed by EB
        data_['gcf'+str(EB)] = gcf
        data_['corr'+str(EB)] = corr



    # Configure backend for recording posterior samples
    fitout = inp.fitname+inp.basename+inp._ext+inp._fitnote+'/'
    if not os.path.exists(fitout):
        if not os.path.exists(inp.fitname):
            os.mkdir(inp.fitname)
        os.mkdir(fitout)
        os.mkdir(fitout+'posteriors/')
    post_file = fitout+'posteriors/emcee_samples.h5'
    if not append:
        os.system('rm -rf '+post_file)
        backend = emcee.backends.HDFBackend(post_file)
        backend.reset(nwalk, ndim)
        
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalk, ndim, lnprob, pool=pool,
                                            backend=backend)
            t0 = time.time()
            sampler.run_mcmc(p0, nsteps, progress=True)
        t1 = time.time()
    else:
        new_backend = emcee.backends.HDFBackend(post_file)
        print("Initial size: {0}".format(new_backend.iteration))

        with Pool() as pool:
            new_sampler = emcee.EnsembleSampler(nwalk, ndim, lnprob, pool=pool,
                                                backend=new_backend)
            t0 = time.time()
            new_sampler.run_mcmc(None, nsteps, progress=True)
        t1 = time.time()
        print("Final size: {0}".format(new_backend.iteration))

    print(' ')
    print(' ')
    print('This run took %.2f hours' % ((t1 - t0) / 3600))

    return



def post_summary(p, prec=0.1, mu='peak', CIlevs=[84.135, 15.865, 50.]):

    # calculate percentiles as designated
    CI_p = np.percentile(p, CIlevs)

    # find peak of posterior
    if (mu == 'peak'):
        kde_p = stats.gaussian_kde(p)
        ndisc = np.int(np.round((CI_p[0] - CI_p[1]) / prec))
        x_p = np.linspace(CI_p[1], CI_p[0], ndisc)
        pk_p = x_p[np.argmax(kde_p.evaluate(x_p))]
    else:
        pk_p = np.percentile(p, 50.)

    # return the peak and upper, lower 1-sigma
    return (pk_p, CI_p[0]-pk_p, pk_p-CI_p[1], CI_p[2])



def post_analysis(cfg_file, burnin=0, autocorr=False, Ntau=200, 
                  corner_plot=True):

    # Load the configuration file contents
    try:
        inp = importlib.import_module('mconfig_'+cfg_file)
    except:
        print('\nThere is a problem with the configuration file:')
        print('trying to use configs/mconfig_'+cfg_file+'.py\n')
        sys.exit()

    # load the emcee backend file
    fitout = inp.fitname+inp.basename+inp._ext+inp._fitnote+'/'
    reader = emcee.backends.HDFBackend(fitout+'posteriors/emcee_samples.h5')

    # parse the samples
    all_samples = reader.get_chain(discard=0, flat=False)
    samples = reader.get_chain(discard=burnin, flat=False)
    samples_ = reader.get_chain(discard=burnin, flat=True)
    logpost_samples = reader.get_log_prob(discard=burnin, flat=False)
    logprior_samples = reader.get_blobs(discard=burnin, flat=False)
    nsteps, nwalk, ndim = samples.shape

    # set parameter labels, truths (NOT HARDCODE!)
    lbls = ['incl', 'PA', 'M', 'r_l', 'z0', 'psi', 'Tb0', 'q', 'Tback', 'dV0',
            'tau0', 'p', 'vsys', 'dx', 'dy']
    theta = inp.pars


    # Plot the integrated autocorrelation time every Ntau steps
    if autocorr:
        Nmax = all_samples.shape[0]
        if (Nmax > Ntau):
            tau_ix = np.empty(np.int(Nmax / Ntau))
            ix = np.empty(np.int(Nmax / Ntau))
            for i in range(len(tau_ix)):
                nn = (i + 1) * Ntau
                ix[i] = nn
                tau = emcee.autocorr.integrated_time(all_samples[:nn,:,:], 
                                                     tol=0)
                tau_ix[i] = np.mean(tau)

        fig = plt.figure()
        plt.plot(ix, tau_ix, '-o')
        plt.xlabel('steps')
        plt.ylabel('autocorr time (steps)')
        plt.xlim([0, Nmax])
        plt.ylim([0, tau_ix.max() + 0.1 * (tau_ix.max() - tau_ix.min())])
        fig.savefig(fitout+'autocorr.png')
        fig.clf()


    # Plot the traces
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(3, 6)

    # log-likelihood
    ax = fig.add_subplot(gs[0,0])
    for iw in range(nwalk):
        ax.plot(np.arange(nsteps), 
                logpost_samples[:,iw] - logprior_samples[:,iw], 
                color='k', alpha=0.03)
    ax.set_xlim([0, nsteps])
    ax.tick_params(which='both', labelsize=6)
    ax.set_ylabel('log likelihood', fontsize=6)
    ax.set_xticklabels([])
    ax.text(0.95, 0.05, 'ln L', fontsize=12, ha='right', color='purple',
            transform=ax.transAxes)

    # log-prior
    ax = fig.add_subplot(gs[0,1])
    for iw in range(nwalk):
        ax.plot(np.arange(nsteps), logprior_samples[:, iw], 
                color='k', alpha=0.03)
    ax.set_xlim([0, nsteps])
    ax.set_ylim([np.min(logprior_samples[:, iw]) - 0.05,
                 np.max(logprior_samples[:, iw]) + 0.05])
    ax.tick_params(which='both', labelsize=6)
    ax.set_ylabel('log prior', fontsize=6)
    ax.set_xticklabels([])
    ax.text(0.95, 0.05, 'ln prior', fontsize=12, ha='right', color='purple',
            transform=ax.transAxes)

    # now cycle through parameters
    ax_ixl = [np.floor_divide(idim, 6) for idim in np.arange(2, ndim+2)]
    ax_ixh = [(idim % 6) for idim in np.arange(2, ndim+2)]
    for idim in range(ndim):
        ax = fig.add_subplot(gs[ax_ixl[idim], ax_ixh[idim]])
        for iw in range(nwalk):
            ax.plot(np.arange(nsteps), samples[:, iw, idim], 
                    color='k', alpha=0.03)
        ax.plot([0, nsteps], [theta[idim], theta[idim]], '--C1', lw=1.5)
        ax.set_xlim([0, nsteps])
        ax.tick_params(which='both', labelsize=6)
        ax.set_ylabel(lbls[idim], fontsize=6)
        if idim != 10:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('steps', fontsize=6)
        ax.text(0.95, 0.05, lbls[idim], fontsize=12, ha='right', color='purple',
                transform=ax.transAxes)

    fig.subplots_adjust(wspace=0.20, hspace=0.05)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.99)
    fig.savefig(fitout+'traces.png')
    fig.clf()


    # corner plot
    if corner_plot:
        levs = 1. - np.exp(-0.5 * (np.arange(3) + 1)**2)
        flat_chain = samples.reshape(-1, ndim)
        fig = corner.corner(flat_chain, plot_datapoints=False, levels=levs,
                            labels=lbls, truths=theta)
        fig.savefig(fitout+'corner.png')
        fig.clf()


    # Parameter inferences (1-D marginalized)
    print(' ')
    prec = [0.01, 0.01, 0.001, 0.1, 0.01, 0.01, 0.1, 0.001, 0.1, 0.1, 0.01, 
            0.01, 0.1, 0.0001, 0.0001]
    for idim in range(ndim):
        fmt = '{:.'+str(np.abs(np.int(np.log10(prec[idim]))))+'f}'
        pk, hi, lo, med = post_summary(samples_[:,idim], prec=prec[idim])
        print((lbls[idim] + ' = '+fmt+' +'+fmt+' / -'+fmt).format(pk, hi, lo))
    print(' ')




