"""
call this like

python analyze_emcee.py <filename> <burnin>

"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import emcee
import corner
from post_summary import post_summary


# load the emcee backend file
fname = 'posteriors/'+sys.argv[1]+'.h5'
reader = emcee.backends.HDFBackend(fname)

# parse the samples
all_samples = reader.get_chain(discard=0, flat=False)
samples = reader.get_chain(discard=np.int(sys.argv[2]), flat=False)
samples_ = reader.get_chain(discard=np.int(sys.argv[2]), flat=True)
logpost_samples = reader.get_log_prob(discard=np.int(sys.argv[2]), flat=False)
logprior_samples = reader.get_blobs(discard=np.int(sys.argv[2]), flat=False)
nsteps, nwalk, ndim = samples.shape

# set parameter labels, truths (NOT HARDCODE!)
lbls = ['incl', 'PA', 'M', 'r_l', 'z0', 'psi', 'Tb0', 'q', 'Tback', 'dV0', 
        'vsys', 'dx', 'dy']
theta = [40, 130, 0.7, 200, 2.3, 1.0, 205., 0.5, 20., 347.7, 5200, 0., 0.]


# Plot the integrated autocorrelation time every Ntau steps
#Ntau = 200
#Nmax = all_samples.shape[0]
#if (Nmax > Ntau):
#    tau_ix = np.empty(np.int(Nmax / Ntau))
#    ix = np.empty(np.int(Nmax / Ntau))
#    for i in range(len(tau_ix)):
#        nn = (i + 1) * Ntau
#        ix[i] = nn
#        tau = emcee.autocorr.integrated_time(all_samples[:nn,:,:], tol=0)
#        tau_ix[i] = np.mean(tau)
#
#    fig = plt.figure()
#    plt.plot(ix, tau_ix, '-o')
#    plt.xlabel('steps')
#    plt.ylabel('autocorr time (steps)')
#    plt.xlim([0, Nmax])
#    plt.ylim([0, tau_ix.max() + 0.1 * (tau_ix.max() - tau_ix.min())])
#    fig.savefig('mcmc_analysis/'+sys.argv[1]+'.autocorr.png')
#    fig.clf()


# Plot the traces
fig = plt.figure(figsize=(15, 6))
gs = gridspec.GridSpec(3, 5)

# log-likelihood
ax = fig.add_subplot(gs[0,0])
for iw in range(nwalk):
    ax.plot(np.arange(nsteps), logpost_samples[:,iw] - logprior_samples[:,iw], 
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
    ax.plot(np.arange(nsteps), logprior_samples[:, iw], color='k', alpha=0.03)
    ax.set_xlim([0, nsteps])
    ax.set_ylim([np.min(logprior_samples[:, iw]) - 0.05,
                 np.max(logprior_samples[:, iw]) + 0.05])
    ax.tick_params(which='both', labelsize=6)
    ax.set_ylabel('log prior', fontsize=6)
    ax.set_xticklabels([])
    ax.text(0.95, 0.05, 'ln prior', fontsize=12, ha='right', color='purple',
            transform=ax.transAxes)

# now cycle through parameters
ax_ixl = [np.floor_divide(idim, 5) for idim in np.arange(2, ndim+2)]
ax_ixh = [(idim % 5) for idim in np.arange(2, ndim+2)]
for idim in range(ndim):
    ax = fig.add_subplot(gs[ax_ixl[idim], ax_ixh[idim]])
    for iw in range(nwalk):
        ax.plot(np.arange(nsteps), samples[:, iw, idim], color='k', alpha=0.03)
    ax.plot([0, nsteps], [theta[idim], theta[idim]], '--C1', lw=1.5)
    ax.set_xlim([0, nsteps])
    ax.tick_params(which='both', labelsize=6)
    ax.set_ylabel(lbls[idim], fontsize=6)
    if idim != 8:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('steps', fontsize=6)
    ax.text(0.95, 0.05, lbls[idim], fontsize=12, ha='right', color='purple',
            transform=ax.transAxes)


fig.subplots_adjust(wspace=0.20, hspace=0.05)
fig.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.99)
fig.savefig('mcmc_analysis/'+sys.argv[1]+'.traces.png')
fig.clf()



if np.int(sys.argv[2]) > 0:

    # Corner plot to visualize covariances
    levs = 1. - np.exp(-0.5 * (np.arange(3) + 1)**2)
    flat_chain = samples.reshape(-1, ndim)
    fig = corner.corner(flat_chain, plot_datapoints=False, levels=levs, 
                        labels=lbls, truths=theta)
    fig.savefig('mcmc_analysis/'+sys.argv[1]+'.corner.png')
    fig.clf()


    # Parameter inferences (1-D marginalized)
    print(' ')
    prec = [0.01, 0.01, 0.001, 0.1, 0.01, 0.01, 0.1, 0.001, 0.1, 0.1, 0.1, 
            0.0001, 0.0001]
    for idim in range(ndim):
        fmt = '{:.'+str(np.abs(np.int(np.log10(prec[idim]))))+'f}'
        pk, hi, lo, med = post_summary(samples_[:,idim], prec=prec[idim])
        print((lbls[idim] + ' = '+fmt+' +'+fmt+' / -'+fmt).format(pk, hi, lo))
    print(' ')
