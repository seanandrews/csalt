import os, sys
import numpy as np
import scipy.constants as sc
from scipy.stats import norm

### User input for priors

# Stellar mass, inclination, scale height, rc, rin, psi, PA, dust parameter, vturb

pri_types = [ 'normal',  'uniform', 'uniform',  'uniform', 'uniform',
              'truncnorm', 'normal',  'loguniform', 'uniform']
pri_pars = [ [36, 1], [0.2, 0.6], [7, 25], [100, 300], [0.1, 10],
             [1.3, 0.1, 1, 2], [156, 2], [1e-5, 1e-3], [0.0, 0.2]]


### Pre-defined standard functions
# uniform, bounded prior: ppars = [low, hi]
def logprior_uniform(theta, ppars):
    if np.logical_and((theta >= ppars[0]), (theta <= ppars[1])):
        return np.log(1 / (ppars[1] - ppars[0]))
    else:
        return -np.inf

# Gaussian prior: ppars = [mean, std dev]
def logprior_normal(theta, ppars):
    foo = np.log(ppars[1] * np.sqrt(2 * np.pi)) + \
          0.5 * ((theta - ppars[0])**2 / ppars[1]**2)
    return -foo
    
# Gaussian bounded prior: ppars = [mean, std dev, low, hi]
def logprior_truncnorm(theta, ppars):
    if np.logical_and((theta >= ppars[2]), (theta <= ppars[3])):
        normalisation = np.log(norm.cdf((ppars[3]-ppars[0])/ppars[1]) - norm.cdf((ppars[2]-ppars[0])/ppars[1]))
        foo = np.log(ppars[1] * np.sqrt(2 * np.pi)) + \
              0.5 * ((theta - ppars[0])**2 / ppars[1]**2)
        return -foo-normalisation
    else:
        return -np.inf
        
# Log uniform bounded prior: ppars = [low, hi]
def logprior_loguniform(theta, ppars):
    if np.logical_and((theta >= ppars[0]), (theta <= ppars[1])):
        return -np.log(theta * np.log(ppars[1]/ppars[0]))
    else:
        return -np.inf



### Log-Prior calculator
def logprior(theta):

    # initialize
    logptheta = np.empty_like(theta)

    # user-defined calculations
    for i in range(len(theta)):
        cmd = 'logprior_'+pri_types[i]+'(theta['+str(i)+'], '+\
              str(pri_pars[i])+')'
        logptheta[i] = eval(cmd)

    # return
    return logptheta
