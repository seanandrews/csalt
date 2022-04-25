import os, sys
import numpy as np

### User inputs
pri_types = ['uniform', 'uniform', 'uniform', 'uniform', 'uniform',
             'uniform', 'uniform', 'uniform', 'uniform', 'uniform',
             'uniform', 'uniform', 'uniform', 'uniform', 'uniform']
pri_pars = [ [0, 90], [0, 180], [0.5, 1.5], [150, 350], [0.0, 0.5],
             [0.0, 2.0], [50, 250], [-1, 0], [5, 100], [50, 700],
             [1.0, 4.0], [-2, 0], [4e3, 6e3], [-0.1, 0.1], [-0.1, 0.1] ]


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
