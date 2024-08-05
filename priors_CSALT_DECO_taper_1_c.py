import os, sys
import numpy as np
import scipy.constants as sc

### User inputs
pri_types = ['normal',  'normal', 'uniform',  'normal', 'normal',
             'uniform', 'normal', 'uniform', 'normal',
             'uniform',  'normal', 'uniform',  'normal',
             'uniform', 'uniform', 'uniform', 'uniform',
             'uniform', 'uniform',
             'normal',  'normal',  'normal']
#             'uniform', 'normal']
pri_pars = [ [35, 5], [150, 5], [0.5, 1.5], [150, 50], [150,50],
            [0.0, 0.5], [1.25, 0.5], [0.0, 0.5], [1.25, 0.5],
            [20, 150], [-0.5, 0.2], [20, 150], [-0.5, 0.2],
            [0, 10], [0, 10], [0, 10], [0, 10],
            [1.0, 4.0], [-3, 0],
            [4000, 2e2], [0.0, 0.05], [0.0, 0.05]]
#            [0.3, 1.5], [1.5, 0.5] ]


### Pre-defined standard functions
# uniform, bounded prior: ppars = [low, hi]
def logprior_uniform(theta, ppars):
    if np.logical_and((theta >= ppars[0]), (theta <= ppars[1])):
        return np.log(1 / (ppars[1] - ppars[0]))
    else:
        return -np.inf

# Gaussian prior: ppars = [mean, std dev]
def logprior_normal(theta, ppars):
    foo = -np.log(ppars[1] * np.sqrt(2 * np.pi)) \
          -0.5 * ((theta - ppars[0])**2 / ppars[1]**2) 
    return foo


### Log-Prior calculator
def logprior(theta):

    # initialize
    logptheta = np.empty_like(theta)

    # user-defined calculations. 
    for i in range(len(theta)):
        cmd = 'logprior_'+pri_types[i]+'(theta['+str(i)+'], '+\
              str(pri_pars[i])+')'
        logptheta[i] = eval(cmd)

    # return
    return logptheta
