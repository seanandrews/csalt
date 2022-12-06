import os, sys
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt


r = np.logspace(-1, 3, 2048)

p, gam, S0, rd, r0 = 1., 2., 100., 160., 10.

Sigma = S0 * (r / r0)**-p * np.exp(-(r / rd)**gam)

dSdr_num = np.gradient(Sigma, r)

dSdr_ana = -(Sigma / r) * (p + gam * (r / rd)**gam)

plt.plot(r, dSdr_num, 'C0')
plt.plot(r, dSdr_ana, '--C1')
plt.xlim([0.1, 1000])
plt.xscale('log')
plt.show()
