import os, sys
import numpy as np

topo_in = np.loadtxt('TOPO1.txt').T

lsrk_out = np.empty_like(topo_in)
for i in range(len(topo_in)):
    lsrk_out[i] = au.topoToLSRK(topo_in[i], '2022/08/01/00:00:00', '16:00:00',
                                '-40.00.00')

np.savetxt('LSRK2.txt', lsrk_out)
