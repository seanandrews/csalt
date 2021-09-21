import os, sys
import numpy as np


# Load configuration file
execfile('configs/mconfig_'+sys.argv[-1]+'.py')


# load the metadata
data_dict = np.load(dataname+'.npy', allow_pickle=True).item()
nobs = data_dict['nobs']


# process the model and residual MS
for EB in range(nobs):
    # load model and residual visibilities
    m_ = np.load(dataname+_ext+'_EB'+str(EB)+'.MOD.npz')

    # pack up model
    os.system('rm -rf '+dataname+_ext+'_EB'+str(EB)+'.MOD.ms*')
    os.system('cp -r '+dataname+_ext+'_EB'+str(EB)+'.DAT.ms '+ \
                       dataname+_ext+'_EB'+str(EB)+'.MOD.ms')
    tb.open(dataname+_ext+'_EB'+str(EB)+'.MOD.ms', nomodify=False)
    tb.putcol("DATA", m_['model'])
    tb.close()

    # pack up residuals
    os.system('rm -rf '+dataname+_ext+'_EB'+str(EB)+'.RES.ms*')
    os.system('cp -r '+dataname+_ext+'_EB'+str(EB)+'.DAT.ms '+\
                       dataname+_ext+'_EB'+str(EB)+'.RES.ms')
    tb.open(dataname+_ext+'_EB'+str(EB)+'.RES.ms', nomodify=False)
    tb.putcol("DATA", m_['resid'])
    tb.close()


os.system('rm -rf *.last')
