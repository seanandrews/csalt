import os, sys
import numpy as np
execfile('mconfig_'+sys.argv[-1]+'.py')
execfile('CASA_scripts/image_cube.py')


# load the metadata
data_dict = np.load(dataname+'.npy', allow_pickle=True).item()
nobs = data_dict['nobs']


# process the model and residual MS
for i in range(nobs):
    # load model and residual visibilities
    m_ = np.load(dataname+'_EB'+str(i)+'.MOD.npz')

    # pack up model
    os.system('rm -rf '+dataname+'_EB'+str(i)+'.MOD.ms*')
    os.system('cp -r '+dataname+'_EB'+str(i)+'.DAT.ms '+\
                       dataname+'_EB'+str(i)+'.MOD.ms')
    tb.open(dataname+'_EB'+str(i)+'.MOD.ms', nomodify=False)
    tb.putcol("DATA", m_['model'])
    tb.close()

    # pack up residuals
    os.system('rm -rf '+dataname+'_EB'+str(i)+'.RES.ms*')
    os.system('cp -r '+dataname+'_EB'+str(i)+'.DAT.ms '+\
                       dataname+'_EB'+str(i)+'.RES.ms')
    tb.open(dataname+'_EB'+str(i)+'.RES.ms', nomodify=False)
    tb.putcol("DATA", m_['resid'])
    tb.close()


# Make a (Keplerian) mask if requested (or it doesn't already exist)
if np.logical_or(gen_msk, ~os.path.exists(dataname+'.mask')):
    generate_kepmask(sys.argv[-1], dataname+'_EB0.DAT', dataname+'.DAT')


# Image the cubes if requested
filetype = ['DAT', 'MOD', 'RES']
for i in range(len(filetype)):
    if gen_img[i]:
        files_ = [dataname+'_EB'+str(j)+'.'+filetype[i]+'.ms' 
                  for j in range(nobs)]
        clean_cube(sys.argv[-1], files_, dataname+'.'+filetype[i], 
                   maskname=dataname+'.mask')

os.system('rm -rf *.last')
