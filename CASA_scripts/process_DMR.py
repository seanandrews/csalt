import os, sys
import numpy as np
execfile('CASA_scripts/image_cube.py')


# Load configuration file
execfile('configs_modeling/mconfig_'+sys.argv[-1]+'.py')


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




# IMAGING

# Make a (Keplerian) mask if requested (or it doesn't already exist)
if np.logical_or(gen_msk, ~os.path.exists(dataname+_ext+'.mask')):
    generate_kepmask(sys.argv[-1], dataname+_ext+'_EB0.DAT', 
                     dataname+_ext+'.DAT')


# Image the cubes if requested
filetype = ['DAT', 'MOD', 'RES']
for i in [0]:	#range(len(filetype)):
    if gen_img[i]:
        files_ = [dataname+_ext+'_EB'+str(j)+'.'+filetype[i]+'.ms' 
                  for j in range(nobs)]
        print(sys.argv[-1])
        print(files_)
        clean_cube(sys.argv[-1], files_, dataname+_ext+'.'+filetype[i], 
                   maskname=dataname+'.mask')

os.system('rm -rf *.last')
