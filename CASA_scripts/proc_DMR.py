import os, sys
import numpy as np
execfile('fconfig.py')


# load the metadata
data_dict = np.load(dataname+'.npy', allow_pickle=True).item()
nobs = data_dict['nobs']

# loop through each EB
for i in range(nobs):

    # Regrid data if necessary
    if np.logical_or(~os.path.exists(dataname+'_DAT_regrid'+str(i)+'.ms'),
                     proc_D):
        os.system('rm -rf '+dataname+'_DAT_regrid'+str(i)+'.ms*')
        mstransform(vis=dataname+'_EB'+str(i)+'.ms', 
                    outputvis=dataname+'_DAT_regrid'+str(i)+'.ms',
                    keepflags=False, regridms=True, datacolumn='data', 
                    mode='velocity', outframe='LSRK', veltype='radio',
                    start=chanstart, width=chanwidth, nchan=nchan_out,
                    restfreq=str(nu_rest/1e9)+'GHz')

    # Create model and residual MS files
    os.system('rm -rf '+dataname+'_EB'+str(i)+'.MOD.ms*')
    os.system('cp -r '+dataname+'_EB'+str(i)+'.ms ' + \
               dataname+'_EB'+str(i)+'.MOD.ms')
    tb.open(dataname+'_EB'+str(i)+'.MOD.ms', nomodify=False)
    tb.putcol("DATA") = np.load(dataname+'_EB'+str(i)+'.npz')['model']
    tb.close()

    os.system('rm -rf '+dataname+'_EB'+str(i)+'.RES.ms*')
    os.system('cp -r '+dataname+'_EB'+str(i)+'.ms ' + \
               dataname+'_EB'+str(i)+'.RES.ms')
    tb.open(dataname+'_EB'+str(i)+'.RES.ms', nomodify=False)
    tb.putcol("DATA") -= np.load(dataname+'_EB'+str(i)+'.npz')['model']
    tb.close()
 
    # Regrid model and residuals
    os.system('rm -rf '+dataname+'_MOD_regrid'+str(i)+'.ms*')
    mstransform(vis=dataname+'EB'+str(i)+'.MOD.ms',
                outputvis=dataname+'_MOD_regrid'+str(i)+'.ms',
                keepflags=False, regridms=True, datacolumn='data',
                mode='velocity', outframe='LSRK', veltype='radio',
                start=chanstart, width=chanwidth, nchan=nchan_out,
                restfreq=str(nu_rest/1e9)+'GHz')

    os.system('rm -rf '+dataname+'_RES_regrid'+str(i)+'.ms*')
    mstransform(vis=dataname+'EB'+str(i)+'.RES.ms',
                outputvis=dataname+'_RES_regrid'+str(i)+'.ms',
                keepflags=False, regridms=True, datacolumn='data',
                mode='velocity', outframe='LSRK', veltype='radio',
                start=chanstart, width=chanwidth, nchan=nchan_out,
                restfreq=str(nu_rest/1e9)+'GHz')


# Concatenate the regridded EBs (if necessary)
if np.logical_or(~os.path.exists(dataname+'_DAT_regrid.ms'), proc_D):
    dfiles = [dataname+'_DAT_regrid'+str(i)+'.ms' for i in range(nobs)]
    os.system('rm -rf '+dataname+'_DAT_regrid.ms*')
    concat(vis=dfiles, concatvis=dataname+'_DAT_regrid.ms', dirtol='0.1arcsec',
           copypointing=False)

mfiles = [dataname+'_MOD_regrid'+str(i)+'.ms' for i in range(nobs)]
os.system('rm -rf '+dataname+'_MOD_regrid.ms*')
concat(vis=mfiles, concatvis=dataname+'_MOD_regrid.ms', dirtol='0.1arcsec',
       copypointing=False)

rfiles = [dataname+'_RES_regrid'+str(i)+'.ms' for i in range(nobs)]
os.system('rm -rf '+dataname+'_RES_regrid.ms*')
concat(vis=mfiles, concatvis=dataname+'_RES_regrid.ms', dirtol='0.1arcsec',
       copypointing=False)



