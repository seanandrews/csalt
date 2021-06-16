import os, sys
import numpy as np
execfile('const.py')
execfile('fconfig.py')


# Make sure outdir exists
if not os.path.exists(outdir):
    os.system('mkdir '+outdir)


# Load original MS datafile
tb.open(orig_MS)
spw_col = tb.getcol('DATA_DESC_ID')
obs_col = tb.getcol('OBSERVATION_ID')
field_col = tb.getcol('FIELD_ID')
tb.close()

# Identify unique EBs
obs_ids = np.unique(obs_col)
nEB = len(obs_ids)

# Separate EBs and split out with time-averaging (if desired)
for i in obs_ids:
    spws = np.unique(spw_col[np.where(obs_col==i)])
    if len(spws) == 1:
        spw_str = str(spws[0])
    else:
        spw_str = "%d~%d" % (spws[0], spws[-1])

    fields = np.unique(field_col[np.where(obs_col==i)]) 
    if len(fields) == 1:
        field_str = str(fields[0])
    else:
        field_str = "%d~%d" % (fields[0], fields[-1])

    os.system('rm -rf '+dataname+'_tmp'+str(i)+'.ms*')
    split(vis=orig_MS, outputvis=dataname+'_tmp'+str(i)+'.ms', spw=spw_str, 
          field=field_str, datacolumn='data', timebin=tavg, keepflags=False)


# Create a data dictionary
data_dict = {'nobs': nEB, 
             'orig_datafile': orig_MS, 
             'nu_rest': nu_rest,
             'bounds_V': bounds_V,
             'chpad': chpad,
             'tavg': tavg}
np.save(dataname, data_dict)


# Loop through each EB
for i in range(nEB):

    # Get data, timestamps
    tb.open(dataname+'_tmp'+str(i)+'.ms')
    data_all = np.squeeze(tb.getcol('DATA'))
    u, v = tb.getcol('UVW')[0,:], tb.getcol('UVW')[1,:]
    weights = tb.getcol('WEIGHT')
    tstamps = np.unique(tb.getcol('TIME'))
    tb.close()

    # Get TOPO frequencies
    tb.open(dataname+'_tmp'+str(i)+'.ms/SPECTRAL_WINDOW')
    nu_TOPO_all = np.squeeze(tb.getcol('CHAN_FREQ'))
    tb.close()

    # Calculate LSRK frequencies for each timestamp
    nu_LSRK_all = np.empty((len(tstamps), len(nu_TOPO_all)))
    ms.open(dataname+'_tmp'+str(i)+'.ms')
    for j in range(len(tstamps)):
        nu_LSRK_all[j,:] = ms.cvelfreqs(mode='channel', outframe='LSRK', 
                                        obstime=str(tstamps[j])+'s')
    ms.close()

    # Identify channel boundaries for the requested LSRK range
    V_LSRK_all = c_ * (1 - nu_LSRK_all / nu_rest)
    chslo = np.argmin(np.abs(V_LSRK_all - bounds_V[0]), axis=1)
    chshi = np.argmin(np.abs(V_LSRK_all - bounds_V[1]), axis=1)
    if np.diff(nu_TOPO_all)[0] < 0:
        chlo, chhi = chslo.min(), chshi.max()
    else:
        chlo, chhi = chshi.min(), chslo.max()	

    # Slice out the data of interest
    nu_TOPO = nu_TOPO_all[chlo-chpad:chhi+chpad+1]
    nu_LSRK = nu_LSRK_all[:,chlo-chpad:chhi+chpad+1]
    data = data_all[:,chlo-chpad:chhi+chpad+1,:]

    # Parse visibility weights if they do not have a spectral dependence
    if weights.shape != data.shape:
        weights = np.rollaxis(np.tile(weights, (len(nu_TOPO), 1, 1)), 1)

    # Pack a data object into an .npz file
    os.system('rm -rf '+dataname+'_EB'+str(i)+'.npz')
    np.savez_compressed(dataname+'_EB'+str(i), u=u, v=v, data=data, 
                        weights=weights, nu_TOPO=nu_TOPO, nu_LSRK=nu_LSRK)

    # Split off a MS with the data of interest (for future imaging use)
    os.system('rm -rf '+dataname+'_EB'+str(i)+'.ms*')
    split(vis=dataname+'_tmp'+str(i)+'.ms', 
          outputvis=dataname+'_EB'+str(i)+'.ms',
          datacolumn='data', spw='0:'+str(chlo-chpad)+'~'+str(chhi+chpad))

    # Clean up temporary MS files
    if not preserve_tmp:
        os.system('rm -rf '+dataname+'_tmp'+str(i)+'.ms*')

os.system('rm -rf *.last')
