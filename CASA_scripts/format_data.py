import os, sys
import numpy as np
execfile('const.py')
execfile('mconfig_'+sys.argv[-1]+'.py')


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
             'restfreq': restfreq,
             'bounds_V': bounds_V,
             'bounds_pad': bounds_pad,
             'tavg': tavg}
np.save(dataname, data_dict)


# Loop through each EB
for iobs in range(nEB):

    # Get data
    tb.open(dataname+'_tmp'+str(iobs)+'.ms')
    data_all = np.squeeze(tb.getcol('DATA'))
    u, v = tb.getcol('UVW')[0,:], tb.getcol('UVW')[1,:]
    wgt_all = tb.getcol('WEIGHT')
    times = tb.getcol('TIME')
    tb.close()

    # Parse timestamps
    tstamps = np.unique(times)
    tstamp_ID = np.empty_like(times)
    for i in range(len(tstamps)):
        tstamp_ID[times == tstamps[i]] = i

    # Get TOPO frequencies
    tb.open(dataname+'_tmp'+str(iobs)+'.ms/SPECTRAL_WINDOW')
    nu_TOPO_all = np.squeeze(tb.getcol('CHAN_FREQ'))
    tb.close()

    # Calculate LSRK frequencies for each timestamp
    nu_LSRK_all = np.empty((len(tstamps), len(nu_TOPO_all)))
    ms.open(dataname+'_tmp'+str(iobs)+'.ms')
    for i in range(len(tstamps)):
        nu_LSRK_all[i,:] = ms.cvelfreqs(mode='channel', outframe='LSRK', 
                                        obstime=str(tstamps[i])+'s')
    ms.close()

    # Identify channel boundaries for the requested LSRK range
    V_LSRK_all = c_ * (1 - nu_LSRK_all / restfreq)
    chslo = np.argmin(np.abs(V_LSRK_all - bounds_V[0]), axis=1)
    chshi = np.argmin(np.abs(V_LSRK_all - bounds_V[1]), axis=1)
    if np.diff(nu_TOPO_all)[0] < 0:
        chlo, chhi = chslo.min(), chshi.max()
    else:
        chlo, chhi = chshi.min(), chslo.max()	

    # Slice out the data of interest
    nu_TOPO = nu_TOPO_all[chlo-bounds_pad:chhi+bounds_pad+1]
    nu_LSRK = nu_LSRK_all[:,chlo-bounds_pad:chhi+bounds_pad+1]
    data = data_all[:,chlo-bounds_pad:chhi+bounds_pad+1,:]
    if wgt_all.shape == data_all.shape: 
        wgt = wgt_all[:,chlo-bounds_pad:chhi+bounds_pad+1,:]
    else:
        wgt = wgt_all

    # Pack a data object into an .npz file
    os.system('rm -rf '+dataname+'_EB'+str(iobs)+'.npz')
    np.savez_compressed(dataname+'_EB'+str(iobs), data=data, um=u, vm=v, 
                        weights=wgt, tstamp_ID=tstamp_ID, 
                        nu_TOPO=nu_TOPO, nu_LSRK=nu_LSRK)

    # Split off a MS with the data of interest (for future imaging use)
    os.system('rm -rf '+dataname+'_EB'+str(iobs)+'.DAT.ms*')
    spwtag = '0:'+str(chlo-bounds_pad)+'~'+str(chhi+bounds_pad)
    split(vis=dataname+'_tmp'+str(iobs)+'.ms', 
          outputvis=dataname+'_EB'+str(iobs)+'.DAT.ms',
          datacolumn='data', spw=spwtag)


# Cleanup
os.system('rm -rf '+dataname+'_tmp*.ms*')
os.system('rm -rf *.last')
