import os, sys
import numpy as np
import scipy.constants as sc


# Load configuration file
execfile(sys.argv[3]+'.py')


# Make sure outdir exists
outdir = reduced_dir+'/'+basename+'-'+extname+'/'
if not os.path.exists(outdir):
    os.system('mkdir '+outdir)


# Load original MS datafile
if len(sys.argv) > 3:
    in_MS += '.'+sys.argv[4]
tb.open(in_MS+'.ms')
spw_col = tb.getcol('DATA_DESC_ID')
obs_col = tb.getcol('OBSERVATION_ID')
field_col = tb.getcol('FIELD_ID')
tb.close()


# Identify unique EBs
obs_ids = np.unique(obs_col)
nEB = len(obs_ids)


# Separate EBs and split out with time-averaging (if desired)
for EB in range(nEB):
    spws = np.unique(spw_col[np.where(obs_col == obs_ids[EB])])
    if len(spws) == 1:
        spw_str = str(spws[0])
    else:
        spw_str = "%d~%d" % (spws[0], spws[-1])

    fields = np.unique(field_col[np.where(obs_col == obs_ids[EB])]) 
    if len(fields) == 1:
        field_str = str(fields[0])
    else:
        field_str = "%d~%d" % (fields[0], fields[-1])

    os.system('rm -rf '+dataname+'_tmp'+str(EB)+'.ms*')
    split(vis=in_MS+'.ms', outputvis=dataname+'_tmp'+str(EB)+'.ms', 
          spw=spw_str, field=field_str, datacolumn='data', timebin=tavg[EB], 
          keepflags=False)


# Create a data dictionary
data_dict = {'nobs': nEB, 
             'original_MS': in_MS, 
             'V_bounds_V': V_bounds,
             'tavg': tavg}
np.save(dataname, data_dict)


# Loop through each EB
for EB in range(nEB):

    # Get data
    tb.open(dataname+'_tmp'+str(EB)+'.ms')
    data_all = np.squeeze(tb.getcol('DATA'))
    u, v = tb.getcol('UVW')[0,:], tb.getcol('UVW')[1,:]
    wgt_all = tb.getcol('WEIGHT')
    times = tb.getcol('TIME')
    tb.close()

    # Parse timestamps
    tstamps = np.unique(times)
    tstamp_ID = np.empty_like(times)
    for istamp in range(len(tstamps)):
        tstamp_ID[times == tstamps[istamp]] = istamp

    # Get TOPO frequencies
    tb.open(dataname+'_tmp'+str(EB)+'.ms/SPECTRAL_WINDOW')
    nu_TOPO_all = np.squeeze(tb.getcol('CHAN_FREQ'))
    tb.close()

    # Calculate LSRK frequencies for each timestamp
    nu_LSRK_all = np.empty((len(tstamps), len(nu_TOPO_all)))
    ms.open(dataname+'_tmp'+str(EB)+'.ms')
    for istamp in range(len(tstamps)):
        nu_LSRK_all[istamp,:] = ms.cvelfreqs(mode='channel', outframe='LSRK', 
                                             obstime=str(tstamps[istamp])+'s')
    ms.close()

    # Identify channel boundaries for the requested LSRK range
    V_LSRK_all = sc.c * (1 - nu_LSRK_all / nu_rest)
    chslo = np.argmin(np.abs(V_LSRK_all - V_bounds[0]), axis=1)
    chshi = np.argmin(np.abs(V_LSRK_all - V_bounds[1]), axis=1)
    if np.diff(nu_TOPO_all)[0] < 0:
        chlo, chhi = chslo.min(), chshi.max()
    else:
        chlo, chhi = chshi.min(), chslo.max()	

    # Slice out the data of interest
    nu_TOPO = nu_TOPO_all[chlo-ch_pad:chhi+ch_pad+1]
    nu_LSRK = nu_LSRK_all[:,chlo-ch_pad:chhi+ch_pad+1]
    data = data_all[:,chlo-ch_pad:chhi+ch_pad+1,:]
    if wgt_all.shape == data_all.shape: 
        wgt = wgt_all[:,chlo-ch_pad:chhi+ch_pad+1,:]
    else:
        wgt = wgt_all

    # Pack a data object into an .npz file
    os.system('rm -rf '+dataname+'_EB'+str(EB)+'.npz')
    np.savez_compressed(dataname+'_EB'+str(EB), data=data, um=u, vm=v, 
                        weights=wgt, tstamp_ID=tstamp_ID, 
                        nu_TOPO=nu_TOPO, nu_LSRK=nu_LSRK)

    # Split off a MS with the data of interest (for future imaging use)
    os.system('rm -rf '+dataname+'_EB'+str(EB)+'.DAT.ms*')
    spwtag = '0:'+str(chlo-ch_pad)+'~'+str(chhi+ch_pad)
    split(vis=dataname+'_tmp'+str(EB)+'.ms', 
          outputvis=dataname+'_EB'+str(EB)+'.DAT.ms',
          datacolumn='data', spw=spwtag)


# Cleanup
os.system('rm -rf '+dataname+'_tmp*.ms*')
os.system('rm -rf *.last')
