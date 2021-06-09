import os, sys
import numpy as np

inp_MS = '/data/sandrews/LP/sa_work/Sz129/Sz129_combined_CO_selfcal.ms.contsub'
basename = 'Sz129'
odir = 'data/'+basename+'/'
tmpEBs = odir+basename+'_init'
pre_out = odir+basename+'_EB'
restfreq = 230.538e9
bounds_V = [-10e3, 10e3]
chpad = 5
tavg = '30s'

c_ = 2.99792e8


class vdata:
   def __init__(self, u, v, vis, wgt, nu_topo, nu_lsrk):
        self.u = u
        self.v = v
        self.vis = vis
        self.wgt = wgt
        self.nu_topo = nu_topo
        self.nu_lsrk = nu_lsrk


# Make sure odir exists
if not os.path.exists(odir):
    os.system('mkdir '+odir)


""" 
Start by time-averaging a self-calibrated composite MS, and splitting it into
the constituent EBs.
"""
# Load original MS datafile
tb.open(inp_MS)
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

    os.system('rm -rf '+tmpEBs+str(i)+'.ms*')
    split(vis=inp_MS, spw=spw_str, field=field_str, datacolumn='data',
          timebin=tavg, keepflags=False, outputvis=tmpEBs+str(i)+'.ms')


#nEB = 4


# create a data dictionary
data_dict = {"nobs": 4}

for i in range(nEB):

    print(i)

    # get data, timestamps
    tb.open(tmpEBs+str(i)+'.ms')
    data_all = np.squeeze(tb.getcol('DATA'))
    uvw = tb.getcol('UVW')
    weights = tb.getcol('WEIGHT')
    tstamps = np.unique(tb.getcol('TIME'))
    tb.close()

    # get TOPO frequencies
    tb.open(tmpEBs+str(i)+'.ms/SPECTRAL_WINDOW')
    nu_TOPO_all = np.squeeze(tb.getcol('CHAN_FREQ'))
    tb.close()

    # calculate LSRK frequencies for each timestamp
    nu_LSRK_all = np.empty((len(tstamps), len(nu_TOPO_all)))
    print(nu_LSRK_all.shape)
    ms.open(tmpEBs+str(i)+'.ms')
    for j in range(len(tstamps)):
        nu_LSRK_all[j,:] = ms.cvelfreqs(mode='channel', outframe='LSRK', 
                                        obstime=str(tstamps[j])+'s')
    ms.close()
    print(nu_LSRK_all.shape)

    # identify channel boundaries for the requested LSRK range
    V_LSRK_all = c_ * (1 - nu_LSRK_all / restfreq)
    chslo = np.argmin(np.abs(V_LSRK_all - bounds_V[0]), axis=1)
    chshi = np.argmin(np.abs(V_LSRK_all - bounds_V[1]), axis=1)
    if np.diff(nu_TOPO_all)[0] < 0:
        chlo, chhi = chslo.min(), chshi.max()
    else:
        chlo, chhi = chslo.max(), chshi.min()	# <--- revisit this!

    # slice out the data of interest
    nu_TOPO = nu_TOPO_all[chlo-chpad:chhi+chpad+1]
    nu_LSRK = nu_LSRK_all[:,chlo-chpad:chhi+chpad+1]
    data = data_all[:,chlo-chpad:chhi+chpad+1,:]

    # parse visibility weights if they do not have a spectral dependence
    print('gonna tile')
    if weights.shape != data.shape:
        weights = np.rollaxis(np.tile(weights, (len(nu_TOPO), 1, 1)), 1)

    # pack into a data object, store as .npz, and update the dictionary
    print('gonna class')
    out_data = vdata(uvw[0,:], uvw[1,:], data, weights, nu_TOPO, nu_LSRK)
    print('gonna save')
    np.savez(pre_out+str(i), data=out_data)
