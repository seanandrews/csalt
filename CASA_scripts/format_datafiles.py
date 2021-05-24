import os, sys
import numpy as np

inp_MS = '/data/sandrews/LP/sa_work/Sz129/Sz129_combined_CO_selfcal.ms.contsub'
tmpEBs = 'obs_templates/Sz129_init'
pre_out = 'obs_templates/Sz129_EB'
restfreq = 230.538e9
chpad = 5
tavg = '30s'


## Load original MS datafile
#tb.open(inp_MS)
#spw_col = tb.getcol('DATA_DESC_ID')
#obs_col = tb.getcol('OBSERVATION_ID')
#field_col = tb.getcol('FIELD_ID')
#tb.close()
#
## Identify unique EBs
#obs_ids = np.unique(obs_col)
#nEB = len(obs_ids)
#
## Separate EBs and split out with time-averaging (if desired)
#for i in obs_ids:
#    spws = np.unique(spw_col[np.where(obs_col==i)])
#    if len(spws) == 1:
#        spw_str = str(spws[0])
#    else:
#        spw_str = "%d~%d" % (spws[0], spws[-1])
#
#    fields = np.unique(field_col[np.where(obs_col==i)]) 
#    if len(fields) == 1:
#        field_str = str(fields[0])
#    else:
#        field_str = "%d~%d" % (fields[0], fields[-1])
#
#    os.system('rm -rf '+tmpEBs+str(i)+'.ms*')
#    split(vis=inp_MS, spw=spw_str, field=field_str, datacolumn='data',
#          timebin=tavg, keepflags=False, outputvis=tmpEBs+str(i)+'.ms')



for i in [0]:	#range(nEB):

    # open MS for individual EB and extract relevant information
    tb.open(tmpEBs+str(i)+'.ms')
    data = np.squeeze(tb.getcol("DATA"))
    uvw = tb.getcol("UVW")
    weights = tb.getcol("WEIGHT")
    times = tb.getcol("TIME")
    spw = np.unique(tb.getcol("DATA_DESC_ID"))
    field = np.unique(tb.getcol("FIELD_ID"))
    tb.close()

    # get TOPO channel frequencies
    tb.open(tmpEBs+str(i)+'.ms/SPECTRAL_WINDOW')
    nch_in = tb.getcol('NUM_CHAN').tolist()[0]
    freq_TOPO_inp = np.squeeze(tb.getcol("CHAN_FREQ"))
    tb.close()

    # identify unique timestamps (in MJD)
    tstamps = np.unique(times)
    nstamps = len(tstamps)

    # LSRK channel frequencies for each timestamp
    freq_LSRK_inp = np.empty((nstamps, nch_in))
    ms.open(tmpEBs+str(i)+'.ms')
    for j in range(nstamps):
        freq_LSRK_inp[j,:] = ms.cvelfreqs(spwids=spw, fieldids=field, 
                                          mode='channel', outframe='LSRK', 
                                          obstime=str(tstamps[j])+'s')
    ms.close()

    # Now linear interpolate onto fixed LSRK velocity grid at native spacing
    print(np.diff(freq_LSRK_inp[0,:]))
    print(np.diff(freq_TOPO_inp))






#    chlims = LSRKvel_to_chan(tmpEBs+str(i)+'.ms', field, spw, restfreq, 
#                             1e-3*np.array([vlo, vhi]))

    # split out the reduced, time-averaged data
#    spec_out = str(spw[0])+':'+str(np.int(chlims[0])-chpad) + \
#               '~'+str(np.int(chlims[1])+chpad)
    #os.system('rm -rf '+pre_out+str(i)+'.ms*')
    #split(vis=tmpEBs+str(i)+'.ms', field=field[0], spw=spec_out, timebin=tavg, 
    #      datacolumn='data', keepflags=False, outputvis=pre_out+str(i)+'.ms')

    # delete the intermediate MS files
    #os.system('rm -rf '+tmpEBs+str(i)+'.ms*')


