import os, sys
import numpy as np
execfile('CASA_scripts/reduction_utils.py')

inp_MS = '/data/sandrews/LP/sa_work/Sz129/Sz129_combined_CO_selfcal.ms.contsub'
pre_0 = 'obs_templates/Sz129_init'
pre_out = 'obs_templates/Sz129_EB'
restfreq = 230.538e9
chpad = 5
tavg = '30s'


# split the input MS into the individual EBs
nobs = 4	#split_all_obs(inp_MS, 'obs_templates/'+pre_0)

# establish the channel range to split out (based on desired LSRK outputs)
vLSRK_lo = chanstart_out 
vLSRK_hi = chanstart_out + chanwidth_out * (nchan_out - 1) 
get_ch = np.zeros((2, nobs))
for i in range(nobs):
    # get field 
    tb.open(pre_0+str(i)+'.ms/FIELD')
    field = tb.getcol('NAME')
    tb.close()

    # get SPW 
    tb.open(pre_0+str(i)+'.ms')
    spw_col = tb.getcol('DATA_DESC_ID')
    tb.close()
    spw = np.unique(spw_col)

    # identify channel bounds around desired LSRK velocities
    chlims = LSRKvel_to_chan(pre_0+str(i)+'.ms', field, spw,
                             restfreq, 1e-3*np.array([vLSRK_lo, vLSRK_hi]))

    # split out the reduced, time-averaged data
    spec_out = str(spw[0])+':'+str(np.int(chlims[0])-chpad) + \
               '~'+str(np.int(chlims[1])+chpad)
    #os.system('rm -rf '+pre_out+str(i)+'.ms*')
    #split(vis=pre_0+str(i)+'.ms', field=field[0], spw=spec_out, timebin=tavg, 
    #      datacolumn='data', keepflags=False, outputvis=pre_out+str(i)+'.ms')

    # delete the intermediate MS files
    #os.system('rm -rf '+pre_0+str(i)+'.ms*')

    
