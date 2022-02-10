"""
    Extract the relevant information from a given measurement set and 
    record it (in HDF5 format) for easier use in csalt infrastructure.
"""

import os, sys
import numpy as np
import h5py

def ms_to_hdf5(MS_in, HDF_out, append=False, groupname=None):

    # Check input file formatting
    if MS_in[-3:] == '.ms': MS_in = MS_in[:-3]
    if not os.path.exists(MS_in+'.ms'):
        print('The MS file you are trying to write out to HDF5 does not exist.')
        return

    # Check output kwargs
    if append:
        # make sure the output file already exists
        if not os.path.exists(HDF_out+'.h5'):
            print('The HDF5 file you are trying to append does not exist.')
            return
        elif groupname == None:
            print('You need to specify a groupname to append in the HDF5 file')
            return
        else:
            if groupname[-1] != '/': groupname += '/'
    else:
        if groupname == None:
            groupname = ''
        else:
            if groupname[-1] != '/': groupname += '/'

    # Acquire MS information (for easier use external to CASA)
    tb.open(MS_in+'.ms')
    data = np.squeeze(tb.getcol("DATA"))
    u, v = tb.getcol('UVW')[0,:], tb.getcol('UVW')[1,:]
    weights = tb.getcol('WEIGHT')
    times = tb.getcol("TIME")
    tb.close()

    # Index the timestamps
    tstamps = np.unique(times)
    tstamp_ID = np.empty_like(times)
    for istamp in range(len(tstamps)):
        tstamp_ID[times == tstamps[istamp]] = istamp

    # Acquire the TOPO frequency channels (Hz)
    tb.open(MS_in+'.ms/SPECTRAL_WINDOW')
    nu_TOPO = np.squeeze(tb.getcol('CHAN_FREQ'))
    tb.close()

    # Compute the LSRK frequencies (Hz) for each timestamp
    nu_LSRK = np.empty((len(tstamps), len(nu_TOPO)))
    ms.open(MS_in+'.ms')
    for istamp in range(len(tstamps)):
        nu_LSRK[istamp,:] = ms.cvelfreqs(mode='channel', outframe='LSRK',
                                         obstime=str(tstamps[istamp])+'s')
    ms.close()

    # Record the results in HDF5 format
    if append:
        outp = h5py.File(HDF_out+'.h5', "a")
    else:
        os.system('rm -rf '+HDF_out+'.h5')
        outp = h5py.File(HDF_out+'.h5', "w")
        outp.attrs['nobs'] = 1
    outp.create_dataset(groupname+'um', data=u)
    outp.create_dataset(groupname+'vm', data=v)
    outp.create_dataset(groupname+'vis_real', data=data.real)
    outp.create_dataset(groupname+'vis_imag', data=data.imag)
    outp.create_dataset(groupname+'weights', data=weights)
    outp.create_dataset(groupname+'nu_TOPO', data=nu_TOPO)
    outp.create_dataset(groupname+'nu_LSRK', data=nu_LSRK)
    outp.create_dataset(groupname+'tstamp_ID', data=tstamp_ID)
    outp.close()

    return
