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
    data = np.squeeze(tb.getcol('DATA'))
    u, v = tb.getcol('UVW')[0,:], tb.getcol('UVW')[1,:]
    weights = tb.getcol('WEIGHT')
    times = tb.getcol('TIME')
    field_ids = tb.getcol('FIELD_ID')
    data_desc_ids = tb.getcol('DATA_DESC_ID')
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
    
    # Iterate over unique combinations of FIELD_ID and DATA_DESC_ID values to get different execution blocks
    unique_combinations = np.unique(np.column_stack((field_ids, data_desc_ids)), axis=0)
    ebs = len(unique_combinations)

    # Compute the LSRK frequencies (Hz) for each timestamp
    nu_LSRK = np.empty((len(tstamps), len(nu_TOPO), ebs))
    ms.open(MS_in+'.ms')
    for istamp in range(len(tstamps)):
        for eb in range(ebs):
            nu_LSRK[istamp,:, eb] = ms.cvelfreqs(mode='channel', outframe='LSRK',
                                         restfreq='345.796GHz', obstime=str(tstamps[istamp])+'s')
    ms.close()
    
    if append:
        outp = h5py.File(HDF_out+'.h5', "a")
    else:
        os.system('rm -rf '+HDF_out+'.h5')
        outp = h5py.File(HDF_out+'.h5', "w")
        outp.attrs['nobs'] = ebs
    
    for eb in range(len(unique_combinations)):
        # Record the results in HDF5 format
        field_id, data_desc_id = unique_combinations[eb]
        groupname_full = 'EB' + str(eb) + '/'
        outp.create_dataset(groupname_full+'um', data=u[field_ids == field_id])
        outp.create_dataset(groupname_full+'vm', data=v[field_ids == field_id])
        outp.create_dataset(groupname_full+'vis_real', data=data.real[:, :, data_desc_ids == data_desc_id])
        outp.create_dataset(groupname_full+'vis_imag', data=data.imag[:, :, data_desc_ids == data_desc_id])
        outp.create_dataset(groupname_full+'weights', data=weights[:, data_desc_ids == data_desc_id])
        outp.create_dataset(groupname_full+'nu_TOPO', data=nu_TOPO[:, eb])
        outp.create_dataset(groupname_full+'nu_LSRK', data=nu_LSRK[:, :, eb])
        outp.create_dataset(groupname_full+'tstamp_ID', data=tstamp_ID[field_ids == field_id])
    
    outp.close()

    return
