import os, sys
import numpy as np
import casatasks
import casatools
import scipy.constants as sc

# General visibility dataset object
class dataset:

   def __init__(self, um, vm, vis, wgt, nu_TOPO, nu_LSRK, tstamp_ID):

        # spectral frequencies in Hz units (LSRK for each timestamp)
        self.nu_TOPO = nu_TOPO
        self.nu_LSRK = nu_LSRK

        # spatial frequencies in meters units
        self.um = um
        self.vm = vm

        # spatial frequencies in lambda units
        self.ulam = self.um * np.mean(self.nu_TOPO) / sc.c
        self.vlam = self.vm * np.mean(self.nu_TOPO) / sc.c        

        # data visibilities, weights, and timestamp IDs
        self.vis = vis
        self.npol, self.nchan, self.nvis = vis.shape
        self.wgt = wgt

        self.tstamp = tstamp_ID
        self.nstamps = len(np.unique(tstamp_ID))



# General function to load an MS file (containing an arbitrary number of EBs) 
# to a data dictionary containing datasets for each observation.
def loadMS(msfile):

    # Make sure the file exists
    if not os.path.exists(msfile):
        print('I cannot find '+msfile+'.  Exiting.')
        return

    # Load the basic MS file information
    tb = casatools.table()
    tb.open(msfile)
    obs_col = tb.getcol('OBSERVATION_ID')
    SPW_col = tb.getcol('DATA_DESC_ID')
    field_col = tb.getcol('FIELD_ID')
    tb.close()

    # Identify the distinct EBs in the input MS
    EB_ids = np.unique(obs_col)
    nEB = len(EB_ids)

    # Initialize a data dictionary
    data = {'nEB': nEB}

    # Cycle through EBs to load information into individual dataset objects, 
    # packed into the data dictionary (if nEB = 1, just load directly)
    for EB in range(nEB):
        # prepare a temporary MS filename
        if nEB == 1:
            tmp_MS = msfile
        else:
            tmp_MS = 'tmp_'+str(EB)+'.ms'
            os.system('rm -rf '+tmp_MS+'*')

            # identify unique SPWs and fields
            spws = np.unique(SPW_col[np.where(obs_col == EB_ids[EB])])
            fields = np.unique(field_col[np.where(obs_col == EB_ids[EB])])
            spw_str = "%d~%d" % (spws[0], spws[-1])

            # SPW and field strings
            if len(spws) == 1:
                spw_str = str(spws[0])
            else:
                spw_str = "%d~%d" % (spws[0], spws[-1])
            if len(fields) == 1:
                field_str = str(fields[0])
            else:
                field_str = "%d~%d" % (fields[0], fields[-1])

            # split to a temporary MS file
            casatasks.split(msfile, outputvis=tmp_MS, datacolumn='data',
                            spw=spw_str, field=field_str, keepflags=False)

        # load the data from the split MS file
        tb.open(tmp_MS)
        vis = np.squeeze(tb.getcol('DATA'))
        u, v = tb.getcol('UVW')[0,:], tb.getcol('UVW')[1,:]
        wgts = tb.getcol('WEIGHT')
        times = tb.getcol('TIME')
        tb.close()

        # index the timestamps
        tstamps = np.unique(times)
        tstamp_ID = np.empty_like(times)
        for istamp in range(len(tstamps)):
            tstamp_ID[times == tstamps[istamp]] = istamp

        # acquire the TOPO frequency channels
        tb.open(tmp_MS+'/SPECTRAL_WINDOW')
        nu_TOPO = np.squeeze(tb.getcol('CHAN_FREQ'))
        tb.close()

        # compute the LSRK frequencies for each timestamp
        ms = casatools.ms()
        nu_LSRK = np.empty((len(tstamps), len(nu_TOPO)))
        ms.open(tmp_MS)
        for istamp in range(len(tstamps)):
            nu_LSRK[istamp,:] = ms.cvelfreqs(mode='channel', outframe='LSRK',
                                             obstime=str(tstamps[istamp])+'s')
        ms.close()

        # append a dataset object to the dataset dictionary
        data[str(EB)] = dataset(u, v, vis, wgts, nu_TOPO, nu_LSRK, tstamp_ID)

        # clean up the temporary MS
        os.system('rm -rf '+tmp_MS+'*')

    return data



