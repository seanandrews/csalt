import os, sys
import copy
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



# Function to read contents of MS file into a dictionary
def read_MS(msfile):

    # Make sure the file exists
    if not os.path.exists(msfile):
        print('I cannot find '+msfile+'.  Exiting.')
        return

    # Ingest the SPW dictionary
    ms = casatools.ms()
    ms.open(msfile)
    spw_dict = ms.getspectralwindowinfo()
    ms.close()

    # Identify the number of distinct execution blocks
    Nobs = len(spw_dict)

    # Initialize a data dictionary
    data_dict = {'Nobs': Nobs, 'input_file': msfile}

    # Loop over executions to load information into dataset objects, packed 
    # into the data dictionary
    for EB in range(Nobs):

        # compute the TOPO frequencies
        spw = spw_dict[str(EB)]
        nu = spw['Chan1Freq'] + spw['ChanWidth'] * np.arange(spw['NumChan'])

        # open the MS file for this EB
        ms.open(msfile)
        ms.selectinit(datadescid=EB)

        # load the data into a dictionary
        d = ms.getdata(['data', 'weight', 'u', 'v', 'time'])

        # identify the unique timestamps
        tstamps = np.unique(d['time'])

        # timestamp index and LSRK frequency grids
        tstamp_ID = np.empty_like(d['time'])
        nu_ = np.empty((len(tstamps), len(nu)))

        # loop over timestamps to populate index and LSRK frequency grids
        for istamp in range(len(tstamps)):

            tstamp_ID[d['time'] == tstamps[istamp]] = istamp

            nu_[istamp,:] = ms.cvelfreqs(spwids=[EB],
                                         mode='channel', 
                                         outframe='LSRK',
                                         obstime=str(tstamps[istamp])+'s')

        # close the MS file
        ms.close()

        # append a dataset object to the data dictionary
        data_dict[str(EB)] = dataset(d['u'], 
                                     d['v'], 
                                     d['data'], 
                                     d['weight'],
                                     nu, 
                                     nu_,
                                     tstamp_ID)

    return data_dict

        



def writeMS(datadict, outfile='out.ms'):

    # Check that the original input MS file can be located.
    if not os.path.exists(datadict['input_file']):
        print('I cannot find the input MS file '+datadict['input_file']+\
              ' to make a copy.  Exiting.')
        return

    # Load the basic MS file information
    tb = casatools.table()
    tb.open(datadict['input_file'])
    obs_col = tb.getcol('OBSERVATION_ID')
    SPW_col = tb.getcol('DATA_DESC_ID')
    field_col = tb.getcol('FIELD_ID')
    tb.close()
    EB_ids = np.unique(obs_col)

    # Cycle through EBs to replace information
    pure_files =[]
    for EB in range(datadict['Nobs']):
        # prepare a temporary MS filename
        if datadict['Nobs'] == 1:
            tmp_MS = outfile
            os.system('rm -rf '+outfile)
            os.system('cp -r '+datadict['input_file']+' '+outfile)
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
            casatasks.split(datadict['input_file'], outputvis=tmp_MS, 
                            datacolumn='data', spw=spw_str, field=field_str, 
                            keepflags=False)

        # pack the appropriate dataset into each MS file
        tb.open(tmp_MS, nomodify=False)
        tb.putcol('DATA', datadict[str(EB)].vis)
        tb.putcol('WEIGHT', datadict[str(EB)].wgt)
        tb.flush()
        tb.close()

        # update file lists
        pure_files += [tmp_MS]

    # concatenate if necessary
    if datadict['Nobs'] > 1:
        os.system('rm -rf '+outfile)
        casatasks.concat(pure_files, concatvis=outfile, dirtol='0.1arcsec',
                         copypointing=False)
        [os.system('rm -rf '+i) for i in pure_files]

    return
