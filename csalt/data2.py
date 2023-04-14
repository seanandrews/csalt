import os
import sys
import copy
import numpy as np
import casatasks
import casatools
import scipy.constants as sc


"""
The dataset class for transporting visibility data.
"""
class dataset:

   def __init__(self, um, vm, vis, wgt, nu_TOPO, nu_LSRK, tstamp_ID):

        # Spectral frequencies in Hz units (LSRK for each timestamp)
        self.nu_TOPO = nu_TOPO
        self.nu_LSRK = nu_LSRK

        # Spatial frequencies in meters and lambda units
        self.um = um
        self.vm = vm
        self.ulam = self.um * np.mean(self.nu_TOPO) / sc.c
        self.vlam = self.vm * np.mean(self.nu_TOPO) / sc.c        

        # Visibilities, weights, and timestamp IDs
        self.vis = vis
        self.wgt = wgt
        self.tstamp = tstamp_ID

        # Utility size trackers
        self.npol, self.nchan, self.nvis = vis.shape
        self.nstamps = len(np.unique(tstamp_ID))


"""
Function to read contents of MS file into a data dictionary.
"""
def read_MS(msfile):

    # Make sure the file exists
    if not os.path.exists(msfile):
        print('I cannot find '+msfile+'.  Exiting.')
        sys.exit()

    # Ingest the SPW information into a dictionary
    ms = casatools.ms()
    ms.open(msfile)
    spw_dict = ms.getspectralwindowinfo()
    ms.close()

    # Identify the number of distinct execution blocks
    Nobs = len(spw_dict)

    # Initialize a data dictionary
    data_dict = {'Nobs': Nobs, 'input_file': msfile}

    # Loop over executions to load dataset objects into the data dictionary
    for EB in range(Nobs):
        # Compute the TOPO frequencies
        spw = spw_dict[str(EB)]
        nu = spw['Chan1Freq'] + spw['ChanWidth'] * np.arange(spw['NumChan'])

        # Open the MS file for this EB
        ms.open(msfile)
        ms.selectinit(datadescid=EB)

        # Load the data into a dictionary
        d = ms.getdata(['data', 'weight', 'u', 'v', 'time'])

        # Identify the unique timestamps
        tstamps = np.unique(d['time'])

        # Allocate the timestamp index and LSRK frequency grids
        tstamp_ID = np.empty_like(d['time'])
        nu_ = np.empty((len(tstamps), len(nu)))

        # Loop over the timestamps to populate index and LSRK frequency grids
        for istamp in range(len(tstamps)):
            tstamp_ID[d['time'] == tstamps[istamp]] = istamp
            nu_[istamp,:] = ms.cvelfreqs(spwids=[EB],
                                         mode='channel', 
                                         outframe='LSRK',
                                         obstime=str(tstamps[istamp])+'s')

        # Close the MS file
        ms.close()

        # Append a dataset object to the data dictionary
        data_dict[str(EB)] = dataset(d['u'], 
                                     d['v'], 
                                     d['data'], 
                                     d['weight'],
                                     nu, 
                                     nu_,
                                     tstamp_ID)

    return data_dict



"""
Function to write contents of a data dictionary to MS file 
"""
def write_MS(data_dict, outfile='out.ms', resid=False):

    # Copy the input MS file to the output file
    if not os.path.exists(data_dict['input_file']):
        print('I cannot find the input MS file '+data_dict['input_file']+\
              ' to make a copy.  Exiting.')
        sys.exit()
    else:
        os.system('rm -rf '+outfile)
        os.system('cp -r '+data_dict['input_file']+' '+outfile)

    # Loop over the observations to pack into the MS file
    ms = casatools.ms()
    for EB in range(data_dict['Nobs']):
        # open the MS file for this EB
        ms.open(outfile, nomodify=False)
        ms.selectinit(datadescid=EB)

        # pull the data array
        d = ms.getdata(['data'])

        # replace with the model array or the residuals
        if resid:
            d['data'] -= data_dict[str(EB)].vis
        else:
            d['data'] = data_dict[str(EB)].vis
        ms.putdata(d)

        # close the MS file
        ms.close()

    return None
