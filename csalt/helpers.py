import os
import sys
import warnings
import importlib
import datetime
import numpy as np
import scipy.constants as sc
from csalt.keplerian_mask import *
from casatasks import tclean
import casatools


warnings.filterwarnings("ignore")

class dataset:

    def __init__(self, um, vm, vis, wgt, nu_TOPO, nu_LSRK, tstamp_ID):

        # Spectral frequencies in Hz units (LSRK for each timestamp)
        self.nu_TOPO = nu_TOPO
        self.nu_LSRK = nu_LSRK
        self.nchan = len(nu_TOPO)

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
        self.npol, self.nvis = vis.shape[0], vis.shape[2]
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

        # Loop over timestamps to populate index and LSRK frequency grids
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


""" TCLEAN WRAPPER """
def imagecube(msfile, outfile,
              mk_kepmask=True, kepmask_kwargs=None, tclean_kwargs=None,
              dirty_only=False):

    # Populate tclean keywords
    kw = {} if tclean_kwargs is None else tclean_kwargs
    if 'specmode' not in kw: 
        kw['specmode'] = 'cube' 
    if 'datacolumn' not in kw: 
        kw['datacolumn'] = 'data'
    if 'outframe' not in kw: 
        kw['outframe'] = 'LSRK'
    if 'veltype' not in kw: 
        kw['veltype'] = 'radio'
    if 'start' not in kw:
        kw['start'] = '-5.0km/s'
    if 'width' not in kw:
        kw['width'] = '0.16km/s'
    if 'nchan' not in kw:
        kw['nchan'] = 70
    if 'restfreq' not in kw:
        kw['restfreq'] = '230.538GHz'
    if 'imsize' not in kw:
        kw['imsize'] = 512
    if 'cell' not in kw:
        kw['cell'] = '0.02arcsec'
    if 'deconvolver' not in kw:
        kw['deconvolver'] = 'multiscale'
    if 'scales' not in kw:
        kw['scales'] = [0, 10, 30, 50]
    if 'weighting' not in kw:
        kw['weighting'] = 'briggs'
    if 'robust' not in kw:
        kw['robust'] = 0.5
    if 'threshold' not in kw:
        kw['threshold'] = '10mJy'
    if 'restoringbeam' not in kw:
        kw['restoringbeam'] = 'common'
    if 'uvtaper' not in kw:
        kw['uvtaper'] = ''
    if 'niter' not in kw:
        kw['niter'] = 100000
    niter_hold = kw['niter']

    # Prepare workspace by removing old files
    ext = ['image', 'model', 'pb', 'psf', 'residual', 'sumwt']
    [os.system('rm -rf '+outfile+'.'+j) for j in ext]

    # Assign masking
    if mk_kepmask:
        # Prepare workspace by removing old files
        ext = ['image', 'model', 'pb', 'psf', 'residual', 'sumwt', 'mask']
        [os.system('rm -rf '+outfile+'.'+j) for j in ext]

        # First make a dirty image as a guide
        kw['niter'] = 0
        kw['mask']  = ''
        os.system('rm -rf '+outfile+'.mask')
        tclean(msfile, imagename=outfile, **kw)

        # Parse the mask keywords
        mkw = {} if kepmask_kwargs is None else kepmask_kwargs
        if 'inc' not in mkw:
            mkw['inc'] = 40.
        if 'PA' not in mkw:
            mkw['PA'] = 130.
        if 'mstar' not in mkw:
            mkw['mstar'] = 1.0
        if 'vlsr' not in mkw:
            mkw['vlsr'] = 0.0e3
        if 'dist' not in mkw:
            mkw['dist'] = 150.
        _ = make_kepmask(outfile+'.image', **mkw)

        # Now assign the mask
        os.system('rm -rf '+outfile+'.kep.mask')
        os.system('mv '+outfile+'.mask.image '+outfile+'.kep.mask')
        kw['mask'] = outfile+'.kep.mask'

    else:
        kw['mask'] = kw.pop('mask', '') 

    if not dirty_only:
        # Prepare workspace by removing old files
        ext = ['image', 'model', 'pb', 'psf', 'residual', 'sumwt', 'mask']
        [os.system('rm -rf '+outfile+'.'+j) for j in ext]

        # Image
        kw['niter'] = niter_hold
        tclean(msfile, imagename=outfile, **kw)

    return None



def LST_to_UTC(date, LST, longitude):

    # Load the measures and quanta tools
    me = casatools.measures()
    qa = casatools.quanta()

    # Parse input LST into hours
    h, m, s = LST.split(LST[2])
    LST_hours = int(h) + int(m) / 60. + float(s) / 3600.

    # Calculate the MJD
    mjd = me.epoch('utc', date)['m0']['value']
    jd = mjd + 2400000.5
    T = (jd - 2451545.0) / 36525.0
    sidereal = 280.46061837 + 360.98564736629 * (jd - 2451545.) \
               + 0.000387933*T**2 - (1. / 38710000.)*T**3
    sidereal += longitude
    sidereal /= 360.
    sidereal -= np.floor(sidereal)
    sidereal *= 24.0
    if (sidereal < 0): sidereal += 24
    if (sidereal >= 24): sidereal -= 24
    delay = (LST_hours - sidereal) / 24.0
    if (delay < 0.0): delay += 1.0
    mjd += delay / 1.002737909350795

    # Convert to UTC date/time string
    today = me.epoch('utc', 'today')
    today['m0']['value'] = mjd
    hhmmss = qa.time(today['m0'], form='', prec=6, showform=False)[0]
    dt = qa.splitdate(today['m0'])
    ut = '%s%s%02d%s%02d%s%s' % \
         (dt['year'], '/', dt['month'], '/', dt['monthday'], '/', hhmmss)

    return ut


def doppler_set(nu_rest, vel_tune, datestring, RA, DEC,
                equinox='J2000', observatory='ALMA'):

    # Load the measures and quanta tools
    me = casatools.measures()

    # Set direction and epoch
    position = me.direction(equinox, RA, DEC.replace(':', '.'))
    obstime = me.epoch('utc', datestring)

    # Define the radial velocity
    rvel = me.radialvelocity('LSRK', str(vel_tune * 1e-3)+'km/s')

    # Define the observing frame
    me.doframe(position)
    me.doframe(me.observatory(observatory))
    me.doframe(obstime)
    me.showframe()

    # Convert to the TOPO frame
    rvel_top = me.measure(rvel, 'TOPO')
    dopp = me.todoppler('RADIO', rvel_top)
    nu_top = me.tofrequency('TOPO', dopp,
                            me.frequency('rest', str(nu_rest / 1e9)+'GHz'))

    return nu_top['m0']['value']



