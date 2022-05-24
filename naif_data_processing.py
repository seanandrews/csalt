import os
import numpy as np
execfile('/pool/asha0/SCIENCE/csalt/csalt/CASA_scripts/ms_to_hdf5.py')

idir = 'storage/data/fiducial_hires/'
ifil = 'fiducial_hires_pure.DATA'
ofil = 'fiducial_hires_pure_naif_natv.DATA'

start = '-5.00km/s'
width = '0.08km/s'
nchan = 250

restfreq = '230.538GHz'


# MSTRANSFORM to regular velocity channels
os.system('rm -rf '+idir+ofil+'.ms')
mstransform(vis=idir+ifil+'.ms', outputvis=idir+ofil+'.ms', datacolumn='data',
            keepflags=False, regridms=True, mode='velocity', outframe='LSRK',
            start=start, width=width, nchan=nchan, restfreq=restfreq)

# make HDF5 file (***needs adjusting if more than 1 EB)
ms_to_hdf5(idir+ofil, idir+ofil, groupname='EB0/')
