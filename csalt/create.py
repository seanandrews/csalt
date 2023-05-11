import os
import sys
import importlib
import numpy as np
import scipy.constants as sc
sys.path.append("/home/sandrews/casatools/analysis_scripts/")
import analysisUtils as au
from casatasks import (simobserve, concat)
import casatools
import warnings

"""
The create class
"""
class create:

    def __init__(self, path=None, quiet=True):

        if quiet:
            warnings.filterwarnings("ignore")

        self.path = path


    """ Create a blank MS template """
    def template_MS(self, config, t_total, msfile, sim_save=False,
                    RA='16:00:00.00', DEC='-30:00:00.00', nu_rest=230.538e9, 
                    dnu_native=122e3, V_span=10e3, V_tune=0.0e3,
                    t_integ='6s', HA_0='0h', date='2023/03/20',
                    XXX=0):

        # Parse / determine the executions
        if np.isscalar(config): config = np.array([config])
        if np.isscalar(t_total): t_integ = np.array([t_total])
        Nobs = len(config)

        # things to format check:
		# RA, DEC have full hms/dms string formatting
		# HA_0 has proper string formatting
		# date has proper string formatting
		# msfile has proper '.ms' ending

        # If only scalars specified for keywords, copy them for each execution
        if np.isscalar(dnu_native): dnu_native = np.repeat(dnu_native, Nobs)
        if np.isscalar(V_span): V_span = np.repeat(V_span, Nobs)
        if np.isscalar(V_tune): V_tune = np.repeat(V_tune, Nobs)
        if np.isscalar(HA_0): HA_0 = np.repeat(HA_0, Nobs)
        if np.isscalar(date): date = np.repeat(date, Nobs)
        if np.isscalar(t_integ): t_integ = np.repeat(t_integ, Nobs)

        # Move to simulation space
        cwd = os.getcwd()
        out_path, out_name = os.path.split(msfile)
        if out_path != '': 
            if not os.path.exists(out_path): os.system('mkdir '+out_path)
            os.chdir(out_path)

        # Loop over execution blocks
        obs_files = []
        for i in range(Nobs):

            # Calculate the number of native channels
            nch = 2 * int(V_span[i] / (sc.c * dnu_native[i] / nu_rest)) + 1

            # Convert RA, DEC coordinates into degrees
            RA_  = np.array([float(RA.split(':')[j]) for j in range(3)])
            DEC_ = np.array([float(DEC.split(':')[j]) for j in range(3)])
            dms = np.array([1, 60, 3600])
            RAdeg, DECdeg = 15 * np.sum(RA_ / dms), np.sum(DEC_ / dms)

            # Calculate the UT starting time of the execution block
            LST_0 = au.hoursToHMS(RAdeg / 15 + float((HA_0[i])[:-1]))
            UT_0  = au.lstToUT(LST_0, date[i])
            dUT_0 = UT_0[0][:-3].replace('-', '/').replace(' ', '/')

            # Calculate the TOPO frequency = tuning velocity (SPW center)
            nu_tune = au.restToTopo(nu_rest, 1e-3 * V_tune[i], dUT_0, RA, DEC)

            # Generate a dummy (empty) cube
            ia = casatools.image()
            dummy = ia.makearray(v=0.001, shape=[64, 64, 4, nch])
            ia.fromarray(outfile='dummy.image', pixels=dummy, overwrite=True)
            ia.done()

            # Generate the template sub-MS file
            simobserve(project=out_name[:-3]+'_'+str(i)+'.sim',
                       skymodel='dummy.image', 
                       antennalist=config[i], 
                       totaltime=t_total[i], 
                       integration=t_integ[i],
                       thermalnoise='', 
                       hourangle=HA_0[i], 
                       indirection='J2000 '+RA+' '+DEC, 
                       refdate=date[i],
                       incell='0.01arcsec', 
                       mapsize='5arcsec',
                       incenter=str(nu_tune / 1e9)+'GHz',
                       inwidth=str(dnu_native[i] * 1e-3)+'kHz', 
                       outframe='TOPO')

            # Pull the sub-MS file out of the simulation directory
            cfg_dir, cfg_file = os.path.split(config[i])
            sim_MS = out_name[:-3]+'_'+str(i)+'.sim/'+out_name[:-3]+ \
                     '_'+str(i)+'.sim.'+cfg_file[:-4]+'.ms'
            os.system('rm -rf '+out_name[:-3]+'_'+str(i)+'.ms*')
            os.system('mv '+sim_MS+' '+out_name[:-3]+'_'+str(i)+'.ms')

            # Delete the simulation directory if requested
            if not sim_save: 
                os.system('rm -rf '+out_name[:-3]+'_'+str(i)+'.sim')

            # Delete the dummy (empty) cube
            os.system('rm -rf dummy.image')

            # Update the file list
            obs_files += [out_name[:-3]+'_'+str(i)+'.ms']

        # Concatenate the sub-MS files into a single MS
        os.system('rm -rf '+out_name[:-3]+'.ms*')
        if Nobs > 1:
            concat(vis=obs_files, 
                   concatvis=out_name[:-3]+'.ms',
                   dirtol='0.1arcsec',
                   copypointing=False)
        else: 
            os.system('mv '+out_name[:-3]+'_0.ms '+out_name[:-3]+'.ms')

        # Clean up
        os.system('rm -rf '+out_name[:-3]+'_*.ms*')
        os.chdir(cwd)

        return None
