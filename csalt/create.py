import os
import sys
import datetime
import importlib
import numpy as np
import scipy.constants as sc
from casatasks import (simobserve, concat)
import casatools
import warnings

# Load the measures and quanta tools
me = casatools.measures()
qa = casatools.quanta()

"""
The create class
"""
class create:

    def __init__(self, quiet=True):

        if quiet:
            warnings.filterwarnings("ignore")


    def LST_to_UTC(self, date, LST, longitude):
        
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


    def doppler_set(self, nu_rest, vel_tune, datestring, RA, DEC, 
                    equinox='J2000', observatory='ALMA'):

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


    """ Create a blank MS template """
    def template_MS(self, config, t_total, msfile, sim_save=False,
                    RA='16:00:00.00', DEC='-30:00:00.00', nu_rest=230.538e9, 
                    dnu_native=122e3, V_span=10e3, V_tune=0.0e3,
                    t_integ='6s', HA_0='0h', date='2023/03/20',
                    observatory='ALMA'):

        # Parse / determine the executions
        if np.isscalar(config): config = np.array([config])
        if np.isscalar(t_total): t_total = np.array([t_total])
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

            # Calculate the LST starting time of the execution block
            h, m, s = RA.split(':')
            LST_h = int(h) + int(m)/60 + float(s)/3600 + float(HA_0[i][:-1])
            LST_0 = str(datetime.timedelta(hours=LST_h))
            if (LST_h < 10.): LST_0 = '0' + LST_0

            # Get the observatory longitude
            obs_long = np.degrees(me.observatory(observatory)['m0']['value'])

            # Calculate the UT starting time of the execution block
            UT_0 = self.LST_to_UTC(date[i], LST_0, obs_long)

            # Calculate the TOPO tuning frequency
            nu_tune_0 = self.doppler_set(nu_rest, V_tune[i], UT_0, RA, DEC, 
                                         observatory=observatory)

            # Generate a dummy (empty) cube
            ia = casatools.image()
            dummy = ia.makearray(v=0.001, shape=[64, 64, 4, nch])
            ia.fromarray(outfile='dummy.image', pixels=dummy, overwrite=True)
            ia.done()

            # Compute the midpoint HA
            if (t_total[i][-1] == 'h'):
                tdur = float(t_total[i][:-1])
            elif (t_total[i][-3:] == 'min'):
                tdur = float(t_total[i][:-3]) / 60
            elif (t_total[i][-1] == 's'):
                tdur = float(t_total[i][:-1]) / 3600
            HA_mid = str(float(HA_0[i][:-1]) + 0.5 * tdur) +'h'

            # Generate the template sub-MS file
            simobserve(project=out_name[:-3]+'_'+str(i)+'.sim',
                       skymodel='dummy.image', 
                       antennalist=config[i], 
                       totaltime=t_total[i], 
                       integration=t_integ[i],
                       thermalnoise='', 
                       hourangle=HA_mid, 
                       indirection='J2000 '+RA+' '+DEC, 
                       refdate=date[i],
                       incell='0.01arcsec', 
                       mapsize='5arcsec',
                       incenter=str(nu_tune_0 / 1e9)+'GHz',
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
