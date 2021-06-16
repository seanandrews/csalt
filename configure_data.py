import os, sys
import numpy as np


# Set project name
project_name = 'Sz129'


# Make sure the configuration file exists
if not os.path.isfile('fconfig_'+project_name+'.py'):
    print('The proper configuration file does not exist...exiting.')
    sys.exit()

# Copy the configuration file to general name and import it.
os.system('cp fconfig_'+project_name+'.py fconfig.py')
import fconfig as inp


""" 
Run the CASA script to parse the original MS to individual EBs.  This will save
the outcomes into individual .npz files in data/project_name (among other 
things).  Here, we'll gather them up into a data dictionary and pickle it as 
the de facto data product for subsequent use.
"""
clog = '--logfile CASA_logs/format_datafiles_'+project_name+'.log'
os.system('casa --nologger '+clog+' -c CASA_scripts/format_datafiles.py')



