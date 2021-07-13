"""
    mk_template.py

    Usage: 
        > python mk_template.py <template_name>

    Settings are edited by the user in 'tconfig_<template_name>.py'.

    Outputs:
	- see CASA_scripts/mock_obs.py
"""
import os, sys, time, importlib
import numpy as np
import const as const


# Load template inputs
inp = importlib.import_module('tconfig_'+sys.argv[-1])


# Set up template storage space (if necessary)
if inp.template_dir[-1] != '/': inp.template_dir += '/'
if not os.path.exists(inp.template_dir):
    os.system('mkdir '+inp.template_dir)
    os.system('mkdir '+inp.template_dir+'sims')
elif not os.path.exists(inp.template_dir+'sims'):
    os.system('mkdir '+inp.template_dir+'sims')


# Generate the template
os.system('rm -rf CASA_logs/mock_obs_'+sys.argv[-1]+'.log')
os.system('casa --nologger --logfile CASA_logs/mock_obs_'+sys.argv[-1]+\
          '.log -c CASA_scripts/mock_obs.py '+sys.argv[-1])
