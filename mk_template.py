"""
    mk_template.py

    Usage:
    > python mk_template.py <template_name>

    Settings are edited by the user in 'tconfig_<template_name>.py'.

    Outputs:
    - see CASA_scripts/mock_obs.py
"""
import os
import sys
from importlib import import_module

# Parse the template name.

template_name = sys.argv[-1]

# Load template inputs

inp = import_module('tconfig_{}'.format(template_name))

# Set up template storage space (if necessary)

inp.template_dir += '/' if inp.template_dir[-1] != '/' else ''
if not os.path.exists(inp.template_dir):
    os.system('mkdir {}'.format(inp.template_dir))
if not os.path.exists(inp.template_dir+'sims'):
    os.system('mkdir {}sims'.format(inp.template_dir))

# Generate the template

os.system('rm -rf CASA_logs/mock_obs_'+sys.argv[-1]+'.log')
os.system('casa --nologger '
          + '--logfile CASA_logs/mock_obs_{}.log '.format(template_name)
          + '-c CASA_scripts/mock_obs.py {}'.format(template_name))
