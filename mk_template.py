"""
A script to generate an observational template -- a set of (u, v) tracks -- to 
use in generating synthetic observations.

The settings are controlled by the user in 'tconfig_<template_name>.py'.

Usage: 
	> python mk_template.py <template_name>
"""

import os, sys, importlib


# Load template inputs
template_name = sys.argv[-1]
inp = importlib.import_module('tconfig_'+template_name)


# Set up template storage space (if necessary)
if inp.template_dir[-1] != '/': inp.template_dir += '/'
if not os.path.exists(inp.template_dir):
    os.system('mkdir '+inp.template_dir)
    os.system('mkdir '+inp.template_dir+'sims')
elif not os.path.exists(inp.template_dir+'sims'):
    os.system('mkdir '+inp.template_dir+'sims')


# Generate the template
os.system('rm -rf CASA_logs/mock_obs_'+template_name+'.log')
os.system('casa --nologger --logfile CASA_logs/mock_obs_'+template_name+\
          '.log -c CASA_scripts/mock_obs.py '+template_name)
