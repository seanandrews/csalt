import os
execfile('mconfig.py')

# antenna configuration file
cfg_dir = '/pool/asha0/casa-release-5.7.2-4.el7/data/alma/simmos/'
cfg_str = cfg_dir+'alma.cycle7.'+config+'.cfg'

# generate (u,v) tracks
os.chdir('obs_templates/sims/')
simobserve(project=template+'.sim', skymodel='../'+template+'.fits', 
           antennalist=cfg_str, totaltime=ttotal, integration=integ, 
           thermalnoise='', refdate=date, hourangle=HA, mapsize='10arcsec')

# make a template UVFITS file
infile = template+'.sim/'+template+'.sim.alma.cycle7.'+config+'.ms'
exportuvfits(vis=infile, fitsfile='../'+template+'.uvfits',
             datacolumn='data', overwrite=True)
os.chdir('../')
