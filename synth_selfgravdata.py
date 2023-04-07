import os, sys, importlib
from csalt.synthesize import make_template, make_data
from csalt.utils import *
sys.path.append('configs/')
from parametric_disk_RADMC3D2 import parametric_disk as pardisk_radmc
from csalt.models import cube_to_fits


#cfg = ['sg_sharpchi', 'sg_taper2hi', 'sg_taper2hi_kep', 'sg_taper2hi_prs',
#       'sg_taper2hi_sg', 'sg_taper1hi', 'sg_taper3hi']

#cfg = ['sg_taper2hi_M15', 'sg_taper2hi_M05']

#cfg = ['sg_taper2hi_kep', 'sg_taper2hi_prs', 'sg_taper2hi_sg']
#cfg = ['sg_modelc', 'sg_modela', 'sg_modelc']
cfg = ['sg_modeld']

do_raw = True
raw_tau_only = False
do_sim = False
do_img = False
templ = False


# Loop through and create data, image the pure and noisy cubes
for i in range(len(cfg)):

    # load config file
    inp = importlib.import_module('gen_'+cfg[i])

    if do_sim:
        # impose a model onto the synthetic tracks
        make_data(cfg[i], mtype='RADMC3D', calctype='def', new_template=templ)

        # clean-up for space-saving
        os.system('rm -rf '+inp.synthraw_dir+inp.basename+'/*')
        os.system('rm -rf '+inp.reduced_dir+inp.basename+'/subMS')

    if do_img:
        # image the cubes
        img_cube(inp.reduced_dir+inp.basename+'/'+inp.basename+'_noisy.DATA',
                 inp.reduced_dir+inp.basename+'/images/'+ \
                 inp.basename+'_noisy.DATA', 'gen_'+cfg[i], masktype='kep')

        img_cube(inp.reduced_dir+inp.basename+'/'+inp.basename+'_pure.DATA',
                 inp.reduced_dir+inp.basename+'/images/'+ \
                 inp.basename+'_pure.DATA', 'gen_'+cfg[i], masktype='kep')

    if do_raw:
        # make a "raw" cube calculated on the output pure/noisy cube velocities
        # channel setups
        velax = 1e3 * float(inp.chanstart[:-4]) + \
                1e3 * float(inp.chanwidth[:-4]) * np.arange(inp.nchan_out)
        fixed = inp.nu_rest, inp.FOV[0], inp.Npix[0], inp.dist, inp.cfg_dict

        # tau surface only?
#        if raw_tau_only:
        taur = pardisk_radmc(velax, inp.pars, fixed, tausurf=True)

        sys.exit()



        # make a "raw" cube calculation on the pure/noisy cube velocities
#        else:
        os.system('mv '+inp.radmcname+'image.out '+ \
                  inp.radmcname+'oimage.out')
        cuber = pardisk_radmc(velax, inp.pars, fixed)
        cube_to_fits(cuber, inp.radmcname+'raw_cube.fits', 
                     RA=inp.RAdeg, DEC=inp.DECdeg)
        os.system('mv '+inp.radmcname+'image.out '+ \
                  inp.radmcname+'raw_image.out')
        os.system('mv '+inp.radmcname+'oimage.out '+ \
                  inp.radmcname+'image.out')
