import pymcfost as mcfost
import scipy.constants as sc
import numpy as np
from vis_sample.classes import SkyImage
import matplotlib.pyplot as plt


def parametric_disk(velax, pars, pars_fixed, newcube):

    restfreq, FOV, npix, dist, cfg_dict = pars_fixed  # these need to come in somewhere, right now they are manually in the para file
    inc, mass, h, rc, rin, psi, pa, dust_param, vturb = pars

    model = write_run_mcfost(inc, mass, h, rc, rin, psi, pa, dust_param, vturb)

    x = model.pixelscale * (np.arange(model.nx) - model.cx +1)
    y = model.pixelscale * (np.arange(model.ny) - model.cy +1)

    cube = model.lines[:, :, :]

    print(len(x), len(y))

    for_csalt = SkyImage(np.transpose(cube), x, y, model.nu, None)

    return for_csalt



def write_run_mcfost(inclination, stellar_mass, scale_height, r_c, r_in, flaring_exp, PA, dust_param, vturb):
    # Rewrite mcfost para file
    print(inclination, stellar_mass, scale_height, r_c, r_in, flaring_exp, PA, dust_param, vturb)
    updating = mcfost.Params('dmtau.para')
    updating.map.RT_imin = inclination+180
    updating.map.RT_imax = inclination+180
    updating.stars[0].M = stellar_mass
    updating.zones[0].h0 = scale_height
    updating.zones[0].Rc = r_c
    updating.zones[0].Rin = r_in
    updating.zones[0].flaring_exp = flaring_exp
    updating.map.PA = PA
    updating.simu.viscosity = dust_param
    updating.mol.v_turb = vturb
    updating.writeto('dmtau.para')    # Run mcfost
    mcfost.run('dmtau.para', options="-mol -casa -photodissociation", delete_previous=True, logfile='mcfost.log')
    model = mcfost.Line('data_CO/')
    return model
