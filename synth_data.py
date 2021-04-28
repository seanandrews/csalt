"""
decription tbd

"""

import os, sys, time
import numpy as np
import const as c_
import config as in_


# Generate a synthetic observation template
if in_.gen_uv:

    # Set the channels in frequency and LSRK velocity domains
    nu_span = in_.restfreq * (in_.vsys - in_.vspan) / c_.c
    nu_sys  = in_.restfreq * (1 - in_.vsys / c_.c)
    nch = np.int(2 * np.abs(nu_span) / (in_.dfreq0 / in_.sosampl) + 1)
    freq = nu_sys - nu_span - (in_.dfreq0 / in_.sosampl) * np.arange(nch)
    vel = c_.c * (1. - freq / in_.restfreq)

    # Parse the target coordinates into degrees
    RA_pieces = [np.float(in_.RA.split(':')[i]) for i in np.arange(3)]
    RAdeg = 15 * np.sum(np.array(RA_pieces) / [1., 60., 3600.])
    DEC_pieces = [np.float(in_.DEC.split(':')[i]) for i in np.arange(3)]
    DECdeg = np.sum(np.array(DEC_pieces) / [1., 60., 3600.])

    # Make a dummy template FITS cube for CASA simulations
    if in_.sosampl > 1:
        outfile = tname+str(in_.sosampl)+'x'
    else: outfile = tname
    cube_parser(dist=150., r_max=300., r_l = 300., FOV=8.0, Npix=256,
                restfreq=restfreq, RA=RAdeg, DEC=DECdeg, Vsys=Vsys, vel=vel*1e3,
                outfile='template_cubes/'+outfile+'.fits')
 









