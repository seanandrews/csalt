import os, sys, time
import numpy as np
import scipy.constants as sc
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.special import ellipk, ellipe
from astropy.io import ascii
import matplotlib.pyplot as plt
import warnings

# constants (in cgs units)
_msun = 1.989e33
_AU = sc.au * 1e2
_mu = 2.37
_mH = (sc.m_e + sc.m_p) * 1e3
_k  = sc.k * 1e7
_G  = sc.G * 1e3
_pc = 3.09e18


class radmc_setup:

    def __init__(self, cfg_dict, writegrid=True):

        # parse the cfg_dict dictionary arguments
        self.modelname = cfg_dict["radmcname"]
        self.gridpars = cfg_dict["grid_params"]
        self.setup = cfg_dict["setup_params"]

        # populate the spatial grids
        self._read_spatial_grid(self.gridpars["spatial"])

        # populate the wavelength grid
        if "wavelength" not in self.gridpars:
            self.gridpars["wavelength"] = self.gridpars.pop("wavelength", {})
        self._read_wavelength_grid(self.gridpars["wavelength"])

        # write out the grids into RADMC3D formats
        if writegrid:
            if not os.path.exists(self.modelname): os.mkdir(self.modelname)
            self.write_wavelength_grid()
            self.write_spatial_grid()
            self.write_config_files()

        # note number of dust species
        if self.setup["incl_dust"]:
            if self.setup["dust"]["form"] == 'composite':
                self.ndust = 1

            if self.setup["dust"]["form"] == 'multi':
                # load index to size correspondance
                dind, dsize = np.loadtxt('opacs/'+self.setup["dustspec"] + \
                                         '_sizeindex.txt').T

                # find index for maximum dust size
                amax = self.setup["dust"]["arguments"]["amax"]
                dindmax = np.max(dind[dsize <= amax])

                # queue up dust species
                self.ndust = np.int(dindmax + 1)


    def _read_spatial_grid(self, args, refine=False):
        self.nr, self.nt = args["nr"], args["nt"]
        self.np = args.pop("np", 1)

        # radial grid in [cm]
        self.r_in  = args["r_min"] * _AU
        self.r_out = args["r_max"] * _AU
        self.r_walls = np.logspace(np.log10(self.r_in), np.log10(self.r_out),
                                   self.nr+1)
        self.r_centers = np.average([self.r_walls[:-1], self.r_walls[1:]],
                                    axis=0)
        assert self.r_centers.size == self.nr

        # number of cells
        self.ncells = self.nr * self.nt * self.np

        # theta (altitude angle from pole toward equator) grid in [rad]
        self.t_offset = args.get("t_offset", 0.1)
        self.t_min = args.get("t_min", 0.0) + self.t_offset
        self.t_max = args.get("t_max", 0.5 * np.pi) + self.t_offset
        self.t_walls = np.logspace(np.log10(self.t_min),
                                   np.log10(self.t_max), self.nt+1)
        self.t_walls = 0.5 * np.pi + self.t_offset - self.t_walls[::-1]
        self.t_min = self.t_walls.min()
        self.t_max = self.t_walls.max()
        self.t_centers = np.average([self.t_walls[:-1], self.t_walls[1:]],
                                    axis=0)
        assert self.t_centers.size == self.nt

        # phi (azimuth angle) grid in [rad]
        self.p_min = args.get("p_min", 0.0)
        self.p_max = args.get("p_max", 0.0)
        self.p_walls = np.linspace(self.p_min, self.p_max, self.np + 1)
        self.p_centers = np.average([self.p_walls[:-1], self.p_walls[1:]],
                                    axis=0)
        assert self.p_centers.size == self.np

    
    def _read_wavelength_grid(self, params):
        self.nw = params.get("nw", 100)
        self.logw_min = params.get("logw_min", -1.0)
        self.logw_max = params.get("logw_max", 4.0)
        self.w_centers = np.logspace(self.logw_min, self.logw_max, self.nw)


#    def _make_starspec(self, params):
#        teff, lstar, mstar = params["T_eff"], params["L_star"], params["M_star"]
#        swl, sfnu = stellarspectrum(params["T_eff"], params["L_star"],
#                                    mstar=params["M_star"])
#        sint = interp1d(swl, sfnu)
#        self.Fnu_star = sint(self.w_centers)


#    def write_starspec(self, fileout='stars.inp'):
#        header = '2\n1    {:d}\n'.format(self.nw)
#        rstar = np.sqrt(self.hostpars["L_star"] * self.lsun / \
#                        (4 * np.pi * self.sigSB * self.hostpars["T_eff"]**4))
#        header += '%.6e   %.6e   0.   0.   0.' % \
#                  (rstar, self.hostpars["M_star"] * self.msun)
#        tosave = np.concatenate((self.w_centers, self.Fnu_star))
#        np.savetxt(self.modelname + '/' + fileout, tosave, header=header,
#                   comments='')


    def write_wavelength_grid(self, fileout='wavelength_micron.inp'):
        np.savetxt(self.modelname + '/' + fileout, self.w_centers,
                   header=str(self.nw) + '\n', comments='')


    def write_spatial_grid(self, fileout='amr_grid.inp'):
        """ Write the spatial grid to file """
        header = '1\n0\n100\n0\n1 1 0\n'
        header += '{:d} {:d} {:d}'.format(self.nr, self.nt, self.np)
        tosave = np.concatenate([self.r_walls, self.t_walls, self.p_walls])
        np.savetxt(self.modelname + '/' + fileout, tosave, header=header,
                   comments='')


    def write_config_files(self, fileout='radmc3d.inp'):

        """ Write the RADMC3D configuration files """
        # open file
        f = open(self.modelname + '/' + fileout, 'w')

        # spectral line, continuum, or both
        f.write('incl_dust = %d\n' % self.setup["incl_dust"])
        f.write('incl_lines = %d\n' % self.setup["incl_lines"])
        f.write('incl_freefree = %d\n' % self.setup.pop("incl_freefree", 0))
        f.write('nphot = %d\n' % self.setup.pop("nphot", 1000000))
        f.write('modified_random_walk = %d\n' % self.setup.pop("mrw", 1))
        f.write('mc_scat_maxtauabs = 5.d0\n')

        # treatment of (continuum) scattering
        if self.setup["scattering"] == 'None':
            f.write('scattering_mode_max= %d \n' % 0)
        elif self.setup["scattering"] == 'Isotropic':
            f.write('scattering_mode_max= %d\n' % 1)
            f.write('nphot_scat=2000000\n')
        elif self.setup["scattering"] == 'HG':
            f.write('scattering_mode_max = %d \n' % 2)
            f.write('nphot_scat=10000000\n')
        elif self.setup["scattering"] == 'Mueller':
            f.write('scattering_mode_max = %d \n' % 3)
            f.write('nphot_scat=100000000\n')

        # select ascii or binary storage
        if "binary" not in self.setup: self.setup["binary"] = False
        if self.setup["binary"]:
            f.write('writeimage_unformatted = 1\n')
            f.write('rto_single = 1\n')
            f.write('rto_style = 3\n')
        else:
            f.write('rto_style = 1\n')

        # raytrace intensities or optical depths
        if "camera_tracemode" not in self.setup:
            self.setup["camera_tracemode"] = 'image'
        if self.setup["camera_tracemode"] == 'image':
            f.write('camera_tracemode = 1\n')
        elif self.setup["camera_tracemode"] == 'tau':
            f.write('camera_tracemode = -2\n')


        # LTE excitation calculations (hard-coded)
        f.write('lines_mode = 1\n')

        f.close()


        ### DUST CONFIG FILE
        if (self.setup["incl_dust"] == 1):
            if self.setup["dust"]["form"] == 'composite':
                self.ndust = 1
                f = open(self.modelname + '/' + 'dustopac.inp', 'w')
                f.write('2\n1\n')
                f.write('============================================================================\n')
                f.write('1\n0\n')
                f.write('%s\n' % self.setup["dustspec"])
                f.write('----------------------------------------------------------------------------')
                f.close()

                # copy appropriate opacity file
                os.system('cp opacs/dustkappa_' + self.setup["dustspec"] + \
                          '.inp ' + self.modelname + '/')

            if self.setup["dust"]["form"] == 'multi':
                # load index to size correspondance
                dind, dsize = np.loadtxt('opacs/' + self.setup["dustspec"] + \
                                         '_sizeindex.txt').T

                # find index for maximum dust size
                amax = self.setup["dust"]["arguments"]["amax"]
                dindmax = np.max(dind[dsize <= amax])

                # queue up dust species
                self.ndust = np.int(dindmax + 1)
                idust = [str(ix).rjust(2, '0') for ix in range(self.ndust)]

                # generate control file
                dbarr = '============================================================================'
                sbarr = '----------------------------------------------------------------------------'
                f = open(self.modelname + '/' + 'dustopac.inp', 'w')
                f.write('2\n')
                f.write(str(self.ndust)+'\n')
                f.write(dbarr+'\n')
                for ix in range(self.ndust):
                    f.write('1\n0\n')
                    f.write('%s\n' % (self.setup["dustspec"] + '_' + idust[ix]))
                    f.write(sbarr)
                    if (ix < self.ndust-1): f.write('\n')
                    os.system('cp opacs/dustkappa_' + \
                              self.setup["dustspec"] + '_' + idust[ix] + \
                              '.inp ' + self.modelname + '/')
                f.close()


        ### LINE DATA CONFIG FILE
        if (self.setup["incl_lines"] == 1):
            f = open(self.modelname + '/lines.inp', 'w')
            f.write('2\n1\n')
            f.write('%s    leiden    0    0    0' % self.setup["molecule"])
            f.close()

            # copy appropriate molecular data file
            os.system('cp moldata/' + self.setup["molecule"]+'.dat ' + \
                      self.modelname + \
                      '/molecule_' + self.setup["molecule"]+'.inp')







class radmc_structure:

    def __init__(self, cfg_dict, func_temperature, func_sigma, func_omega,
                 func_abund, func_nonthermal_linewidth, grid=None, 
                 writestruct=True):

        # if no grid passed, make one
        if grid is None:
            grid = radmc_setup(cfg_dict, writegrid=writestruct)

        # parse the cfg_dict dictionary arguments
        self.modelname = cfg_dict["radmcname"]
        self.gridpars = cfg_dict["grid_params"]
        self.setup = cfg_dict["setup_params"]
        self.do_vprs = cfg_dict["dPdr"]
        self.do_vsg = cfg_dict["selfgrav"]
        self.func_temperature = func_temperature
        self.func_sigma = func_sigma
        self.func_omega = func_omega
        self.func_abund = func_abund
        self.func_nonthermal_linewidth = func_nonthermal_linewidth

        # parse the grid
        # (spherical coordinate system)
        self.rvals, self.tvals = grid.r_centers, grid.t_centers
        self.nr, self.nt = grid.nr, grid.nt
        self.rr, self.tt = np.meshgrid(self.rvals, self.tvals)

        # corresponding cylindrical quantities
        self.rcyl = self.rr * np.sin(self.tt)
        self.zcyl = self.rr * np.cos(self.tt)

        # default header for outputs
        self.hdr = '1\n%d' % (self.nr * self.nt)

        # passable file name
        self.smol = self.setup["molecule"]


# ----
        # calculate the gas temperature structure
        self.set_temperature()

        # calculate the gas density structure
        self.set_density()

        # set molecular number density structure
        self.set_nmol()

        # set line-width structure
        self.set_turbulence()

        # set gas velocity structure
        self.set_vgas()




    def set_temperature(self, write=True):
        """
        set the temperature structure
        """
        self.temperature = self.func_temperature(self.rcyl, self.zcyl)

        if self.setup["incl_lines"]:
            np.savetxt(self.modelname+'/gas_temperature.inp',
                       np.ravel(self.temperature), fmt='%.6e',
                       header=self.hdr, comments='')
        if self.setup["incl_dust"]:
            np.savetxt(self.modelname+'/dust_temperature.dat',
                       np.ravel(self.temperature), fmt='%.6e',
                       header=self.hdr+'\n1', comments='')


    def set_density(self, write=True, _min=1., _max=None):
        """
        calculate the 2-d density structure in vertical hydrostatic equilibrium
        """
        # loop through cylindrical coordinates (ugh)
        self.rho_gas = np.zeros((self.nt, self.nr))
        for i in range(self.nr):
            for j in range(self.nt):
                # cylindrical coordinates
                r, z = self.rcyl[j,i], self.zcyl[j,i]

                # define a special z grid for integration (zg)
                zmin, zmax, nz = 0.1, 5.*r, 1024
                zg = np.logspace(np.log10(zmin), np.log10(zmax + zmin), nz)
                zg -= zmin

                # if z >= zmax, return the minimum density
                if (z >= zmax):
                    self.rho_gas[j,i] = _min * _mH * _mu
                else:
                    # vertical temperature gradient
                    Tz = self.func_temperature(r, zg)
                    dT, dz = np.diff(np.log(Tz)), np.diff(zg)
                    dlnTdz = np.append(dT, dT[-1]) / np.append(dz, dz[-1])

                    # vertical gravity
                    gz = self.func_omega(r, zg)**2 * zg
                    gz /= (_k * Tz / (_mu * _mH))

                    # vertical density gradient
                    dlnpdz = -dlnTdz - gz

                    # numerical integration
                    lnp = integrate.cumtrapz(dlnpdz, zg, initial=0)
                    rho0 = np.exp(lnp)

                    # normalize
                    rho = 0.5 * rho0 * self.func_sigma(r)
                    rho /= integrate.trapz(rho0, zg)

                    # interpolator to go back to the original gridpoint
                    f = interp1d(zg, rho)

                    # gas density at specified height
                    min_rho = _min * _mu * _mH
                    self.rho_gas[j,i] = np.max([f(z), min_rho])

        if write:
            np.savetxt(self.modelname+'/gas_density.inp', 
                       np.ravel(self.rho_gas), fmt='%.6e', header=self.hdr, 
                       comments='')


    def set_nmol(self, write=True):

        abund_ = self.func_abund(self.rcyl, self.zcyl) 
        self.nmol = abund_ * self.rho_gas / (_mu * _mH)

        if write:
            np.savetxt(self.modelname+'/numberdens_'+self.smol+'.inp',
                       np.ravel(self.nmol), fmt='%.6e', header=self.hdr, 
                       comments='')


    def set_turbulence(self, write=True):

        self.dVturb = self.func_nonthermal_linewidth(self.rcyl, self.zcyl)

        if write:
            np.savetxt(self.modelname+'/microturbulence.inp',
                       np.ravel(self.dVturb), fmt='%.6e', header=self.hdr,
                       comments='')


    def set_vgas(self, write=True):
        
        # rotation
        vkep2 = (self.func_omega(self.rcyl, self.zcyl) * self.rcyl)**2

        # pressure
        if self.do_vprs:
            P = self.rho_gas * _k * self.temperature / _mH / _mu
            dP = np.gradient(P, self.rvals, axis=1) * np.sin(self.tt) + \
                 np.gradient(P, self.tvals, axis=0) * np.cos(self.tt) / self.rr
            vprs2 = self.rr * np.sin(self.tt) * dP / self.rho_gas
            vprs2 = np.where(np.isfinite(vprs2), vprs2, 0.0)
        else:
            vprs2 = 0.

        # self-gravity
        if self.do_vsg:
            rp = np.logspace(np.log10(self.rvals[0]), 
                             np.log10(3*self.rvals[-1]), 4096)
            kk = 4 * rp[None,:] * self.rcyl[:,:,None] / \
                 ((self.rcyl[:,:,None] + rp[None,:])**2 + \
                  self.zcyl[:,:,None]**2)
            br = ellipk(np.sqrt(kk)) - 0.25 * (kk / (1. - kk)) * \
                 ((rp[None,:] / self.rcyl[:,:,None]) - \
                  (self.rcyl[:,:,None] / rp[None,:]) + \
                  (self.zcyl[:,:,None]**2/(self.rcyl[:,:,None]*rp[None,:]))) * \
                 ellipe(np.sqrt(kk))
            integ = br * np.sqrt(rp[None,:] / self.rcyl[:,:,None]) * \
                    np.sqrt(kk) * self.func_sigma(rp[None,:])
            vsg2 = _G * np.trapz(integ, rp, axis=-1)
        else:
            vsg2 = 0.

        # combined azimuthal velocity profile
        vtot2 = vkep2 + vprs2 + vsg2
        vtot2[vtot2 < 0] = 0.
        self.vphi = np.sqrt(vtot2)

        if write:
            vgas = np.ravel(self.vphi)
            foos = np.zeros_like(vgas)
            np.savetxt(self.modelname+'/gas_velocity.inp',
                       list(zip(foos, foos, vgas)),
                       fmt='%.6e', header=self.hdr, comments='')


    def get_cube(self, inc, PA, dist, nu_rest, FOV, Npix, velax=[0], vlsr=0):

        # position angle convention
        posang = PA - 90.

        # spatial settings
        sizeau = FOV * dist

        # frequency settings
        wlax = 1e6 * sc.c / (nu_rest * (1. - velax / sc.c))
        os.system('rm -rf '+self.modelname+'/camera_wavelength_micron.inp')
        np.savetxt(self.modelname+'/camera_wavelength_micron.inp', wlax,
                   header=str(len(wlax))+'\n', comments='')

        # run the raytracer
        cwd = os.getcwd()
        os.chdir(self.modelname)
        os.system('rm -rf image.out')
        os.system('radmc3d image '+ \
                  'incl %.2f ' % inc + \
                  'posang %.2f ' % posang + \
                  'npix %d ' % Npix + \
                  'sizeau %d ' % sizeau + \
                  'loadlambda ' + \
                  'setthreads 6')
        os.chdir(cwd)

        # load the output into a proper cube array
        imagefile = open(self.modelname+'/image.out')
        iformat = imagefile.readline()
        im_nx, im_ny = imagefile.readline().split() #npixels along x and y axes
        im_nx, im_ny = np.int(im_nx), np.int(im_ny)
        nlam = np.int(imagefile.readline())

        pixsize_x, pixsize_y = imagefile.readline().split() #pixel sizes in cm 
        pixsize_x = np.float(pixsize_x)
        pixsize_y = np.float(pixsize_y)

        imvals = ascii.read(self.modelname+'/image.out', format='fast_csv', 
                            guess=False, data_start=4,
                            fast_reader={'use_fast_converter':True})['1']
        lams = imvals[:nlam]

        # erg cm^-2 s^-1 Hz^-1 str^-1 --> Jy / pixel
        data = np.reshape(imvals[nlam:],[nlam, im_ny, im_nx])
        data *= 1e23 * pixsize_x * pixsize_y / (dist * _pc)**2

        return data
