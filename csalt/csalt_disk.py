import sys
import numpy as np
from scipy.spatial import Delaunay
import scipy.constants as sc
import warnings


class csalt_disk:
    
    def __init__(self, quiet=True):
        if quiet:
            warnings.filterwarnings("ignore")

  
    # -- Gaussian function
    def gaussian(self, x, x0, dx, A):
        """Simple Gaussian function."""
        return A * np.exp(-0.5*((x - x0) / dx)**2)


    # -- Delaunay triangulation
    @staticmethod
    def interp_weights(xyz, uvw):
        tri = Delaunay(xyz)
        simplex = tri.find_simplex(uvw)
        vertices = np.take(tri.simplices, simplex, axis=0)
        temp = np.take(tri.transform, simplex, axis=0)
        delta = uvw - temp[:,2]
        bary = np.einsum('njk,nk->nj', temp[:,:2,:], delta)
        return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))


    # -- Interpolation on Delaunay vertices
    @staticmethod
    def interpolate_vals(values, vtx, wts, fill_value=np.nan):
        ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
        ret[np.any(wts < 0, axis=1)] = fill_value
        return ret


    # -- Populate the sky frame pixel coordinates. Distances in [arcsec].
    def set_sky_coords(self, FOV, npix):
        self.x_sky = np.linspace(-FOV / 2.0, FOV / 2.0, npix)
        self.x_sky, self.y_sky = np.meshgrid(self.x_sky, self.x_sky)
        self.cell_sky = np.diff(self.x_sky).mean()
        self.npix = npix
        self.FOV = FOV


    # -- Populate the disk frame pixel coordinates.  
    def deproj_coords(self, dx, dy, incl, PA, z_func=None, 
                      side='front', r_span=5., N_span=64):
        # proper geometric orientation
        if side == 'front':
            PAr = np.radians(PA)
            inc = np.radians(incl)
            az_sgn = 1.
        elif side == 'back':
            PAr = np.radians(PA + 180)
            inc = -np.radians(incl)
            az_sgn = -1.

        # a "background" (flat) transformation
        xf = (self.x_sky - dx) * np.sin(PAr) + (self.y_sky - dy) * np.cos(PAr)
        yf = (self.x_sky - dx) * np.cos(PAr) - (self.y_sky - dy) * np.sin(PAr)
        yf /= np.cos(inc)
        rf = np.hypot(xf, yf)
        tf = np.arctan2(yf, az_sgn * xf)
        zf = np.zeros_like(rf)

        # generate a 3D grid in the disk frame 
        x_ = np.linspace(-r_span, r_span, N_span)
        y_ = np.linspace(-r_span, r_span, N_span)
        x_disk, y_disk = np.meshgrid(x_, y_)
        z_disk = z_func(np.hypot(x_disk, y_disk))
        z_disk = np.where(np.isfinite(z_disk), z_disk, 0.0)

        # Incline the disk.
        x_dep = x_disk
        y_dep = y_disk * np.cos(inc) - z_disk * np.sin(inc)

        # Remove shadowed pixels.
        if incl < 0.:
            y_dep = np.minimum.accumulate(y_dep[::-1], axis=0)[::-1]
        else:
            y_dep = np.maximum.accumulate(y_dep, axis=0)

        # interpolation
        points_2D = np.column_stack((x_dep.flatten(), y_dep.flatten()))
        xsys = np.column_stack((xf.flatten(), yf.flatten() * np.cos(inc)))
        vtx, wts = self.interp_weights(points_2D, xsys)
        x_obs = np.reshape(self.interpolate_vals(x_disk.flatten(), vtx, wts),
                           self.x_sky.shape)
        y_obs = np.reshape(self.interpolate_vals(y_disk.flatten(), vtx, wts),
                           self.x_sky.shape)
        r_obs = np.hypot(x_obs, y_obs)
        t_obs = np.arctan2(y_obs, az_sgn * x_obs)
        z_obs = z_func(r_obs)

        # backfill extrapolations to a flat disk
        r = np.where(np.isfinite(r_obs), r_obs, rf)
        t = np.where(np.isfinite(t_obs), t_obs, tf)
        z = np.where(np.isfinite(z_obs), z_obs, zf)

        return r, t, z


    def set_disk_coords(self, dx, dy, incl, PA, dist, z_func=None,
                        side='both', r_span=5., N_span=64):

        if side.lower() == 'both':
            # front surface
            r, t, z = self.deproj_coords(dx, dy, incl, PA, z_func=z_func, 
                                         side='front', r_span=r_span, 
                                         N_span=N_span)
            self.r_disk_f, self.t_disk_f, self.z_disk_f = r * dist, t, z * dist
            self.dx_f, self.dy_f = dx, dy
            self.incl_f, self.PA_f = incl, PA
            self.dist_f = dist
            self.x_disk_f = self.r_disk_f * np.cos(self.t_disk_f)
            self.y_disk_f = self.r_disk_f * np.sin(self.t_disk_f)
            # back surface
            r, t, z = self.deproj_coords(dx, dy, incl, PA, z_func=z_func,
                                         side='back', r_span=r_span, 
                                         N_span=N_span)
            self.r_disk_b, self.t_disk_b, self.z_disk_b = r * dist, t, z * dist
            self.dx_b, self.dy_b = dx, dy
            self.incl_b, self.PA_b = incl, PA
            self.dist_b = dist
            self.x_disk_b = self.r_disk_b * np.cos(self.t_disk_b)
            self.y_disk_b = self.r_disk_b * np.sin(self.t_disk_b)
        elif side.lower() == 'front':
            r, t, z = self.deproj_coords(dx, dy, incl, PA, z_func=z_func,
                                         side='front', r_span=r_span, 
                                         N_span=N_span)
            self.r_disk_f, self.t_disk_f, self.z_disk_f = r * dist, t, z * dist
            self.dx_f, self.dy_f = dx, dy
            self.incl_f, self.PA_f = incl, PA
            self.dist_f = dist
            self.x_disk_f = self.r_disk_f * np.cos(self.t_disk_f)
            self.y_disk_f = self.r_disk_f * np.sin(self.t_disk_f)
        elif side.lower() == 'back':
            r, t, z = self.deproj_coords(dx, dy, incl, PA, z_func=z_func,
                                         side='back', r_span=r_span, 
                                         N_span=N_span)
            self.r_disk_b, self.t_disk_b, self.z_disk_b = r * dist, t, z * dist
            self.dx_b, self.dy_b = dx, dy
            self.incl_b, self.PA_b = incl, PA
            self.dist_b = dist
            self.x_disk_b = self.r_disk_b * np.cos(self.t_disk_b)
            self.y_disk_b = self.r_disk_b * np.sin(self.t_disk_b)
        else:
            raise ValueError("`side` must be 'front', 'back' or None.")


    # -- The gas temperature distribution along the emitting surfaces
    def set_Tgas_profile(self, function, side='both', vmin=0.0, vmax=None):
        if side.lower() == 'front':
            self.Tgas_f = np.clip(function(self.r_disk_f), 
                                  a_min=vmin, a_max=vmax)
        elif side.lower() == 'back':
            self.Tgas_b = np.clip(function(self.r_disk_b), 
                                  a_min=vmin, a_max=vmax)
        else:
            raise ValueError("`side` must be 'front' or 'back'.")


    def Tgas_function(self, function, radius, vmin=0, vmax=None):
        return np.clip(function(radius), a_min=vmin, a_max=vmax)


    # -- The line profile width distribution along the emitting surfaces
    def set_dV_profile(self, function, side='both', vmin=0.0, vmax=None):
        if side.lower() == 'both':
            self.dV_f = np.clip(function(self.r_disk_f), 
                                a_min=vmin, a_max=vmax)
            self.dV_b = np.clip(function(self.r_disk_b), 
                                a_min=vmin, a_max=vmax)
        elif side.lower() == 'front':
            self.dV_f = np.clip(function(self.r_disk_f), 
                                a_min=vmin, a_max=vmax)
        elif side.lower() == 'back':
            self.dV_b = np.clip(function(self.r_disk_b), 
                                a_min=vmin, a_max=vmax)
        else:
            raise ValueError("`side` must be 'front', 'back' or 'both'.")


    # -- The optical depth distribution along the emitting surfaces
    def set_tau_profile(self, function, side='both', vmin=0.0, vmax=None):
        if side.lower() == 'both':
            self.tau_f = np.clip(function(self.r_disk_f), 
                                 a_min=vmin, a_max=vmax)
            self.tau_b = np.clip(function(self.r_disk_b), 
                                 a_min=vmin, a_max=vmax)
        elif side.lower() == 'front':
            self.tau_f = np.clip(function(self.r_disk_f), 
                                 a_min=vmin, a_max=vmax)
        elif side.lower() == 'back':
            self.tau_b = np.clip(function(self.r_disk_b), 
                                 a_min=vmin, a_max=vmax)
        else:
            raise ValueError("`side` must be 'front', 'back' or 'both'.")


    # -- The azimuthal velocity distribution along the emitting surfaces
    def set_vtheta_profile(self, function, side='both', vmin=0.0, vmax=None):
        if side.lower() == 'both':
            self.vtheta_f = np.clip(function(self.r_disk_f),
                                    a_min=vmin, a_max=vmax)
            self.vtheta_b = np.clip(function(self.r_disk_b),
                                    a_min=vmin, a_max=vmax)
        elif side.lower() == 'front':
            self.vtheta_f = np.clip(function(self.r_disk_f),
                                    a_min=vmin, a_max=vmax)
        elif side.lower() == 'back':
            self.vtheta_b = np.clip(function(self.r_disk_b),
                                    a_min=vmin, a_max=vmax)
        else:
            raise ValueError("`side` must be 'front', 'back' or 'both'.")


    # -- Projected velocity profiles
    @property
    def vtheta_f_proj(self):
        vt = self.vtheta_f * np.cos(self.t_disk_f)
        return vt * np.sin(abs(np.radians(self.incl_f)))

    @property
    def vtheta_b_proj(self):
        vt = self.vtheta_b * np.cos(self.t_disk_b)
        return vt * np.sin(abs(np.radians(self.incl_b)))


    # -- The emission cube: returns in [Jy/pixel]; velax and vlsr in [m/s]
    def get_cube(self, velax, restfreq=230.538e9, vlsr=0.0):
        # frequencies
        nuax = restfreq * (1 - velax / sc.c)

        # spectrally-dependent optical depths (front surface)
        tau_nuf = self.gaussian(velax[:, None, None],
                                self.vtheta_f_proj[None, :, :] + vlsr,
                                self.dV_f[None, :, :] / np.sqrt(2.0),
                                self.tau_f[None, :, :])

        # Planck function (front surface)
        Bnu_f = (2 * sc.h * nuax[:, None, None]**3 / sc.c**2) / \
                (np.exp(sc.h * nuax[:, None, None] / \
                        (sc.k * self.Tgas_f[None, :, :])) - 1.0)

        # emission distribution (front surface)
        Inu_f = Bnu_f * (1.0 - np.exp(-tau_nuf)) 
        Inu_f = np.where(np.isfinite(Inu_f), Inu_f, 0.0)

        # spectrally-dependent optical depths (back surface)
        tau_nub = self.gaussian(velax[:, None, None],
                                self.vtheta_b_proj[None, :, :] + vlsr,
                                self.dV_b[None, :, :] / np.sqrt(2.0),
                                self.tau_b[None, :, :])

        # Planck function (back surface)
        Bnu_b = (2 * sc.h * nuax[:, None, None]**3 / sc.c**2) / \
                (np.exp(sc.h * nuax[:, None, None] / \
                        (sc.k * self.Tgas_b[None, :, :])) - 1.0)

        # emission distribution (back surface)
        Inu_b = Bnu_b * (1.0 - np.exp(-tau_nub)) * np.exp(-tau_nuf)
        Inu_b = np.where(np.isfinite(Inu_b), Inu_b, 0.0)

        # combined emission distribution, to proper Jy/pixel units
        Inu = Inu_f + Inu_b
        pix_area = (self.cell_sky * np.pi / 180. / 3600.)**2
        Inu *= 1e26 * pix_area

        return Inu
