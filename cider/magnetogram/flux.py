# This file is part of CIDER.
#
# Copyright 2023 CIDER developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""Magnetogram flux computation
"""

import dataclasses
import copy
import numpy as np
import numba

import astropy.units as u
import astropy.units.quantity

import cider.utils.map


@dataclasses.dataclass
class FluxContent:
    
    signed : astropy.units.quantity.Quantity
    unsigned : astropy.units.quantity.Quantity
    area : astropy.units.quantity.Quantity


class Flux:
    """Computes the signed and unsigned fluxes of a magnetogram
    """
        
    @staticmethod
    def compute(magnetogram):
        
        lon, lat = cider.utils.map.get_lon_lat_coordinates(magnetogram, position="edges")
        
        clt = 90.0-lat
        
        # 
        radius = magnetogram.rsun_meters.value
        
        signed_flux, unsigned_flux, area \
            = cider.magnetogram.flux.signed_unsigned_flux_kernel(magnetogram.data,
                                                                 clt[::-1]*np.pi/180.0,
                                                                 lon*np.pi/180.0,
                                                                 radius)

        signed_flux *= magnetogram.unit*u.m**2
        unsigned_flux *= magnetogram.unit*u.m**2
        area *= u.m*u.m

        return FluxContent(signed_flux, unsigned_flux, area)


class Balance:
    """Balances the signed flux of a magnetogram using a multiplicative method.
    """

    @staticmethod
    def multiplicative(magnetogram):

        balanced_magnetogram = copy.deepcopy(magnetogram)
        
        lon, lat = cider.utils.map.get_lon_lat_coordinates(magnetogram, position="edges")
        clt = 90.0-lat

        flux_kernel = lambda m : cider.magnetogram.flux.signed_unsigned_flux_kernel(m,
                                                                                  clt[::-1]*np.pi/180.0,
                                                                                  lon*np.pi/180.0,
                                                                                  1.0)[0]
      
        balanced_magnetogram.data[:, :] \
            = Balance._multiplicative_kernel(magnetogram.data, flux_kernel)[:, :]

        return balanced_magnetogram

    @staticmethod
    def _multiplicative_kernel(Br, flux_kernel):

        # Pixels with pos and neg data
        neg_pixels = np.where(Br <= 0.0)
        pos_pixels = np.where(Br >= 0.0)

        # Construct pos and neg Br arrays
        BrP = np.copy(Br)
        BrP[neg_pixels] = 0.0

        BrM = np.copy(Br)
        BrM[pos_pixels] = 0.0

        # Fluxes of negative and positive polarities
        pos_flux = flux_kernel(BrP)
        neg_flux = flux_kernel(BrM)

        # Constant c chosen to retain the unsigned flux of the origin map
        c = 0.5*(pos_flux - neg_flux)

        # Construct balanced map
        balanced_Br = np.copy(Br)
        balanced_Br[pos_pixels] *= c/pos_flux
        balanced_Br[neg_pixels] *= -c/neg_flux

        return balanced_Br


@numba.njit()
def signed_unsigned_flux_kernel(Bn, clt_edges, lon_edges, R):

    # Pixel center coordinates  
    clt_centers = 0.5*(clt_edges[1::] + clt_edges[0:-1])
    lon_centers = 0.5*(lon_edges[1::] + lon_edges[0:-1])

    signed_flux = 0.0
    unsigned_flux = 0.0
    area = 0.0
    for j, clt in enumerate(clt_centers):
        for k, lon in enumerate(lon_centers):
            
            # Pixel size
            dclt = clt_edges[j+1] - clt_edges[j]
            dlon = lon_edges[k+1] - lon_edges[k]

            # \int B dot dA = \int Br R^2 sin (th) dth dph
            # Assuming B_r = const => Br R^2 \Delta ph [ cos(th - dth/2) - cos(th + dth/2) ]
            dA = R*R*dlon*np.sin(clt)*2.0*np.sin(0.5*dclt)
                       
            signed_flux += Bn[j, k]*dA
            unsigned_flux += np.abs(Bn[j, k]*dA)
            area += dA
            
    return signed_flux, unsigned_flux, area
