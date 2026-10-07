# This file is part of CIDER.
#
# Copyright 2023 CIDER developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""Utilities for regridding magnetograms
"""

import math
import copy
import numpy as np

import numba

import sunpy
import sunpy.sun
import sunpy.map

import astropy
import astropy.units as units
import astropy.coordinates


def regrid_to_grid_of_map(input_map, grid_map):
    """Regrids a map.

    Rebins a map to the grid of a given new map. The regridding is done by
    summing the flux contributions from the input map to the pixels of the new
    map. As a result, the flux is conserved in the regridding. In this procedure,
    the physical quantity is assumed to be constant within each pixel. The exact
    integrated spherical area element is used to compute the area of each pixel.

    Args:
        input_map (sunpy.Map) : Input map to regrid
        grid_map (sunpy.Map)  : Map defining the grid to which to regrid

    Returns:
        sunpy.Map containing the rebinned map. The metadata is copied from grid_map.
    """

    # Edge coordinates of pixels of the input map
    input_map_coordinates = sunpy.map.all_corner_coords_from_map(input_map)

    input_lon = input_map_coordinates.lon[0, :].rad
    input_lat = input_map_coordinates.lat[:, 0].rad

    # Edge coordinates of the new map
    output_map_coordinates = sunpy.map.all_corner_coords_from_map(grid_map)

    output_lon = output_map_coordinates.lon[0, :].rad
    output_lat = output_map_coordinates.lat[:, 0].rad

    # Transform longitude so as to have a monotonically increasing coordinate from 0 to 360
    output_lon = astropy.coordinates.Longitude(output_lon-output_lon[0], unit='deg').value
    input_lon = astropy.coordinates.Longitude(input_lon-input_lon[0], unit='deg').value

    # Make sure lon edge coordinates are not the same
    output_lon[-1] = output_lon[0] + 360.0
    input_lon[-1] = input_lon[0] + 360.0

    # Finally, convert to rad
    for crd in (output_lon, input_lon, output_lat, input_lat):
        crd *= np.pi/180.0

    grid_map_centers = sunpy.map.all_coordinates_from_map(grid_map)
    grid_map_center_coords = (grid_map_centers.lon[0, :].rad, grid_map_centers.lat[:, 0].rad)
    
    input_map_data = np.asarray(input_map.data, dtype=np.float64)

    data = _regrid_kernel(input_map_data,
                          (input_lon, input_lat),
                          (output_lon, output_lat),
                          grid_map_center_coords)

    return sunpy.map.GenericMap(data, copy.copy(grid_map.meta))


@numba.njit
def _regrid_kernel(input_map_data, input_coords, output_coords, grid_map_center_coords):
    """Compute kernel for remapping.
    """

    input_lon, input_lat = input_coords
    output_lon, output_lat = output_coords

    #
    # Find pixels in lat that overlap, and store the indices to their edges
    #
    j_indices = []
    for j, lat in enumerate(grid_map_center_coords[1]):

        jlower = np.where(output_lat[j] >= input_lat)[0][-1]
        jupper = np.where(output_lat[j+1] <= input_lat)[0][0]

        j_indices.append(range(jlower, jupper))

    #
    # Same in lon
    #
    k_indices = []
    for k, lon in enumerate(grid_map_center_coords[0]):

        klower = np.where(output_lon[k] >= input_lon)[0][-1]
        kupper = np.where(output_lon[k+1] <= input_lon)[0][0]

        k_indices.append(range(klower, kupper))

    #
    # Perform regridding
    #
    data = np.zeros((len(output_lat)-1, len(output_lon)-1))
    for j, lat in enumerate(grid_map_center_coords[1]):
        for k, lon in enumerate(grid_map_center_coords[0]):

            # Area of the output pixel (without R^2 as it will cancel out)
            output_area = (output_lon[k+1]-output_lon[k])*(np.sin(output_lat[j+1])-np.sin(output_lat[j]))

            # Compute flux
            flux = 0.0
            for ji in j_indices[j]:
                for ki in k_indices[k]:

                    #
                    # Compute the area of the region where the two pixels overlap
                    #
                    lon_lower = max(input_lon[ki], output_lon[k])
                    lon_upper = min(input_lon[ki+1], output_lon[k+1])
                    lat_lower = max(input_lat[ji], output_lat[j])
                    lat_upper = min(input_lat[ji+1], output_lat[j+1])

                    overlap_area = (lon_upper-lon_lower)*(np.sin(lat_upper)-np.sin(lat_lower))

                    # Flux from this pixel area
                    flux += input_map_data[ji, ki]*overlap_area

            data[j, k] = flux/output_area

    return data
