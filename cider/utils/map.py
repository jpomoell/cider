# This file is part of CIDER.
#
# Copyright 2023 CIDER developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""Utilities for working with Sunpy maps
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


def get_lon_lat_coordinates(m, position="centers"):
    """Computes the center coordinates of the pixels of the map.

    Args:
        m (sunpy.Map) : Input map
    Returns:
        tuple containing lon and lat coordinates
    """

    if position == "centers":
        lon, lat = get_center_coordinates(m)
    elif position == "edges":
        lon, lat = get_edge_coordinates(m)
    else:
        raise ValueError("Unrecongized mode")

    if is_Stonyhurst(m):
        lon_wrap = 180.0*units.deg
    elif is_Carrington(m):
        lon_wrap = 0.0*units.deg

    # Make sure longitude is strictly increasing
    wrapped_lon \
        = astropy.coordinates.Longitude(lon,
                                        unit=m.meta['cunit1'],
                                        wrap_angle=lon_wrap)

    if position == "edges":
        output_lon = wrapped_lon[0].deg + np.linspace(0, 360.0, len(wrapped_lon))
    else:
        output_lon = wrapped_lon.deg

    output_lat = astropy.coordinates.Latitude(lat, unit=m.meta['cunit2']).deg

    return output_lon, output_lat


def is_Stonyhurst(m):
    return 'HG' in m.meta['ctype1']


def is_Carrington(m):
    return 'CR' in m.meta['ctype1']


def get_center_coordinates(m):
    """Computes the center coordinates of the pixels of the map.

    Args:
        m (sunpy.Map) : Input map
    Returns:
        tuple containing the x1 and x2 center coordinates
    """

    x = m.wcs.wcs_pix2world(np.arange(int(m.meta["naxis1"])), 0, 0)[0]
    y = m.wcs.wcs_pix2world(0, np.arange(int(m.meta["naxis2"])), 0)[1]

    return x, y


def get_edge_coordinates(m):
    """Computes the edge coordinates of the pixels of the map.

    Args:
        m (sunpy.Map) : Input map
    Returns:
        tuple containing the x1 and x2 edge coordinates
    """
    #xrange = np.arange(int(m.meta["naxis1"]) + 1)

    x = m.wcs.wcs_pix2world(np.arange(int(m.meta["naxis1"]) + 1) - 0.5, 0, 0)[0]

    y = m.wcs.wcs_pix2world(0, np.arange(int(m.meta["naxis2"]) + 1) - 0.5, 0)[1]

    # The end-points of x (lon) often cause problems, replace assuming a uniform grid
    x[0] = 2.0*x[1] - x[2]
    x[-1] = 2.0*x[-2] - x[-3]

    return x, y


def create_full_sun_plate_carree_map(m, deg_per_pixel, frame):
    """Creates a full-Sun map using equirectangular projection
    """

    # Check that the number of pixels is an exact integer
    if not np.isclose(math.remainder(180.0, deg_per_pixel), 0.0):
        raise ValueError("Unsupported fractional degrees per pixel")

    if not np.isclose(math.remainder(360.0, deg_per_pixel), 0.0):
        raise ValueError("Unsupported fractional degrees per pixel")

    # Size of the data array
    shape_out = (int(180.0/deg_per_pixel), int(360.0/deg_per_pixel))

    # Generate the coordinate frame of the output map
    # The center of the two maps should identically coincide
    # The center in lat is 0, while in lon it can vary
    frame_out = astropy.coordinates.SkyCoord(m.center.lon.deg, 0,
                                             unit=units.deg,
                                             frame=frame,
                                             obstime=m.date,
                                             observer="Earth")

    # Create the map header
    header = sunpy.map.make_fitswcs_header(shape_out,
                                           frame_out,
                                           scale=[180.0/shape_out[0],
                                                  360.0/shape_out[1]]*units.deg/units.pix,
                                           projection_code="CAR")

    # Create the map
    pc_map = sunpy.map.Map((np.zeros(shape_out), header))

    # Copy metadata that is not already defined
    for key, value in m.meta.items():
        pc_map.meta.setdefault(key, value)

    return pc_map


def carrdeg(date):

    cr = sunpy.coordinates.sun.carrington_rotation_number(date)

    return 360.0*(int(cr) + 1.0 - cr)


def carrington_map_to_stonyhurst(m, date):

    # Pixel coordinates of source map
    lon, lat = get_center_coordinates(m)

    # Find the pixel that in Carrington longitude is closest
    # to the Carrington longitude of the given date
    idx = np.abs(lon - carrdeg(date)).argmin()

    # The strip of data located at lon[idx] is set as the central meridian
    # in the new Stonyhurst map.
    crnum = sunpy.coordinates.sun.carrington_rotation_number(date)
    new_date = sunpy.coordinates.sun.carrington_rotation_time(int(crnum) + (360.0 - lon[idx])/360.0)

    # The amount of pixels to shift: distance from the centralmost pixel
    cm_pix_idx = int(180.0/m.meta["CDELT1"])-1
    shifted_data = np.roll(m.data, cm_pix_idx-idx, axis=1)

    # Create new map
    meta = copy.deepcopy(m.meta)

    meta['DATE-OBS'] = new_date.strftime("%Y-%m-%dT%H:%M:%S")
    meta['CTYPE1'] = meta['CTYPE1'].replace('CR', 'HG')
    meta['CTYPE2'] = meta['CTYPE2'].replace('CR', 'HG')

    meta['CRVAL1'] = 0.0
    meta['CRPIX1'] = cm_pix_idx + 1

    return sunpy.map.GenericMap(shifted_data, meta)


def unit(m):

    unit = m.unit

    if unit is None:
        unit = units.Unit(m.meta.get('bunit'), format='fits', parse_strict='silent')

    return unit


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
    input_lon, input_lat = get_edge_coordinates(input_map)

    # Edge coordinates of the new map
    output_lon, output_lat = get_edge_coordinates(grid_map)

    # Transform longitude so as to have a monotonically increasing coordinate from 0 to 360
    output_lon = astropy.coordinates.Longitude(output_lon-output_lon[0], unit='deg').value
    input_lon = astropy.coordinates.Longitude(input_lon-input_lon[0], unit='deg').value

    # Make sure lon edge coordinates are not the same
    output_lon[-1] = output_lon[0] + 360.0
    input_lon[-1] = input_lon[0] + 360.0

    # Finally, convert to rad
    for crd in (output_lon, input_lon, output_lat, input_lat):
        crd *= np.pi/180.0

    grid_map_center_coords = get_center_coordinates(grid_map)

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
