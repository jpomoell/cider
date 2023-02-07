# This file is part of CIDER.
#
# Copyright 2023 CIDER developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""HMI magnetogram readers
"""

import datetime
import numpy as np
import re

import sunpy
import sunpy.sun
import sunpy.map

import astropy.units as units



def read_hmi_synoptic(file_name):
    """Read synoptic HMI magnetogram in FITS format.

    Args:
        file_name : name of the FITS file.

    Returns:
        sunpy.map.Map with corrected/modified metadata
    """

    # Read the file
    hmi_fits = sunpy.io.read_file(file_name)

    # For these datasets, the actual data is in the second segment
    data = np.asarray(hmi_fits[1][0], dtype=np.float64)
    meta = dict(hmi_fits[1][1])

    #
    # Correct/add metadata
    #
    # Change unit from Mx/cm^2 to G so that Astropy recognizes it
    meta['bunit'] = 'G'

    # A CR map does not have a single unique time.
    # For these maps, the T_OBS has been set to be equal to the time of
    # the midpoint of the CR rotation which is used as the reference time

    date = utils.parse_date_string(meta['T_OBS'])

    #print(meta['T_OBS'], date.isoformat())

    # Timestamp includes milliseconds
    #match = re.match(r"\d{4}.\d{2}.\d{2}_\d{2}:\d{2}:\d{2}\.\d", date_str)
    #if match:
    #    date = datetime.datetime.strptime(date_str, "%Y.%m.%d_%H:%M:%S.%f_TAI")

    # Timestamp does not have milliseconds
    #match = re.match(r"\d{4}.\d{2}.\d{2}_\d{2}:\d{2}:\d{2}_TAI", date_str)
    #if match:
    #    print(match)
    #    date = datetime.datetime.strptime(date_str, "%Y.%m.%d_%H:%M:%S_TAI")

    #date = datetime.datetime.strptime(meta['T_OBS'], '%Y.%m.%d_%H:%M:%S_TAI')

    meta['DATE-OBS'] = date.strftime("%Y-%m-%dT%H:%M:%S")

    # Add location info
    observer = sunpy.coordinates.get_earth(date)

    meta['HGLT_OBS'] = observer.lat.to_value(units.deg)
    meta['HGLN_OBS'] = observer.lon.to_value(units.deg)
    meta['DSUN_OBS'] = observer.radius.to_value(units.m)

    meta['CUNIT1'] = 'deg'
    meta['CUNIT2'] = 'deg'

    meta['CDELT1'] *= -1.0
    meta['CDELT2'] = 2.0/(np.pi*meta['NAXIS2']/180.0)

    meta['CRDER1'] = 0.0
    meta['CRDER2'] = 0.0

    return sunpy.map.Map((data, meta))


def read_hmi_daily_synframe(file_name):

    # Read the file
    hmi_fits = sunpy.io.read_file(file_name)

    # For these datasets, the actual data is in the second segment
    data = np.asarray(hmi_fits[1][0], dtype=np.float64)
    meta = dict(hmi_fits[1][1])

    #
    # Correct/add metadata
    #

    # Change unit from Mx/cm^2 to G so that Astropy recognizes it
    meta['bunit'] = 'G'

    # Fill in missing date
    date_str = meta['T_REC']

    # Timestamp includes milliseconds
    match = re.match(r"\d{4}.\d{2}.\d{2}_\d{2}:\d{2}:\d{2}\.\d", date_str)
    if match:
        date = datetime.datetime.strptime(date_str, "%Y.%m.%d_%H:%M:%S.%f_TAI")

    # Timestamp does not have milliseconds
    match = re.match(r"\d{4}.\d{2}.\d{2}_\d{2}:\d{2}:\d{2}_TAI", date_str)
    if match:
        date = datetime.datetime.strptime(date_str, "%Y.%m.%d_%H:%M:%S_TAI")

    meta['DATE-OBS'] = date.strftime("%Y-%m-%dT%H:%M:%S")

    # Add location info
    observer = sunpy.coordinates.get_earth(date)

    meta['HGLT_OBS'] = observer.lat.to_value(units.deg)
    meta['HGLN_OBS'] = observer.lon.to_value(units.deg)
    meta['DSUN_OBS'] = observer.radius.to_value(units.m)

    #self.meta['cdelt2'] = 180 / np.pi * self.meta['cdelt2']
    #self.meta['cdelt1'] = np.abs(self.meta['cdelt1'])

    meta['CUNIT1'] = 'deg'
    meta['CUNIT2'] = 'deg'

    meta["CRVAL1"] = meta["CRLN_OBS"] + 120.0  # Might need modulus?

    meta['CDELT1'] *= -1.0
    meta['CDELT2'] = 2.0/(np.pi*meta['NAXIS2']/180.0)

    meta['CRDER1'] = 0.0
    meta['CRDER2'] = 0.0

    return sunpy.map.Map((data, meta))
