# This file is part of CIDER.
#
# Copyright 2024 CIDER developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""Computing open and closed regions
"""

import numpy as np
import numba
import sunpy.map

import cider.utils.map


def compute_open_closed_map(magnetogram, deg_per_pixel, start_radius, open_radius, tracer):

    open_closed_map \
        = cider.utils.map.create_full_sun_plate_carree_map(magnetogram,
                                                           deg_per_pixel=deg_per_pixel,
                                                           frame=magnetogram.coordinate_frame.name)

    crds = sunpy.map.all_coordinates_from_map(open_closed_map)

    data = _open_closed_kernel(start_radius.si.value,
                               crds.lon[0, :].rad,
                               crds.lat[:, 0].rad,
                               open_radius.si.value,
                               tracer)

    open_closed_map.data[:, :] = data[:, :]

    return open_closed_map


@numba.njit()
def _open_closed_kernel(start_radius, lons, lats, open_radius, tracer):

    clts = 0.5*np.pi - lats
    
    data = np.zeros((len(lats), len(lons)))

    for j, clt in enumerate(clts):
        for k, lon in enumerate(lons):
        
            # Start point of tracing
            beg_pt = (start_radius, clt, lon)

            # Trace along field dir
            end_pt_par = tracer.compute(beg_pt, False)

            # Trace in opposite dir
            end_pt_apar = tracer.compute(beg_pt, True)

            # Field at the start point
            B = tracer.interpolator.value(beg_pt)

            # Max radius of field line end points
            r_end = max(end_pt_par[0], end_pt_apar[0])
            
            if r_end >= open_radius:
                data[j, k] = 2.0*np.sign(B[0])
            else:
                data[j, k] = 1.0*np.sign(B[0])
                
    return data