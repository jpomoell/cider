# This file is part of CIDER.
#
# Copyright 2026 CIDER developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""Outflow models
"""

import numpy as np
import numba

@numba.njit()
def _set_values(r, vr, nu0):

    # Compute log(f) based on given vr
    log_f = np.zeros(len(r))

    # Set value at start. Since f is determined only up to 
    # a constant, this value is arbitrary
    log_f[0] = 0.0

    # Compute log(f) values
    for i in range(len(r)-1):

        dr = r[i+1] - r[i]
        rc = r[i] + 0.5*dr
            
        # d/dr ln f(r) = nu0 v_r(r)
        # (lnf_{i+1/2} - lnf_{i-1/2})/dr = nu0 vr_{i}
        log_f[i+1] = log_f[i] + dr*nu0*vr(rc)
    
    return log_f


class DiscreteOutflowModel:
    
    def __init__(self, r_edges, nu0):
        
        self.r_edges = np.copy(r_edges)
        self.r_centers = 0.5*(r_edges[1::] + r_edges[0:-1])

        self.nu0 = nu0
    
    def set_values(self, vr):

        # Compute log(f) based on given vr
        self.log_f = np.zeros(len(self.r_edges))

        # Store also the radial speed profile for reference
        self.vr = np.zeros(len(self.r_centers))

        # Set value at start. Since f is determined only up to 
        # a constant, this value is arbitrary
        self.log_f[0] = 0.0

        # Compute log(f) values
        for i in range(len(self.r_edges)-1):
        
            dr = self.r_edges[i+1] - self.r_edges[i]
            rc = self.r_edges[i] + 0.5*dr
            
            self.vr[i] = vr(rc)

            # d/dr ln f(r) = nu0 v_r(r)
            # (lnf_{i+1/2} - lnf_{i-1/2})/dr = nu0 vr_{i}
            self.log_f[i+1] = self.log_f[i] + dr*self.nu0*self.vr[i]
        
        #self.log_f = _set_values(self.r_edges, vr, self.nu0)

    def _f(self, r):

        i = np.max(np.searchsorted(self.r_edges, r, "right") - 1, 0)        

        dr = self.r_edges[i+1] - self.r_edges[i]
        ri = self.r_edges[i]
        
        # Value of log(f) computed via interpolation
        log_f = (self.log_f[i+1]*(r - ri) - self.log_f[i]*(r - ri - dr))/dr

        return np.exp(log_f)

    def f(self, r):

        if isinstance(r, np.ndarray):
            return np.asarray([self._f(_r) for _r in r])
        else:
            return self._f(r)