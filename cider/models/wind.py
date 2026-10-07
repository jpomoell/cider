# This file is part of CIDER.
#
# Copyright 2026 CIDER developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""Solar wind models
"""

import dataclasses
import numpy as np
import numba

import astropy
import astropy.constants
import astropy.units as units


@dataclasses.dataclass
class ParkerIsothermalModelParameters:
    
    # Temperature at the reference radius
    T0 : astropy.units.quantity.Quantity

    # Reference radius
    r0 : units.Quantity = dataclasses.field(default_factory=lambda: astropy.constants.R_sun)

    # Mass density at the reference radius
    rho0 : units.Quantity = dataclasses.field(default_factory=lambda: 1e-12*units.kg*units.m**(-3))


class DefaultEquationOfState:
    
    @staticmethod
    def pressure(rho, T):
        return 2.0*rho*astropy.constants.k_B*T/astropy.constants.m_p

    @staticmethod
    def number_density(P, T):
        return P/(2.0*astropy.constants.k_B*T)


@numba.njit()
def _parker_isothermal_vr(x):

    # Dimensionless radial coordinate
    # x = r/critical_radius

    rhs = 4.0*(np.log(x)+1.0/x)-3.0

    # Starting value for Newton-Raphson iteration chosen
    # differently depending on if we are beyond the critical
    # point or not. This ensures that we get the supersonic
    # solution branch.
    # NOTE! The iterations seems to fail at least in the following
    # cases:
    #    * When T0 is low
    #    * for r < r0

    u = 1e-4
    if x > 1.0:
        u = 1.1

    for _ in range(50):
        unew = u - (np.log(u*u)-u*u+rhs)/((2.0/u)-2.0*u)
        u = unew

    return u


@numba.njit()
def _parker_isothermal_vr_array(values):

    u = np.empty_like(len(values))

    for i in range(len(values)):
        u[i] = _parker_isothermal_vr(values[i])

    return u

class ParkerIsothermalSolarWind:

    def __init__(self, params, eos = DefaultEquationOfState()):
        self.params = params
        self.eos = eos

    @property
    def isothermal_sound_speed(self):
        
        # cS^2 = gamma P/rho
        P0 = self.eos.pressure(self.params.rho0, self.params.T0)
        
        return np.sqrt(P0/self.params.rho0)

    @property
    def critical_radius(self):
        return 0.5*astropy.constants.GM_sun/(self.isothermal_sound_speed**2)
        
    def vr(self, r):
        """Computes the radial speed at the heliocentric radius r
        """

        x = (r/self.critical_radius).si.value

        if isinstance(x, np.ndarray):
            return _parker_isothermal_vr_array(x)*self.isothermal_sound_speed
        else:
            return _parker_isothermal_vr(x)*self.isothermal_sound_speed
        
    @property
    def T(self):
        return self.params.T0

    @property
    def vr0(self):
        return self.vr(self.params.r0)
    
    def mass_density(self, r):
        return self.params.rho0*(self.vr0/self.vr(r))*(self.params.r0/r)**2
