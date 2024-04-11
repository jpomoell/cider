# This file is part of CIDER.
#
# Copyright 2024 CIDER developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""Streamline tracer
"""

import numpy as np

import numba
import numba.experimental.jitclass as jitclass

import cider.utils.interpolation


@jitclass()
class SphericalShellMidPointStreamLineTracer:
    """Streamline tracer in a spherical shell domain.
    """

    # Maximum allowed length of stream line
    max_path_length : numba.float64

    # Minimum allowed step size
    min_step_size : numba.float64

    # Step size relative to the local grid spacing
    relative_step_size : numba.float64
    
    # Record field line coordinates?
    record_points : numba.boolean #= False

    # Field line coordinates
    points : numba.types.List(numba.types.Array(numba.types.float64, 1, "C")) 

    # Interpolation method
    interpolator : cider.utils.interpolation.NodalSphericalShellVectorFieldInterpolator.class_type.instance_type

    # Radial domain size
    Rmin : numba.float64
    Rmax : numba.float64
    
    def __init__(self, interpolator):

        # By default, do not limit the minimum step size
        self.min_step_size = 0.0

        # By default, set relative step size ot 1/4 of the local grid size
        self.relative_step_size = 0.25
        
        # By default, point recording is off
        self.record_points = False
        
        # Initialize point structure
        self.points = [np.zeros(3)]
        self.points.clear()

        # Set the interpolator
        self.interpolator = interpolator

        # Radial domain extent        
        self.Rmin = interpolator.components[0].coordinates[0][ 0]
        self.Rmax = interpolator.components[0].coordinates[0][-1]

        # Set default max path length
        self.max_path_length = 2.0*np.pi*self.Rmax
    
    def is_inside_domain(self, pt):

        r = np.sqrt(np.dot(pt, pt))
        
        return (r > self.Rmin) and (r < self.Rmax)

    def sphere_ray_intersection(self, x0, x1, R):
        
        # Straight ray from xin to xout
        # x(t) = x0 + t*dx, t in [0, 1] and where dx = x1 - x0
        dx = x1 - x0
        
        # Find t where the intersection with the sphere takes place,
        # dot(x(t), x(t)) = R^2

        # Coefficients of the resulting second degree polynomial
        a = np.dot(dx, dx)
        b = 2.0*np.dot(x0, dx)
        c = np.dot(x0, x0) - R*R

        # Discriminant
        D = b*b - 4.0*a*c
        if D < 0.0:
            print(x0, x1, R)
            raise ValueError("Unexpected negative discriminant")
            
        sqrt_D = np.sqrt(D)
        
        # The smallest t value should correspond to the the sought t
        t1 = (-b - sqrt_D)/(2.0*a)
        t2 = (-b + sqrt_D)/(2.0*a)

        t = min(abs(t1), abs(t2))

        # Return point of intersection        
        return x0 + t*dx
            
    def get_boundary_intersection(self, xin, xout):

        r = np.sqrt(np.dot(xin, xin))

        R = self.Rmin
        if np.abs(r - self.Rmin) > np.abs(r - self.Rmax):
            R = self.Rmax
    
        return self.sphere_ray_intersection(xin, xout, R)
        
    def compute(self, start_pt, reverse=False):

        # Clear possible previous points
        if self.record_points:
            self.points.clear()

        # Starting point
        x = self.interpolator.to_cartesian(start_pt)

        # Break if point outside the domain
        if not self.is_inside_domain(x):
            return self.interpolator.to_spherical(x)

        # Record starting point (now known to be inside the domain)
        if self.record_points:
            self.points.append(self.interpolator.to_spherical(x))

        # Field direction        
        direction = 1.0
        if reverse:
            direction = -1.0
        
        
        path_length = 0.0
        while path_length < self.max_path_length:

            # Current point in spherical coordinates
            xs = self.interpolator.to_spherical(x)
            
            # Get cell size at current position
            cell_size = self.interpolator.cell_size(xs)
            
            # Determine step size
            ds = direction*max(np.min(cell_size)*self.relative_step_size, self.min_step_size)
        
            # Move half step
            xhalf = x + 0.5*ds*self.interpolator.cartesian_unit_vector(x)

            # Halfway point outside the domain?
            if not self.is_inside_domain(xhalf):                
                # Get point of intersection with boundary
                x = self.get_boundary_intersection(x, xhalf)
                break

            # Move full step using the field at the half step
            xnext = x + ds*self.interpolator.cartesian_unit_vector(xhalf)

            # Next point outside the domain?
            if not self.is_inside_domain(xnext):
                # Get point of intersection with boundary
                x = self.get_boundary_intersection(xhalf, xnext)
                break

            # Add segment to path length
            path_length += np.abs(ds)            

            # Update point
            x = xnext

            # Record
            if self.record_points:
                self.points.append(np.copy(xs))

        # Finished, record last point and also return it
        if self.record_points:
            self.points.append(self.interpolator.to_spherical(x))
        
        return self.interpolator.to_spherical(x)