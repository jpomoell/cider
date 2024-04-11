# This file is part of CIDER.
#
# Copyright 2024 CIDER developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""Interpolation utilities
"""

import numpy as np

import numba
import numba.experimental.jitclass as jitclass

import pysmsh
import pysmsh.interpolate.trilinear


def average_face_staggered_to_nodal(v, spherical_shell=True):
    """Average the given face-staggered vector field to mesh nodes
    """

    if not v.coloc.typeid == pysmsh.colocation.ColocationId.face_staggered:
        raise ValueError("Input field should be a face-staggered vector field")

    nodal_field = pysmsh.Field.Vector(v.mesh, "nodal")

    #
    # Average in-domain data to nodes
    #
    nodal_field.data[0][:, 1:-1, 1:-1] \
        = 0.25*(  v.data[0][:, 0:-1, 0:-1] 
                + v.data[0][:, 1:: , 0:-1]
                + v.data[0][:, 0:-1, 1:: ] 
                + v.data[0][:, 1:: , 1:: ])

    nodal_field.data[1][1:-1, :, 1:-1] \
        = 0.25*(  v.data[1][0:-1, :, 0:-1] 
                + v.data[1][1:: , :, 0:-1] 
                + v.data[1][0:-1, :, 1:: ] 
                + v.data[1][1:: , :, 1:: ])

    nodal_field.data[2][1:-1, 1:-1, :] \
        = 0.25*(  v.data[2][0:-1, 0:-1, :] 
                + v.data[2][1:: , 0:-1, :] 
                + v.data[2][0:-1, 1:: , :] 
                + v.data[2][1:: , 1:: , :])

    if spherical_shell:

        # Periodic lon data at edges
        nodal_field.data[0][:, 1:-1, 0] \
            = 0.25*(  v.data[0][:, 0:-1, -1] 
                    + v.data[0][:, 1:: , -1]
                    + v.data[0][:, 0:-1,  0] 
                    + v.data[0][:, 1:: ,  0])

        nodal_field.data[0][:, 1:-1, -1] = nodal_field.data[0][:, 1:-1, 0]

        nodal_field.data[1][1:-1, :, 0] \
            = 0.25*(  v.data[1][0:-1, :, -1] 
                    + v.data[1][1:: , :, -1] 
                    + v.data[1][0:-1, :,  0] 
                    + v.data[1][1:: , :,  0])

        nodal_field.data[1][1:-1, :, -1] = nodal_field.data[1][1:-1, :, 0]

        # Extrapolate vt and vp linearly down to the lower radial boundary
        nodal_field.data[1][0, :, :] \
            = 2.0*nodal_field.data[1][1, :, :] - nodal_field.data[1][2, :, :]

        nodal_field.data[2][0, :, :] \
            = 2.0*nodal_field.data[2][1, :, :] - nodal_field.data[2][2, :, :]

        # Set polar vr
        for i in range(len(v.mesh.edges.r)):
            nodal_field.data[0][i,  0, :] = np.average(nodal_field.data[0][i,  1, :])
            nodal_field.data[0][i, -1, :] = np.average(nodal_field.data[0][i, -2, :])

            nodal_field.data[1][i,  0, :] = 2.0*nodal_field.data[1][i,  1, :] - nodal_field.data[1][i,  2, :]
            nodal_field.data[1][i, -1, :] = 2.0*nodal_field.data[1][i, -2, :] - nodal_field.data[1][i, -3, :]

            nodal_field.data[2][i,  0, :] = 2.0*nodal_field.data[2][i,  1, :] - nodal_field.data[2][i,  2, :]
            nodal_field.data[2][i, -1, :] = 2.0*nodal_field.data[2][i, -2, :] - nodal_field.data[2][i, -3, :]

    return nodal_field


@jitclass()
class NodalSphericalShellVectorFieldInterpolator:
    """Trilinear interpolation of a node-defined vector field in a spherical shell
    """

    # Component data
    components : numba.types.UniTuple(pysmsh.interpolate.trilinear.TrilinearInterpolation.class_type.instance_type, 3)
    
    def __init__(self, field):

        if not field.is_vector_field:
            raise ValueError("Input field should be a vector field")

        if not field.coloc.typeid == pysmsh.colocation.ColocationId.node_centered:
            raise ValueError("Input field should be a nodal vector field")

        v1_interp = pysmsh.interpolate.trilinear.TrilinearInterpolation(field.data[0], field.mesh.edges, True)
        v2_interp = pysmsh.interpolate.trilinear.TrilinearInterpolation(field.data[1], field.mesh.edges, True)
        v3_interp = pysmsh.interpolate.trilinear.TrilinearInterpolation(field.data[2], field.mesh.edges, True)
                
        self.components = (v1_interp, v2_interp, v3_interp)

    def wrap_longitude(self, p):
        
        # Longitudes (phi-coordinates)
        p_coordinates = self.components[0].coordinates[2]

        return p_coordinates[0] + np.remainder(p - p_coordinates[0], 2.0*np.pi)               
    
    def wrap(self, coordinate):

        r, t, p = coordinate
        
        # Longitudes (phi-coordinates)
        p_coordinates = self.components[0].coordinates[2]

        # Wrap co-latitude
        t_wrapped = t
        p_wrapped = p
    
        if t < 0.0:
            t_wrapped = -t
            p_wrapped += np.pi
        if t > np.pi:
            t_wrapped = 2.0*np.pi - t
            p_wrapped -= np.pi
    
        # Wrap longitude
        p_wrapped = self.wrap_longitude(p_wrapped) 
        #p_coordinates[0] + np.remainder(p_wrapped - p_coordinates[0], 2.0*np.pi)               

        return r, t_wrapped, p_wrapped    
    
    def value(self, coordinate):

        wrapped_coordinate = self.wrap(coordinate)
        
        v1 = self.components[0].value(wrapped_coordinate)
        v2 = self.components[1].value(wrapped_coordinate)
        v3 = self.components[2].value(wrapped_coordinate)
        
        return np.array((v1, v2, v3))

    def value_nowrap(self, coordinate):
        
        v1 = self.components[0].value(coordinate)
        v2 = self.components[1].value(coordinate)
        v3 = self.components[2].value(coordinate)
        
        return np.array((v1, v2, v3))

    def to_cartesian(self, spherical_coordinate):
        # sph to cart 
        
        r, t, p = spherical_coordinate
        
        r_sin_t = r*np.sin(t)
        
        return np.array((r_sin_t*np.cos(p), r_sin_t*np.sin(p), r*np.cos(t)))

    def to_spherical(self, cartesian_coordinate):
        # cart to sph
        
        x, y, z = cartesian_coordinate
        
        # Transform Cartesian coordinate to spherical
        r = np.sqrt(x*x + y*y + z*z)
        t = np.arccos(z/r)        
        p = self.wrap_longitude(np.arctan2(y, x))
        
        return np.array((r, t, p))
        
    def cartesian_unit_vector(self, cartesian_coordinate):

        # Get spherical coordinate
        r, t, p = self.to_spherical(cartesian_coordinate)        
        
        # Compute field vector
        # Since the coordinate are wrapped by the above, no need to wrap again
        vr, vt, vp = self.value_nowrap((r, t, p))

        # Convert to Cartesian basis
        sin_t = np.sin(t)
        cos_t = np.cos(t)
        sin_p = np.sin(p)
        cos_p = np.cos(p)
        
        vx = (vr*sin_t + vt*cos_t)*cos_p - vp*sin_p
        vy = (vr*sin_t + vt*cos_t)*sin_p + vp*cos_p
        vz = vr*cos_t - vt*sin_t
        
        # Magnitude of the vector
        vabs = np.sqrt(vr*vr + vt*vt + vp*vp)

        # Return unit vector
        return np.array((vx/vabs, vy/vabs, vz/vabs))

    def cell_size(self, p):

        # Mesh coordinates (same for all three components)
        crds = self.components[0].coordinates

        # Get wrapped point
        #p_wrapped = self.wrap(p)
        
        # Cell index of the point
        i, j, k = self.components[0].index(p)

        rC = 0.5*(crds[0][i+1] + crds[0][i])
        tC = 0.5*(crds[1][j+1] + crds[1][j])
        
        dr = crds[0][i+1] - crds[0][i]
        dt = crds[1][j+1] - crds[1][j]
        dp = crds[2][k+1] - crds[2][k]

        return np.array((dr, rC*dt, rC*np.sin(tC)*dp))