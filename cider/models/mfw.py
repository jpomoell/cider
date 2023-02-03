# This file is part of CIDER.
#
# Copyright 2023 CIDER developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""The PFSS model
"""

import numpy as np
import pysmsh
import cider.solvers.poisson_shell


class MagnetofrictionalWindModel:
    """Magnetofrictional coronal model including a solar wind outflow.
    """

    def __init__(self, magnetogram, r_edges, flow, lon_start=0.0):

        # Store reference to the magnetogram
        self.magnetogram = magnetogram

        # Store flow object
        self.flow = flow

        # Number of pixels of the input map equals the resolution of the computational mesh
        num_lat_pixels, num_lon_pixels = self.magnetogram.data.shape
    
        # Create the mesh on which the model is computed
        # TODO: Check that r coords are uniform

        self.mesh \
            = pysmsh.Mesh.Rectilinear({"r" : np.copy(r_edges),
                                       "clt" : np.linspace(0.0, np.pi, num_lat_pixels+1),
                                       "lon" : np.linspace(lon_start, lon_start + 360.0, num_lon_pixels + 1)*np.pi/180.0})
    
        # Instantiate solver
        self.solver = cider.solvers.poisson_shell.SphericalShellStretchedPoissonSolver(self.mesh, flow)

    def compute(self):

        input_Br = np.flipud(self.magnetogram.data)/self.flow.f(self.mesh.edges.r[0])
        
        self.solver.compute(("Neumann", input_Br),
                            ("Dirichlet", np.zeros(input_Br.shape)))

    def Br(self):
        """Returns the radial magnetic field component
        """

        Br = np.zeros((self.mesh.num_cells[0]+1, self.mesh.num_cells[1], self.mesh.num_cells[2]))
        dr = self.mesh.spacing.r[0]

        for i in range(1, self.mesh.num_cells[0]):
            Br[i, :, :] = (self.solver.u[i, :, :] - self.solver.u[i-1, :, :])/dr

        for i, r in enumerate(self.mesh.edges.r):
            Br[i, :, :] *= self.flow.f(r)

        Br[ 0, :, :] = np.flipud(self.magnetogram.data)
        Br[-1, :, :] = -self.flow.f(self.mesh.edges.r[-1])*self.solver.u[-1, :, :]/dr

        return Br

    def Bt(self):
        """Returns the co-latitudinal (theta) magnetic field component
        """

        Bt = np.zeros((self.mesh.num_cells[0], self.mesh.num_cells[1]+1, self.mesh.num_cells[2]))
        dt = self.mesh.spacing.clt[0]

        for i, r in enumerate(self.mesh.face_center_coords(1).r):
            Bt[i, 1:-1, :] = (self.solver.u[i, 1::, :] - self.solver.u[i, 0:-1, :])/(r*dt)

        Bt[:,  0, :] = 2.0*Bt[:,  1, :] - Bt[:,  2, :]
        Bt[:, -1, :] = 2.0*Bt[:, -2, :] - Bt[:, -3, :]

        for i, r in enumerate(self.mesh.centers.r):
            Bt[i, :, :] *= self.flow.f(r)

        return Bt

    def Bp(self):
        """Returns the longitudinal (phi) magnetic field component
        """

        Bp = np.zeros((self.mesh.num_cells[0], self.mesh.num_cells[1], self.mesh.num_cells[2]+1))
        dp = self.mesh.spacing.lon[0]

        Bp[:, :, 1:-1] = (self.solver.u[:, :, 1::] - self.solver.u[:, :, 0:-1])/(dp)

        for i, r in enumerate(self.mesh.centers.r):
            Bp[i, :, :] *= 1.0/r

        for j, clt in enumerate(self.mesh.centers.clt):
            Bp[:, j, :] *= 1.0/np.sin(clt)

        for i, r in enumerate(self.mesh.face_center_coords(2).r):
            for j, clt in enumerate(self.mesh.face_center_coords(2).clt):
                Bp[i, j, 0] = (self.solver.u[i, j, 0] - self.solver.u[i, j, -1])/(r*np.sin(clt)*dp)

        Bp[:, :, -1] = Bp[:, :, 0]

        for i, r in enumerate(self.mesh.centers.r):
            Bp[i, :, :] *= self.flow.f(r)

        return Bp

    def magnetic_field(self):
        """Returns the magnetic field computed on the mesh
        """
        
        magnetic_field = pysmsh.Field.Vector(self.mesh, "face_staggered")

        magnetic_field.data[0][:, :, :] = self.Br()
        magnetic_field.data[1][:, :, :] = self.Bt()
        magnetic_field.data[2][:, :, :] = self.Bp()
        
        return magnetic_field



class HyperbolicTanFlowProfile:
    
    def __init__(self):
        
        import sunpy.sun.constants

        # MF coefficient
        self.nu0 = 4.0e-14

        # Terminal wind speed
        self.v1 = 500.0e3
        
        # Width of transition
        self.w = 2.0*sunpy.sun.constants.radius.value
        
        # Radius at which the average speed is obtain
        # In practice, if v0 is ~0, this is close to v1/2
        self.r1 = 4.0*sunpy.sun.constants.radius.value
         
        # Compute wind speed shift in lower corona
        # based on specifying a target wind speed at r=RSun
        v_at_boundary = 1e3
        
        tanh = np.tanh((sunpy.sun.constants.radius.value - self.r1)/self.w)
        
        self.v0 = (2.0*v_at_boundary - self.v1*(1.0 + tanh))/(1.0 - tanh)
                
    def f(self, r):

        log_tanh_term = np.log(np.tanh((r-self.r1)/self.w) + 1.0)

        integral_v_dr = 0.5*r*(self.v1 + self.v0) + 0.5*(self.v1 - self.v0)*(r - self.w*log_tanh_term)
   
        return np.exp(self.nu0*integral_v_dr)

    def vr(self, r):
        
        return 0.5*(self.v1 + self.v0) + 0.5*(self.v1 - self.v0)*np.tanh((r-self.r1)/self.w)