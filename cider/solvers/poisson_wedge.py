# This file is part of CIDER.
#
# Copyright 2023 CIDER developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""Poisson solver in a spherical wedge domain
"""

import numpy as np
import scipy.fftpack

import cider.solvers.tridiagonal


class SphericalWedgePoissonSolver:
    """Poisson solution in a spherical wedge.

    Solves the Poisson equation nabla^2 u = f in a spherical
    wedge-like domain. Neumann boundary conditions are 
    assumed for all edges of the domain.
    """
    
    def __init__(self, mesh):
        
        # Grid coordinates
        r_edges, t_edges, p_edges = mesh.edges

        # Constant grid spacings
        # TODO: Check that the grid is uniform
        self.dr = r_edges[1] - r_edges[0]
        self.dt = t_edges[1] - t_edges[0]
        self.dp = p_edges[1] - p_edges[0]

        # Grid cell centers
        self.r_centers = 0.5*(r_edges[1::] + r_edges[0:-1])
        self.t_centers = 0.5*(t_edges[1::] + t_edges[0:-1])
        
        # Number of cells in longitude
        self.num_lon = len(p_edges) - 1
        
        # Compute static coefficient arrays    

        # U_{i-1, j} coefficient
        self.ax = ((self.r_centers - 0.5*self.dr)/self.dr)**2

        # U_{i+1, j} coefficient
        self.cx = ((self.r_centers + 0.5*self.dr)/self.dr)**2

        # U_{i, j} coefficient, i dependence
        self.bx = -(self.ax + self.cx)

        # U_{i, j-1} coefficient
        self.ay = np.sin(self.t_centers - 0.5*self.dt)/(np.sin(self.t_centers)*self.dt**2)

        # U_{i, j+1} coefficient
        self.cy = np.sin(self.t_centers + 0.5*self.dt)/(np.sin(self.t_centers)*self.dt**2)
         
        # U_{i, j} coefficient, j dependence
        self.by = -(self.ay + self.cy)
        
        # Coefficient with lambda_m
        self.lambda_coeff = np.zeros((len(self.t_centers), self.num_lon))
        
        for k in range(self.num_lon):
            
            lambda_k = 2.0*(np.cos(np.pi*(k)/self.num_lon) - 1.0)

            self.lambda_coeff[:, k] = lambda_k/(np.sin(self.t_centers)*self.dp)**2

        # Block tridiagonal matrix solver
        self.blktri = cider.solvers.tridiagonal.BlockTridiagonalSolver()

    def solve_mode(self, F, lambda_m):
                
        AM = np.copy(self.ax)
        BM = np.copy(self.bx)
        CM = np.copy(self.cx)

        AN = np.copy(self.ay)
        BN = np.copy(self.by) + lambda_m
        CN = np.copy(self.cy)
        
        # Modify coefficients at boundary planes for Neumann boundary conditions
        AM[ 0] = 0.0
        CM[-1] = 0.0
        
        AN[ 0] = 0.0
        CN[-1] = 0.0

        BM[ 0] += self.ax[ 0]
        BM[-1] += self.cx[-1]
        
        BN[ 0] += self.ay[ 0]
        BN[-1] += self.cy[-1]
            
        return self.blktri.solve(AM, BM, CM, AN, BN, CN, F)
        
    def inverse_transform(self, data):
        return scipy.fftpack.idct(data, type=2, norm='ortho')
    
    def transform(self, data):
        return scipy.fftpack.dct(data, type=2, norm='ortho')
    
    def compute(self, boundary_data, rhs):

        # RHS of the system of equations
        F = np.copy(rhs)
        
        for i, r in enumerate(self.r_centers):
            F[i, :, :] *= r*r
        
        # Modify RHS to include non-homogenous Neumann boundary data        
        F[ 0, :, :] += boundary_data[0]*self.ax[ 0]*self.dr
        F[-1, :, :] -= boundary_data[1]*self.cx[-1]*self.dr
        
        for i, r in enumerate(self.r_centers):
            F[i,  0, :] += boundary_data[2][i, :]*self.ay[ 0]*self.dt*r
            F[i, -1, :] -= boundary_data[3][i, :]*self.cy[-1]*self.dt*r
        
        for i, r in enumerate(self.r_centers):
            for j, t in enumerate(self.t_centers):
                F[i, j,  0] += boundary_data[4][i, j]*r/(np.sin(t)*self.dp)
                F[i, j, -1] -= boundary_data[5][i, j]*r/(np.sin(t)*self.dp)

        # Transform to Fourier space
        self.F = self.transform(F)

        # Solve each mode independently        
        self.u = np.zeros(rhs.shape)
                
        for k in range(self.num_lon):
            self.u[:, :, k] = self.solve_mode(self.F[:, :, k], self.lambda_coeff[:, k])

        # Transform back
        self.u = self.inverse_transform(self.u)
        