# This file is part of CIDER.
#
# Copyright 2023 CIDER developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""Poisson solver in a spherical shell
"""

import numpy as np
import scipy.fftpack

import cider.solvers.tridiagonal


class SphericalShellPoissonSolver:
    """Poisson solution in a spherical shell.

    Solves the Poisson equation nabla^2 u = f in a spherical
    shell, i.e. a domain bounded by two spheres. Neumann boundary conditions are 
    assumed to be given at both the inner and outer sphere.
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
        modes = scipy.fftpack.rfftfreq(self.num_lon)*self.num_lon

        for k in range(self.num_lon):
            
            lambda_k = 2.0*(np.cos(modes[k]*self.dp) - 1.0)

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
        
        # Modify coefficients at boundary planes due to boundary conditions
        AM[ 0] = 0.0
        CM[-1] = 0.0
        
        AN[ 0] = 0.0
        CN[-1] = 0.0

        if self.inner_sphere_neumann:
            BM[ 0] += self.ax[ 0]
        
        if self.outer_sphere_neumann:
            BM[-1] += self.cx[-1]
        
        BN[ 0] += self.ay[ 0]
        BN[-1] += self.cy[-1]
            
        return self.blktri.solve(AM, BM, CM, AN, BN, CN, F)
        
    def inverse_transform(self, data):
        return scipy.fftpack.irfft(data)
        
    def transform(self, data):
        return scipy.fftpack.rfft(data)
        
    def compute(self, inner_sphere, outer_sphere, rhs=None):

        # Parse boundary data
        inner_sphere_type, inner_sphere_data = inner_sphere
        outer_sphere_type, outer_sphere_data = outer_sphere

        if inner_sphere_type.lower() == "neumann":
            self.inner_sphere_neumann = True
        else:
            self.inner_sphere_neumann = False

        if outer_sphere_type.lower() == "neumann":
            self.outer_sphere_neumann = True
        else:
            self.outer_sphere_neumann = False

        # RHS of the system of equations
        if rhs is None:
            F = np.zeros((len(self.r_centers), len(self.t_centers), self.num_lon))
        else:
            F = np.copy(rhs)
        
        for i, r in enumerate(self.r_centers):
            F[i, :, :] *= r*r
        
        # Modify RHS to include non-homogenous Neumann boundary data        
        F[ 0, :, :] += inner_sphere_data*self.ax[ 0]*self.dr
        F[-1, :, :] -= outer_sphere_data*self.cx[-1]*self.dr
        
        # Transform to Fourier space
        self.F = self.transform(F)

        # Solve each mode independently        
        self.u = np.zeros(F.shape)
                
        for k in range(self.num_lon):
            self.u[:, :, k] = self.solve_mode(self.F[:, :, k], self.lambda_coeff[:, k])

        # Transform back
        self.u = self.inverse_transform(self.u)



class SphericalShellStretchedPoissonSolver:
    """Poisson solution in a spherical shell.

    Solves the Poisson equation v = f nabla u = f in a spherical
    shell, i.e. a domain bounded by two spheres. Neumann boundary conditions are 
    assumed to be given at both the inner and outer sphere.
    """
    
    def __init__(self, mesh, stretch):
        
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
        
        # Instantiate stretch object
        self.f = lambda r : stretch.f(r)

        # Number of cells in longitude
        self.num_lon = len(p_edges) - 1
        
        # Compute static coefficient arrays    

        # U_{i-1, j} coefficient
        self.ax = ((self.r_centers - 0.5*self.dr)/self.dr)**2
        self.ax *= (self.f(self.r_centers - 0.5*self.dr)/self.f(self.r_centers))

        # U_{i+1, j} coefficient
        self.cx = ((self.r_centers + 0.5*self.dr)/self.dr)**2
        self.cx *= (self.f(self.r_centers + 0.5*self.dr)/self.f(self.r_centers))

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
        modes = scipy.fftpack.rfftfreq(self.num_lon)*self.num_lon

        for k in range(self.num_lon):
            
            lambda_k = 2.0*(np.cos(modes[k]*self.dp) - 1.0)

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
        
        # Modify coefficients at boundary planes due to boundary conditions
        AM[ 0] = 0.0
        CM[-1] = 0.0
        
        AN[ 0] = 0.0
        CN[-1] = 0.0

        if self.inner_sphere_neumann:
            BM[ 0] += self.ax[ 0]
        
        if self.outer_sphere_neumann:
            BM[-1] += self.cx[-1]
        
        BN[ 0] += self.ay[ 0]
        BN[-1] += self.cy[-1]
            
        return self.blktri.solve(AM, BM, CM, AN, BN, CN, F)
        
    def inverse_transform(self, data):
        return scipy.fftpack.irfft(data)
        
    def transform(self, data):
        return scipy.fftpack.rfft(data)
        
    def compute(self, inner_sphere, outer_sphere, rhs=None):

        # Parse boundary data
        inner_sphere_type, inner_sphere_data = inner_sphere
        outer_sphere_type, outer_sphere_data = outer_sphere

        if inner_sphere_type.lower() == "neumann":
            self.inner_sphere_neumann = True
        else:
            self.inner_sphere_neumann = False

        if outer_sphere_type.lower() == "neumann":
            self.outer_sphere_neumann = True
        else:
            self.outer_sphere_neumann = False

        # RHS of the system of equations
        if rhs is None:
            F = np.zeros((len(self.r_centers), len(self.t_centers), self.num_lon))
        else:
            F = np.copy(rhs)
        
        for i, r in enumerate(self.r_centers):
            F[i, :, :] *= r*r
        
        # Modify RHS to include non-homogenous Neumann boundary data        
        F[ 0, :, :] += inner_sphere_data*self.ax[ 0]*self.dr
        F[-1, :, :] -= outer_sphere_data*self.cx[-1]*self.dr
        
        # Transform to Fourier space
        self.F = self.transform(F)

        # Solve each mode independently        
        self.u = np.zeros(F.shape)
                
        for k in range(self.num_lon):
            self.u[:, :, k] = self.solve_mode(self.F[:, :, k], self.lambda_coeff[:, k])

        # Transform back
        self.u = self.inverse_transform(self.u)
