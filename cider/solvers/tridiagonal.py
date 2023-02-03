# This file is part of CIDER.
#
# Copyright 2023 CIDER developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""Tri-diagonal matrix solvers
"""

import numpy as np

import pyfishpack.blktri

class BlockTridiagonalSolver:
    def __init__(self):
        pass

    def solve(self, AM, BM, CM, AN, BN, CN, Y):

        N = len(AN)
        M = len(AM)
        
        K = int(np.ceil(np.log2(N)))+1
        L = 2**(K+1)
        w = np.zeros((K-2)*L + K + 5 + max(2*N, 6*M))
    
        yarr = np.copy(Y, order='F')
    
        ierr1 \
            = pyfishpack.blktri.blktri(0, \
                                       1, N, AN, BN, CN, \
                                       1, M, AM, BM, CM, \
                                       yarr, w)
    
        ierr2 \
            = pyfishpack.blktri.blktri(1, \
                                       1, N, AN, BN, CN, \
                                       1, M, AM, BM, CM, \
                                       yarr, w)
    
        return np.copy(yarr, order="C")
