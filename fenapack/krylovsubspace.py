"""
BSD 3-Clause License

Copyright (c) 2020, Fabio Durastante, Stefano Cipolla
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
# This file contains the routines for the computation of the different Krylov
# subspaces

import numpy as np


def arnoldi_iteration(a, b, eta, n: int):
    """Computes a basis of the (n + 1)-Krylov subspace of A: the space
    spanned by {b, Ab, ..., A^n b}.

    Arguments
    :param a: m × m array
    :param b: initial vector (length m)
    :param eta: residual on basis expansion
    :param n: dimension of Krylov subspace, must be >= 1

    Returns
    :return  Q: m x (n + 1) array, the columns are an orthonormal basis of the
        Krylov subspace.
    :return  h: (n + 1) x n array, A on basis Q. It is upper Hessenberg.
    :return  k: actual size of the krylov subspace
    """
    m = a.shape[0]
    h = np.zeros((n + 1, n))
    Q = np.zeros((m, n + 1))
    nb = np.linalg.norm(b, 2)
    q = b / nb   # Normalize the input vector
    Q[:, 0] = q  # Use it as the first Krylov vector
    eps = min(eta*nb, 1.0E-1)  # If v is shorter than this threshold we stop expanding
    k = 0  # To avoid uninitialized k message on output
    for k in range(n):
        v = a.dot(q)  # Generate a new candidate vector
        for j in range(k+1):  # Subtract the projections on previous vectors
            h[j, k] = np.dot(Q[:, j].conj(), v)
            v = v - h[j, k] * Q[:, j]
        if k >= 1:
            y = np.linalg.solve(h[0:k, 0:k], np.concatenate(([1], np.zeros(k-1))))
        else:
            y = [1]
        h[k + 1, k] = np.linalg.norm(v, 2)
        if h[k + 1, k] * np.abs(y[-1]) <= eps and k > 1:  # If that happens, stop iterating.
            return Q[:, 0:k], h[0:k, 0:k + 1], k
        else:
            q = v / h[k + 1, k]  # Add the produced vector to the list, unless
            Q[:, k + 1] = q      # the zero vector is produced.
    return Q[:, 0:n], h[0:n, 0:n], n