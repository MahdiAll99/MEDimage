#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Tuple

import numpy as np


def min_vol_ellipse(P: np.ndarray,
                    tolerance: np.ndarray) -> Tuple[np.ndarray,
                                                    np.ndarray]:
    """Compute min_vol_ellipse.
    
    Finds the minimum volume enclsing ellipsoid (MVEE) of a set of data
    points stored in matrix P. The following optimization problem is solved:

        minimize log(det(A)) subject to (P_i - c)' * A * (P_i - c) <= 1

    in variables A and c, where `P_i` is the `i-th` column of the matrix `P`.
    The solver is based on Khachiyan Algorithm, and the final solution
    is different from the optimal value by the pre-spesified amount of
    `tolerance`.
    
    Note:
        Adapted from MATLAB code of Nima Moshtagh (nima@seas.upenn.edu)
        University of Pennsylvania.

    Args:
        P (ndarray): (d x N) dimnesional matrix containing N points in R^d.
        tolerance (ndarray): error in the solution with respect to the optimal value.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - A: (d x d) matrix of the ellipse equation in the 'center form':
                $$(x-c)' * A * (x-c) = 1$$
                where d is shape of `P` along 0-axis.
            - c: d-dimensional vector as the center of the ellipse.

    examples:

        >>>P = rand(5,100);
        >>>[A, c] = min_vol_ellipse(P, .01)

        #To reduce the computation time, work with the boundary points only:
        >>>K = convhulln(P)
        >>>K = unique(K(:))
        >>>Q = P(:,K)
        >>>[A, c] = min_vol_ellipse(Q, .01)

    """

    # Solving the Dual problem
    # data points
    d, N = np.shape(P)
    Q = np.ones((d+1, N))
    Q[:-1, :] = P[:, :]

    # initializations
    err = 1
    u = np.ones(N)/N  # 1st iteration
    new_u = np.zeros(N)

    # Khachiyan Algorithm

    while (err > tolerance):
        diag_u = np.diag(u)
        trans_q = np.transpose(Q)
        X = Q @ diag_u @ trans_q

        # M the diagonal vector of an NxN matrix
        inv_x = np.linalg.inv(X)
        M = np.diag(trans_q @ inv_x @ Q)
        maximum = np.max(M)
        j = np.argmax(M)

        step_size = (maximum - d - 1)/((d+1)*(maximum-1))
        new_u = (1 - step_size)*u.copy()
        new_u[j] = new_u[j] + step_size
        err = np.linalg.norm(new_u - u)
        u = new_u.copy()

    # Computing the Ellipse parameters
    # Finds the ellipse equation in the 'center form':
    # (x-c)' * A * (x-c) = 1
    # It computes a dxd matrix 'A' and a d dimensional vector 'c' as the center
    # of the ellipse.
    U = np.diag(u)

    # the A matrix for the ellipse
    c = P @ u
    c = np.reshape(c, (np.size(c), 1), order='F')  # center of the ellipse

    pup_t = P @ U @ np.transpose(P)
    cct = c @ np.transpose(c)
    a_inv = np.linalg.inv(pup_t - cct)
    A = (1/d) * a_inv

    return A, c
