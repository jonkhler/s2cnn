# pylint: disable=R,C,E1101
import numpy as np
import warnings


def so3_near_identity_grid(max_beta=np.pi / 8, max_gamma=np.pi / 8, n_alpha=8, n_beta=3, n_gamma=3):
    '''
    :return: rings of rotations around the identity, all points (rotations) in
    a ring are at the same distance from the identity
    size of the kernel = n_alpha * n_beta * n_gamma
    '''
    alpha = np.linspace(start=0, stop=2 * np.pi, num=n_alpha, endpoint=False)
    beta = np.arange(start=1, stop=n_beta + 1, dtype=np.float) * max_beta / n_beta
    pre_gamma = np.linspace(start=-max_gamma, stop=max_gamma, num=n_gamma, endpoint=True)
    A, B, preC = np.meshgrid(alpha, beta, pre_gamma, indexing='ij')
    C = preC - A
    A = A.flatten()
    B = B.flatten()
    C = C.flatten()
    grid = np.stack((A, B, C), axis=1)
    if sum(grid[:, 1] == 0) > 1:
        warnings.warn("Gimbal lock: beta take value 0 in the grid")
    return tuple(tuple(abc) for abc in grid)


def so3_equatorial_grid(max_beta=0, max_gamma=np.pi / 8, n_alpha=32, n_beta=1, n_gamma=2):
    '''
    :return: rings of rotations around the equator.
    size of the kernel = n_alpha * n_beta * n_gamma
    '''
    alpha = np.linspace(start=0, stop=2 * np.pi, num=n_alpha, endpoint=False)
    beta = np.linspace(start=np.pi/2 - max_beta, stop=np.pi/2 + max_beta, num=n_beta, endpoint=True)
    gamma = np.linspace(start=-max_gamma, stop=max_gamma, num=n_gamma, endpoint=True)
    A, B, C = np.meshgrid(alpha, beta, gamma, indexing='ij')
    A = A.flatten()
    B = B.flatten()
    C = C.flatten()
    grid = np.stack((A, B, C), axis=1)
    if sum(grid[:, 1] == 0) > 1:
        warnings.warn("Gimbal lock: beta take value 0 in the grid")
    return tuple(tuple(abc) for abc in grid)


def so3_soft_grid(b):
    alpha = np.linspace(start=0, stop=2 * np.pi, num=2 * b, endpoint=False)
    beta = (np.arange(2 * b) + 0.5) / (2 * b) * np.pi
    gamma = alpha
    A, B, C = np.meshgrid(alpha, beta, gamma, indexing='ij')
    A = A.flatten()
    B = B.flatten()
    C = C.flatten()
    grid = np.stack((A, B, C), axis=1)
    return tuple(tuple(abc) for abc in grid)
