# pylint: disable=R,C,E1101
import numpy as np
import warnings


def so3_near_identity_grid(max_beta=np.pi / 8, max_gamma=np.pi / 8, n_alpha=8, n_beta=3, n_gamma=3):
    '''
    :return: rings of rotations around the identity, all points (rotations) in
    a ring are at the same distance from the identity
    size of the kernel = n_alpha * n_beta * n_gamma
    '''
    beta = np.arange(start=1, stop=n_beta + 1, dtype=np.float) * max_beta / n_beta
    alpha = np.linspace(start=0, stop=2 * np.pi, num=n_alpha, endpoint=False)
    pre_gamma = np.linspace(start=-max_gamma, stop=max_gamma, num=n_gamma, endpoint=True)
    B, A, preC = np.meshgrid(beta, alpha, pre_gamma, indexing='ij')
    C = preC - A
    B = B.flatten()
    A = A.flatten()
    C = C.flatten()
    grid = np.stack((B, A, C), axis=1)
    if sum(grid[:, 0] == 0) > 1:
        warnings.warn("Gimbal lock: beta take value 0 in the grid")
    return tuple(tuple(bac) for bac in grid)


def so3_equatorial_grid(max_beta=0, max_gamma=np.pi / 8, n_alpha=32, n_beta=1, n_gamma=2):
    '''
    :return: rings of rotations around the equator.
    size of the kernel = n_alpha * n_beta * n_gamma
    '''
    beta = np.linspace(start=np.pi/2 - max_beta, stop=np.pi/2 + max_beta, num=n_beta, endpoint=True)
    alpha = np.linspace(start=0, stop=2 * np.pi, num=n_alpha, endpoint=False)
    gamma = np.linspace(start=-max_gamma, stop=max_gamma, num=n_gamma, endpoint=True)
    B, A, C = np.meshgrid(beta, alpha, gamma, indexing='ij')
    B = B.flatten()
    A = A.flatten()
    C = C.flatten()
    grid = np.stack((B, A, C), axis=1)
    if sum(grid[:, 0] == 0) > 1:
        warnings.warn("Gimbal lock: beta take value 0 in the grid")
    return tuple(tuple(bac) for bac in grid)


def so3_soft_grid(b):
    beta = (np.arange(2 * b) + 0.5) / (2 * b) * np.pi
    alpha = gamma = np.linspace(start=0, stop=2 * np.pi, num=2 * b, endpoint=False)
    B, A, C = np.meshgrid(beta, alpha, gamma, indexing='ij')
    B = B.flatten()
    A = A.flatten()
    C = C.flatten()
    grid = np.stack((B, A, C), axis=1)
    return tuple(tuple(bac) for bac in grid)
