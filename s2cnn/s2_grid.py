# pylint: disable=R,C,E1101
import numpy as np


def s2_near_identity_grid(max_beta=np.pi / 8, n_alpha=8, n_beta=3):
    '''
    :return: rings around the north pole
    size of the kernel = n_alpha * n_beta
    '''
    alpha = np.linspace(start=0, stop=2 * np.pi, num=n_alpha, endpoint=False)
    beta = np.arange(start=1, stop=n_beta + 1, dtype=np.float) * max_beta / n_beta
    A, B = np.meshgrid(alpha, beta, indexing='ij')
    A = A.flatten()
    B = B.flatten()
    grid = np.stack((A, B), axis=1)
    return tuple(tuple(ab) for ab in grid)


def s2_equatorial_grid(max_beta=0, n_alpha=32, n_beta=1):
    '''
    :return: rings around the equator
    size of the kernel = n_alpha * n_beta
    '''
    alpha = np.linspace(start=0, stop=2 * np.pi, num=n_alpha, endpoint=False)
    beta = np.linspace(start=np.pi/2 - max_beta, stop=np.pi/2 + max_beta, num=n_beta, endpoint=True)
    A, B = np.meshgrid(alpha, beta, indexing='ij')
    A = A.flatten()
    B = B.flatten()
    grid = np.stack((A, B), axis=1)
    return tuple(tuple(ab) for ab in grid)


def s2_soft_grid(b):
    alpha = np.linspace(start=0, stop=2 * np.pi, num=2 * b, endpoint=False)
    beta = (np.arange(2 * b) + 0.5) / (2 * b) * np.pi
    A, B = np.meshgrid(alpha, beta, indexing='ij')
    A = A.flatten()
    B = B.flatten()
    grid = np.stack((A, B), axis=1)
    return tuple(tuple(ab) for ab in grid)
