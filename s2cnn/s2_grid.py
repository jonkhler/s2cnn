# pylint: disable=R,C,E1101
import numpy as np


def s2_near_identity_grid(max_beta=np.pi / 8, n_alpha=8, n_beta=3):
    '''
    :return: rings around the north pole
    size of the kernel = n_alpha * n_beta
    '''
    beta = np.arange(start=1, stop=n_beta + 1, dtype=np.float) * max_beta / n_beta
    alpha = np.linspace(start=0, stop=2 * np.pi, num=n_alpha, endpoint=False)
    B, A = np.meshgrid(beta, alpha, indexing='ij')
    B = B.flatten()
    A = A.flatten()
    grid = np.stack((B, A), axis=1)
    return tuple(tuple(ba) for ba in grid)


def s2_equatorial_grid(max_beta=0, n_alpha=32, n_beta=1):
    '''
    :return: rings around the equator
    size of the kernel = n_alpha * n_beta
    '''
    beta = np.linspace(start=np.pi/2 - max_beta, stop=np.pi/2 + max_beta, num=n_beta, endpoint=True)
    alpha = np.linspace(start=0, stop=2 * np.pi, num=n_alpha, endpoint=False)
    B, A = np.meshgrid(beta, alpha, indexing='ij')
    B = B.flatten()
    A = A.flatten()
    grid = np.stack((B, A), axis=1)
    return tuple(tuple(ba) for ba in grid)


def s2_soft_grid(b):
    beta = (np.arange(2 * b) + 0.5) / (2 * b) * np.pi
    alpha = np.linspace(start=0, stop=2 * np.pi, num=2 * b, endpoint=False)
    B, A = np.meshgrid(beta, alpha, indexing='ij')
    B = B.flatten()
    A = A.flatten()
    grid = np.stack((B, A), axis=1)
    return tuple(tuple(ba) for ba in grid)
