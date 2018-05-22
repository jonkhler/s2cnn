# pylint: disable=R,C,E1101
import torch
import numpy as np
from functools import lru_cache
from s2cnn.utils.decorator import cached_dirpklgz


def s2_rft(x, b, grid):
    """
    Real Fourier Transform
    :param x: [..., beta_alpha]
    :param b: output bandwidth signal
    :param grid: tuple of (beta, alpha) tuples
    :return: [l * m, ..., complex]
    """
    # F is the Fourier matrix
    F = _setup_s2_ft(b, grid, device_type=x.device.type, device_index=x.device.index)  # [beta_alpha, l * m, complex]

    assert x.size(-1) == F.size(0)

    sz = x.size()
    x = torch.einsum("ia,afc->fic", (x.view(-1, x.size(-1)), F.clone()))  # [l * m, ..., complex]
    x = x.view(-1, *sz[:-1], 2)
    return x


@cached_dirpklgz("cache/setup_s2_ft")
def __setup_s2_ft(b, grid):
    from lie_learn.representations.SO3.wigner_d import wigner_D_matrix

    # Note: optionally get quadrature weights for the chosen grid and use them to weigh the D matrices below.
    # This is optional because we can also view the filter coefficients as having absorbed the weights already.

    # Sample the Wigner-D functions on the local grid
    n_spatial = len(grid)
    n_spectral = np.sum([(2 * l + 1) for l in range(b)])
    F = np.zeros((n_spatial, n_spectral), dtype=complex)
    for i, (beta, alpha) in enumerate(grid):
        Dmats = [(2 * b) * wigner_D_matrix(l, alpha, beta, 0,
                                           field='complex', normalization='quantum', order='centered', condon_shortley='cs')
                 .conj()
                 for l in range(b)]
        F[i] = np.hstack([Dmats[l][:, l] for l in range(b)])

    # F is a complex matrix of shape (n_spatial, n_spectral)
    # If we view it as float, we get a real matrix of shape (n_spatial, 2 * n_spectral)
    # In the so3_local_ft, we will multiply a batch of real (..., n_spatial) vectors x with this matrix F as xF.
    # The result is a (..., 2 * n_spectral) array that can be interpreted as a batch of complex vectors.
    F = F.view('float').reshape((-1, n_spectral, 2))
    return F


@lru_cache(maxsize=32)
def _setup_s2_ft(b, grid, device_type, device_index):
    F = __setup_s2_ft(b, grid)

    # convert to torch Tensor
    F = torch.tensor(F.astype(np.float32), dtype=torch.float32, device=torch.device(device_type, device_index))  # pylint: disable=E1102

    return F
