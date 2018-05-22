# pylint: disable=R,C,E1101
import torch
import numpy as np
from functools import lru_cache
from s2cnn.utils.decorator import cached_dirpklgz


def so3_rft(x, b, grid):
    """
    Real Fourier Transform
    :param x: [..., beta_alpha_gamma]
    :param b: output bandwidth signal
    :param grid: tuple of (beta, alpha, gamma) tuples
    :return: [l * m * n, ..., complex]
    """
    # F is the Fourier matrix
    F = _setup_so3_ft(b, grid, device_type=x.device.type, device_index=x.device.index)  # [beta_alpha_gamma, l * m * n, complex]

    assert x.size(-1) == F.size(0)

    sz = x.size()
    x = torch.einsum("ia,afc->fic", (x.view(-1, x.size(-1)), F.clone()))  # [l * m * n, ..., complex]
    x = x.view(-1, *sz[:-1], 2)
    return x


@cached_dirpklgz("cache/setup_so3_ft")
def __setup_so3_ft(b, grid):
    from lie_learn.representations.SO3.wigner_d import wigner_D_matrix

    # Note: optionally get quadrature weights for the chosen grid and use them to weigh the D matrices below.
    # This is optional because we can also view the filter coefficients as having absorbed the weights already.
    # The weights depend on the spacing between the point of the grid
    # Only the coefficient sin(beta) can be added without requireing to know the spacings

    # Sample the Wigner-D functions on the local grid
    n_spatial = len(grid)
    n_spectral = np.sum([(2 * l + 1) ** 2 for l in range(b)])
    F = np.zeros((n_spatial, n_spectral), dtype=complex)
    for i, (beta, alpha, gamma) in enumerate(grid):
        Dmats = [wigner_D_matrix(l, alpha, beta, gamma,
                                 field='complex', normalization='quantum', order='centered', condon_shortley='cs')
                 .conj()
                 for l in range(b)]
        F[i] = np.hstack([Dl.flatten() for Dl in Dmats])

    # F is a complex matrix of shape (n_spatial, n_spectral)
    # If we view it as float, we get a real matrix of shape (n_spatial, 2 * n_spectral)
    # In the so3_local_ft, we will multiply a batch of real (..., n_spatial) vectors x with this matrix F as xF.
    # The result is a (..., 2 * n_spectral) array that can be interpreted as a batch of complex vectors.
    F = F.view('float').reshape((-1, n_spectral, 2))
    return F


@lru_cache(maxsize=32)
def _setup_so3_ft(b, grid, device_type, device_index):
    F = __setup_so3_ft(b, grid)

    # convert to torch Tensor
    F = torch.tensor(F.astype(np.float32), dtype=torch.float32, device=torch.device(device_type, device_index))  # pylint: disable=E1102

    return F
