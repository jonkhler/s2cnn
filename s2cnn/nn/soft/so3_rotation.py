#pylint: disable=C,R,E1101
import torch
import numpy as np

from s2cnn.nn.soft.gpu import so3_fft
from s2cnn.utils.complex_utils import complex_mm
from functools import lru_cache

def so3_rotation(x, alpha, beta, gamma):
    '''
    :param x: [..., beta, alpha, gamma] (..., 2b, 2b, 2b)
    '''
    b = x.size()[-1] // 2
    x_size = x.size()

    Us = setup_so3_rotation(b, alpha, beta, gamma, x.get_device() if x.is_cuda else None)
    if isinstance(x, torch.autograd.Variable):
        Us = [torch.autograd.Variable(i) for i in Us]

    # fourier transform
    if isinstance(x, torch.autograd.Variable):
        x = so3_fft.SO3_fft_real()(x) # [l * m * n, ..., complex]
    else:
        x = so3_fft.SO3_fft_real().forward(x) # [l * m * n, ..., complex]

    # rotated spectrum
    Fz_list = []
    begin = 0
    for l in range(b):
        L = 2 * l + 1
        size = L ** 2

        Fx = x[begin:begin+size]
        Fx = Fx.view(L, -1, 2) # [m, n * batch, complex]

        U = Us[l].view(L, L, 2) # [m, n, complex]

        Fz = complex_mm(U, Fx, conj_x=True) # [m, n * batch, complex]

        Fz = Fz.view(size, -1, 2) # [m * n, batch, complex]
        Fz_list.append(Fz)

        begin += size

    Fz = torch.cat(Fz_list, 0) # [l * m * n, batch, complex]
    if isinstance(x, torch.autograd.Variable):
        z = so3_fft.SO3_ifft_real()(Fz)
    else:
        z = so3_fft.SO3_ifft_real().forward(Fz)

    z = z.contiguous()
    z = z.view(*x_size)

    return z

@lru_cache(maxsize=32)
def setup_so3_rotation(b, alpha, beta, gamma, cuda_device=None):
    from lie_learn.representations.SO3.wigner_d import wigner_D_matrix

    Us = [wigner_D_matrix(l, alpha, beta, gamma,
                    field='complex', normalization='quantum', order='centered', condon_shortley='cs')
                    for l in range(b)]
    # Us[l][m, n] = exp(i m alpha) d^l_mn(beta) exp(i n gamma)

    Us = [Us[l].astype(np.complex64).view(np.float32).reshape((2 * l + 1, 2 * l + 1, 2)) for l in range(b)]

    # convert to torch Tensor
    Us = [torch.from_numpy(U) for U in Us]

    if cuda_device is not None:
        Us = [U.cuda(cuda_device) for U in Us]

    return Us
