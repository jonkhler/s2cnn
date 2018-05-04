#pylint: disable=R,C,E1101
import torch

def so3_mm(x, y):
    '''
    :param x: [l * m * n,   batch,    feature_in,  complex]
    :param y: [l * m * n, feature_in, feature_out, complex]
    :return:  [l * m * n,   batch,    feature_out, complex]
    '''
    from s2cnn.utils.complex import complex_mm
    import math

    assert y.size(3) == 2
    assert x.size(3) == 2
    nbatch = x.size(1)
    nfeature_in = x.size(2)
    nfeature_out = y.size(2)
    assert y.size(1) == nfeature_in
    nspec = x.size(0)
    assert y.size(0) == nspec
    nl = math.ceil((3/4 * nspec)**(1/3))
    assert nspec == nl * (4 * nl**2 - 1) // 3

    Fz_list = []
    begin = 0
    for l in range(nl):
        L = 2 * l + 1
        size = L ** 2

        Fx = x[begin:begin+size] # [m * n,   batch,    feature_in,  complex]
        Fy = y[begin:begin+size] # [m * n, feature_in, feature_out, complex]

        Fx = Fx.view(L, L, nbatch, nfeature_in, 2) # [m, n, batch, feature_in, complex]
        Fx = Fx.transpose(0, 1) # [n, m, batch, feature_in, complex]
        Fx = Fx.transpose(0, 2) # [batch, m, n, feature_in, complex]
        Fx = Fx.transpose(2, 3) # [batch, m, feature_in, n, complex]
        Fx = Fx.contiguous()
        Fx = Fx.view(nbatch * L, nfeature_in * L, 2) # [batch * m, feature_in * n, complex]

        Fy = Fy.view(L, L, nfeature_in, nfeature_out, 2) # [m, n, feature_in, feature_out, complex]
        Fy = Fy.transpose(0, 2) # [feature_in, n, m, feature_out, complex]
        Fy = Fy.contiguous()
        Fy = Fy.view(nfeature_in * L, L * nfeature_out, 2) # [feature_in * n, m * feature_out, complex]

        Fz = complex_mm(Fx, Fy, conj_y=True) # [batch * m_x, m_y * feature_out, complex] m_x -> m, m_y -> n
        Fz = Fz.view(nbatch, L * L, nfeature_out, 2) # [batch, m * n, feature_out, complex]
        Fz = Fz.transpose(0, 1) # [m * n, batch, feature_out, complex]

        Fz_list.append(Fz)

        begin += size

    z = torch.cat(Fz_list, 0) # [l * m * n, batch, feature_out, complex]
    return z
