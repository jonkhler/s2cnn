#pylint: disable=R,C,E1101
import torch

def s2_mm(x, y):
    '''
    :param x: [l * m,     batch,      feature_in,  complex]
    :param y: [l * m,     feature_in, feature_out, complex]
    :return:  [l * m * n, batch,      feature_out, complex]
    '''
    from s2cnn.utils.complex import complex_mm

    assert y.size(3) == 2
    assert x.size(3) == 2
    nbatch = x.size(1)
    nfeature_in = x.size(2)
    nfeature_out = y.size(2)
    assert y.size(1) == nfeature_in
    nspec = x.size(0)
    assert y.size(0) == nspec

    nl = round(nspec**0.5)

    Fz_list = []
    begin = 0
    for l in range(nl):
        L = 2 * l + 1
        size = L

        Fx = x[begin:begin+size] # [m, batch,      feature_in,  complex]
        Fy = y[begin:begin+size] # [m, feature_in, feature_out, complex]

        Fx = Fx.view(L * nbatch, nfeature_in, 2) # [m * batch, feature_in, complex]

        Fy = Fy.transpose(0, 1) # [feature_in, m, feature_out, complex]
        Fy = Fy.contiguous()
        Fy = Fy.view(nfeature_in, L * nfeature_out, 2) # [feature_in, m * feature_out, complex]

        Fz = complex_mm(Fx, Fy, conj_y=True) # [m_x * batch, m_y * feature_out, complex] m_x -> m, m_y -> n
        Fz = Fz.view(L, nbatch, L, nfeature_out, 2) # [m, batch, n, feature_out, complex]
        Fz = Fz.transpose(1, 2) # [m, n, batch, feature_out, complex]
        Fz = Fz.contiguous()
        Fz = Fz.view(L * L, nbatch, nfeature_out, 2) # [m * n, batch, feature_out, complex]

        Fz_list.append(Fz)

        begin += size

    z = torch.cat(Fz_list, 0) # [l * m * n, batch, feature_out, complex]
    return z
