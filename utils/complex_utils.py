#pylint: disable=C,R,E1101
import torch
from torch.autograd import Variable

def complex_mm(x, y, conj_x=False, conj_y=False):
    '''
    :param x: [i, k, complex] (M, K, 2)
    :param y: [k, j, complex] (K, N, 2)
    :return:  [i, j, complex] (M, N, 2)
    '''
    xr = x[:, :, 0]
    xi = x[:, :, 1]

    yr = y[:, :, 0]
    yi = y[:, :, 1]

    if not conj_x and not conj_y:
        zr = torch.mm(xr, yr) - torch.mm(xi, yi)
        zi = torch.mm(xr, yi) + torch.mm(xi, yr)
    if conj_x and not conj_y:
        zr = torch.mm(xr, yr) + torch.mm(xi, yi)
        zi = torch.mm(xr, yi) - torch.mm(xi, yr)
    if not conj_x and conj_y:
        zr = torch.mm(xr, yr) + torch.mm(xi, yi)
        zi = torch.mm(xi, yr) - torch.mm(xr, yi)
    if conj_x and conj_y:
        zr = torch.mm(xr, yr) - torch.mm(xi, yi)
        zi = - torch.mm(xr, yi) - torch.mm(xi, yr)

    return torch.stack((zr, zi), 2)
