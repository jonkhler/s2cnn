#pylint: disable=C,R,E1101
import torch
from torch.autograd import Variable

def as_complex(x):
    """
    In pytorch, a complex array is represented as a real array with an extra length-2 axis at the end.
    This function takes a real-valued array x and adds complex axis where the real part is set to x and the imaginary part is set to 0.
    """
    imaginary = torch.zeros(x.size())
    if x.is_cuda:
        imaginary = imaginary.cuda(x.get_device())
    if isinstance(x, Variable):
        imaginary = Variable(imaginary)
    z = torch.stack((x, imaginary), dim=x.ndimension())
    return z

def fftshift(x, axis):
    n = x.size(axis)
    x1 = x.narrow(axis, 0, n - n // 2)
    x2 = x.narrow(axis, n - n // 2, n // 2)
    return torch.cat((x2, x1), dim=axis)

def ifftshift(x, axis):
    n = x.size(axis)
    x1 = x.narrow(axis, 0, n // 2)
    x2 = x.narrow(axis, n // 2, n - n // 2)
    return torch.cat((x2, x1), dim=axis)

def complex_bmm(x, y, conj_x=False, conj_y=False):
    '''
    :param x: [batch, i, k, complex] (nbatch, M, K, 2)
    :param y: [batch, k, j, complex] (nbatch, K, N, 2)
    :return:  [batch, i, j, complex] (nbatch, M, N, 2)
    '''
    xr = x[:, :, :, 0]
    xi = x[:, :, :, 1]

    yr = y[:, :, :, 0]
    yi = y[:, :, :, 1]

    if not conj_x and not conj_y:
        zr = torch.bmm(xr, yr) - torch.bmm(xi, yi)
        zi = torch.bmm(xr, yi) + torch.bmm(xi, yr)
    if conj_x and not conj_y:
        zr = torch.bmm(xr, yr) + torch.bmm(xi, yi)
        zi = torch.bmm(xr, yi) - torch.bmm(xi, yr)
    if not conj_x and conj_y:
        zr = torch.bmm(xr, yr) + torch.bmm(xi, yi)
        zi = torch.bmm(xi, yr) - torch.bmm(xr, yi)
    if conj_x and conj_y:
        zr = torch.bmm(xr, yr) - torch.bmm(xi, yi)
        zi = - torch.bmm(xr, yi) - torch.bmm(xi, yr)

    return torch.stack((zr, zi), 3)

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

def complex_m(x, y, conj_x=False, conj_y=False):
    '''
    :param x: [..., complex] (..., 2)
    :param y: [..., complex] (..., 2)
    :return:  [..., complex] (..., 2)
    '''
    xr = x[..., 0]
    xi = x[..., 1]

    yr = y[..., 0]
    yi = y[..., 1]

    if not conj_x and not conj_y:
        zr = xr * yr - xi * yi
        zi = xr * yi + xi * yr
    if conj_x and not conj_y:
        zr = xr * yr + xi * yi
        zi = xr * yi - xi * yr
    if not conj_x and conj_y:
        zr = xr * yr + xi * yi
        zi = xi * yr - xr * yi
    if conj_x and conj_y:
        zr = xr * yr - xi * yi
        zi = - xr * yi - xi * yr

    return torch.stack((zr, zi), xi.ndimension())