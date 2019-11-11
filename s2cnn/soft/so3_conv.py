# pylint: disable=C,R,E1101
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules import Module

from .so3_fft import SO3_fft_real, SO3_ifft_real
from s2cnn import so3_mm
from s2cnn import so3_rft


class SO3Convolution(Module):
    def __init__(self, nfeature_in, nfeature_out, b_in, b_out, grid):
        '''
        :param nfeature_in: number of input fearures
        :param nfeature_out: number of output features
        :param b_in: input bandwidth (precision of the input SOFT grid)
        :param b_out: output bandwidth
        :param grid: points of the SO(3) group defining the kernel, tuple of (alpha, beta, gamma)'s
        '''
        super(SO3Convolution, self).__init__()
        self.nfeature_in = nfeature_in
        self.nfeature_out = nfeature_out
        self.b_in = b_in
        self.b_out = b_out
        self.grid = grid
        self.kernel = Parameter(torch.empty(nfeature_in, nfeature_out, len(grid)).uniform_(-1, 1))
        self.bias = Parameter(torch.zeros(1, nfeature_out, 1, 1, 1))

        # When useing ADAM optimizer, the variance of each componant of the gradient
        # is normalized by ADAM around 1.
        # Then it is suited to have parameters of order one.
        # Therefore the scaling, needed for the proper forward propagation, is done "outside" of the parameters
        self.scaling = 1. / math.sqrt(len(self.grid) * self.nfeature_in * (self.b_out ** 3.) / (self.b_in ** 3.))

    def forward(self, x):  # pylint: disable=W
        '''
        :x:      [batch, feature_in,  beta, alpha, gamma]
        :return: [batch, feature_out, beta, alpha, gamma]
        '''
        assert x.size(1) == self.nfeature_in
        assert x.size(2) == 2 * self.b_in
        assert x.size(3) == 2 * self.b_in
        assert x.size(4) == 2 * self.b_in

        x = SO3_fft_real.apply(x, self.b_out)  # [l * m * n, batch, feature_in, complex]
        y = so3_rft(self.kernel * self.scaling, self.b_out, self.grid)  # [l * m * n, feature_in, feature_out, complex]
        assert x.size(0) == y.size(0)
        assert x.size(2) == y.size(1)
        z = so3_mm(x, y)  # [l * m * n, batch, feature_out, complex]
        assert z.size(0) == x.size(0)
        assert z.size(1) == x.size(1)
        assert z.size(2) == y.size(2)
        z = SO3_ifft_real.apply(z)  # [batch, feature_out, beta, alpha, gamma]

        z = z + self.bias

        return z


class SO3Shortcut(Module):
    '''
    Useful for ResNet
    '''

    def __init__(self, nfeature_in, nfeature_out, b_in, b_out):
        super(SO3Shortcut, self).__init__()
        assert b_out <= b_in

        if (nfeature_in != nfeature_out) or (b_in != b_out):
            self.conv = SO3Convolution(
                nfeature_in=nfeature_in, nfeature_out=nfeature_out, b_in=b_in, b_out=b_out,
                grid=((0, 0, 0), ))
        else:
            self.conv = None

    def forward(self, x):  # pylint: disable=W
        '''
        :x:      [batch, feature_in,  beta, alpha, gamma]
        :return: [batch, feature_out, beta, alpha, gamma]
        '''
        if self.conv is not None:
            return self.conv(x)
        else:
            return x
