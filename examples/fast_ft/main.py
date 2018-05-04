#pylint: disable=C,R,E1101,W0621
'''
Compare so3_ft with so3_fft
'''
import torch
from s2cnn import so3_ft, so3_soft_grid
from s2cnn.soft.gpu.so3_fft import so3_fft

b = 16
x = torch.randn(2 * b, 2 * b, 2 * b)

y1 = so3_fft(x)
y2 = so3_ft(x, b, so3_soft_grid(b))
