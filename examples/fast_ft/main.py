# pylint: disable=C,R,E1101,E1102,W0621
'''
Compare so3_ft with so3_fft
'''
import time
import torch


b = 6  # bandwidth
# random input data to be Fourier Transform
x = torch.randn(2 * b, 2 * b, 2 * b, dtype=torch.float, device="cuda")  # [beta, alpha, gamma]


# Fast version
from s2cnn.soft.gpu.so3_fft import so3_rfft
t = time.perf_counter()

y1 = so3_rfft(x)

print("so3_rfft: {}s".format(time.perf_counter() - t))


# Equivalent version but using the naive version
from s2cnn import so3_rft, so3_soft_grid
import lie_learn.spaces.S3 as S3
t = time.perf_counter()

# so3_ft computes a non weighted Fourier transform
weights = torch.tensor(S3.quadrature_weights(b), dtype=torch.float, device="cuda")
x = torch.einsum("bac,b->bac", (x, weights))

y2 = so3_rft(x.view(-1), b, so3_soft_grid(b))

print("so3_ft: {}s".format(time.perf_counter() - t))


# Compare values
assert (y1 - y2).abs().max().item() < 1e-4 * y1.abs().mean().item()
