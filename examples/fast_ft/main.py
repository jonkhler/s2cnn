# pylint: disable=C,R,E1101,E1102,W0621
'''
Compare so3_ft with so3_fft
'''
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

b_in, b_out = 6, 6  # bandwidth
# random input data to be Fourier Transform
x = torch.randn(2 * b_in, 2 * b_in, 2 * b_in, dtype=torch.float, device=device)  # [beta, alpha, gamma]


# Fast version
from s2cnn.soft.so3_fft import so3_rfft

y1 = so3_rfft(x, b_out=b_out)


# Equivalent version but using the naive version
from s2cnn import so3_rft, so3_soft_grid
import lie_learn.spaces.S3 as S3

# so3_ft computes a non weighted Fourier transform
weights = torch.tensor(S3.quadrature_weights(b_in), dtype=torch.float, device=device)
x = torch.einsum("bac,b->bac", (x, weights))

y2 = so3_rft(x.view(-1), b_out, so3_soft_grid(b_in))


# Compare values
assert (y1 - y2).abs().max().item() < 1e-4 * y1.abs().mean().item()
