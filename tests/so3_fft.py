# pylint: disable=C,R,E1101,E1102,W0621
'''
Compare so3_ft with so3_fft
'''
import torch
from functools import partial

def test_so3_rfft(b_in, b_out, device):
    x = torch.randn(2 * b_in, 2 * b_in, 2 * b_in, dtype=torch.float, device=device)  # [beta, alpha, gamma]

    from s2cnn.soft.so3_fft import so3_rfft
    y1 = so3_rfft(x, b_out=b_out)

    from s2cnn import so3_rft, so3_soft_grid
    import lie_learn.spaces.S3 as S3

    # so3_ft computes a non weighted Fourier transform
    weights = torch.tensor(S3.quadrature_weights(b_in), dtype=torch.float, device=device)
    x2 = torch.einsum("bac,b->bac", (x, weights))

    y2 = so3_rft(x2.view(-1), b_out, so3_soft_grid(b_in))
    assert (y1 - y2).abs().max().item() < 1e-4 * y1.abs().mean().item() 

test_so3_rfft(7, 5, torch.device("cpu"))
# test_so3_rfft(5, 7, torch.device("cpu"))  # so3_rft introduce aliasing

if torch.cuda.is_available():
    test_so3_rfft(7, 5, torch.device("cuda:0"))
    # test_so3_rfft(5, 7, torch.device("cuda:0"))  # so3_rft introduce aliasing




def test_inverse(f, g, b_in, b_out, device, complex):
    if complex:
        x = torch.randn(2 * b_in, 2 * b_in, 2 * b_in, 2, dtype=torch.float, device=device)  # [beta, alpha, gamma]
    else:
        x = torch.randn(2 * b_in, 2 * b_in, 2 * b_in, dtype=torch.float, device=device)  # [beta, alpha, gamma]
    
    x = g(f(x, b_out=b_out), b_out=b_in)

    y = g(f(x, b_out=b_out), b_out=b_in)

    assert (x - y).abs().max().item() < 1e-4 * y.abs().mean().item() 


def test_inverse2(f, g, b_in, b_out, device):
    x = torch.randn(b_in * (4 * b_in**2 - 1) // 3, 2, dtype=torch.float, device=device)  # [beta, alpha, gamma]
    
    x = g(f(x, b_out=b_out), b_out=b_in)

    y = g(f(x, b_out=b_out), b_out=b_in)

    assert (x - y).abs().max().item() < 1e-4 * y.abs().mean().item() 


from s2cnn.soft.so3_fft import so3_fft, so3_ifft
test_inverse(so3_fft, so3_ifft, 7, 7, torch.device("cpu"), True)
test_inverse(so3_fft, so3_ifft, 5, 4, torch.device("cpu"), True)
test_inverse(so3_fft, so3_ifft, 7, 4, torch.device("cpu"), True)

test_inverse2(so3_ifft, so3_fft, 7, 7, torch.device("cpu"))
test_inverse2(so3_ifft, so3_fft, 5, 4, torch.device("cpu"))
test_inverse2(so3_ifft, so3_fft, 4, 7, torch.device("cpu"))

if torch.cuda.is_available():
    test_inverse(so3_fft, so3_ifft, 7, 7, torch.device("cuda:0"), True)
    test_inverse(so3_fft, so3_ifft, 7, 5, torch.device("cuda:0"), True)
    test_inverse(so3_fft, so3_ifft, 4, 6, torch.device("cuda:0"), True)

    test_inverse2(so3_ifft, so3_fft, 7, 7, torch.device("cuda:0"))
    test_inverse2(so3_ifft, so3_fft, 5, 4, torch.device("cuda:0"))
    test_inverse2(so3_ifft, so3_fft, 4, 7, torch.device("cuda:0"))

from s2cnn.soft.so3_fft import so3_rfft, so3_rifft
test_inverse(so3_rfft, so3_rifft, 7, 7, torch.device("cpu"), False)
test_inverse(so3_rfft, so3_rifft, 5, 4, torch.device("cpu"), False)
test_inverse(so3_rfft, so3_rifft, 4, 6, torch.device("cpu"), False)

test_inverse2(so3_rifft, so3_rfft, 7, 7, torch.device("cpu"))
test_inverse2(so3_rifft, so3_rfft, 5, 4, torch.device("cpu"))
test_inverse2(so3_rifft, so3_rfft, 4, 7, torch.device("cpu"))

if torch.cuda.is_available():
    test_inverse(so3_rfft, so3_rifft, 7, 7, torch.device("cuda:0"), False)
    test_inverse(so3_rfft, so3_rifft, 5, 4, torch.device("cuda:0"), False)
    test_inverse(so3_rfft, so3_rifft, 4, 6, torch.device("cuda:0"), False)

    test_inverse2(so3_rifft, so3_rfft, 7, 7, torch.device("cuda:0"))
    test_inverse2(so3_rifft, so3_rfft, 5, 4, torch.device("cuda:0"))
    test_inverse2(so3_rifft, so3_rfft, 4, 7, torch.device("cuda:0"))



def compare_cpu_gpu(f, x):
    z1 = f(x.cpu())
    z2 = f(x.cuda()).cpu()

    q = (z1 - z2).abs().max().item() / z1.std().item()
    assert q < 1e-4

for b_in, b_out in [(2, 9), (6, 6), (9, 2), (10, 11), (11, 10)]:
    x = torch.rand(2 * b_in, 2 * b_in, 2 * b_in, 2)  # [..., beta, alpha, gamma, complex]
    compare_cpu_gpu(partial(so3_fft, b_out=b_out), x)

    x = torch.rand(2 * b_in, 2 * b_in, 2 * b_in)  # [..., beta, alpha, gamma]
    compare_cpu_gpu(partial(so3_rfft, b_out=b_out), x)

    x = torch.rand(b_in * (4 * b_in**2 - 1) // 3, 2)  # [l * m * n, ..., complex]
    compare_cpu_gpu(partial(so3_ifft, b_out=b_out), x)

    x = torch.rand(2 * b_in, 2 * b_in, 2 * b_in)  # [..., beta, alpha, gamma]
    x = so3_rfft(x)
    compare_cpu_gpu(partial(so3_rifft, b_out=b_out), x)

