# pylint: disable=C,R,E1101,E1102
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

import torch
from s2cnn import S2Convolution, SO3Convolution, so3_rotation
from s2cnn import s2_near_identity_grid, so3_near_identity_grid


def s2_rotation(x, a, b, c):
    x = so3_rotation(x.view(*x.size(), 1).expand(*x.size(), x.size(-1)), a, b, c)
    return x[..., 0]


def plot(x, text, normalize=False):
    assert x.size(0) == 1
    assert x.size(1) in [1, 3]
    x = x[0]
    if x.dim() == 4:
        x = x[..., 0]

    nch = x.size(0)
    is_rgb = (nch == 3)

    if normalize:
        x = x - x.view(nch, -1).mean(-1).view(nch, 1, 1)
        x = 0.4 * x / x.view(nch, -1).std(-1).view(nch, 1, 1)

    x = x.detach().cpu().numpy()
    x = x.transpose((1, 2, 0)).clip(0, 1)

    print(x.shape)
    if is_rgb:
        plt.imshow(x)
    else:
        plt.imshow(x[:, :, 0], cmap='gray')
    plt.axis("off")

    plt.text(0.5, 0.5, text,
             horizontalalignment='center',
             verticalalignment='center',
             transform=plt.gca().transAxes,
             color='white', fontsize=20)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load image
    x = imread("earth128.jpg").astype(np.float32).transpose((2, 0, 1)) / 255
    b = 64
    x = torch.tensor(x, dtype=torch.float, device=device)
    x = x.view(1, 3, 2 * b, 2 * b)

    # equivariant transformation
    s2_grid = s2_near_identity_grid(max_beta=0.2, n_alpha=12, n_beta=1)
    s2_conv = S2Convolution(3, 50, b_in=b, b_out=b, grid=s2_grid)
    s2_conv.to(device)

    so3_grid = so3_near_identity_grid(max_beta=0.2, n_alpha=12, n_beta=1)
    so3_conv = SO3Convolution(50, 1, b_in=b, b_out=b, grid=so3_grid)
    so3_conv.to(device)

    def phi(x):
        x = s2_conv(x)
        x = torch.nn.functional.softplus(x)
        x = so3_conv(x)
        return x

    # test equivariance
    abc = (0.5, 1, 0)  # rotation angles

    y1 = phi(s2_rotation(x, *abc))
    y2 = so3_rotation(phi(x), *abc)
    print((y1 - y2).std().item(), y1.std().item())

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plot(x, "x : signal on the sphere")

    plt.subplot(2, 3, 2)
    plot(phi(x), "phi(x) : convolutions", True)

    plt.subplot(2, 3, 3)
    plot(so3_rotation(phi(x), *abc), "R(phi(x))", True)

    plt.subplot(2, 3, 4)
    plot(s2_rotation(x, *abc), "R(x) : rotation using fft")

    plt.subplot(2, 3, 5)
    plot(phi(s2_rotation(x, *abc)), "phi(R(x))", True)

    plt.tight_layout()
    plt.savefig("fig.jpeg")


if __name__ == "__main__":
    main()
