# pylint: disable=C,R,E1101,E1102
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from scipy.ndimage.interpolation import zoom

import torch
from s2cnn import S2Convolution, SO3Convolution, so3_rotation
from s2cnn import s2_near_identity_grid, so3_near_identity_grid


def s2_rotation(x, a, b, c):
    # TODO: check that this is indeed a correct s2 rotation
    x = so3_rotation(x.view(*x.size(), 1).expand(*x.size(), x.size(-1)), a, b, c)
    return x[..., 0]


def plot(x):
    assert x.size(0) == 1
    assert x.size(1) == 3
    x = x[0]
    if x.dim() == 4:
        x = x[..., 0]

    x = x.detach().cpu().numpy()
    x = x.transpose((1, 2, 0)).clip(0, 1)
    plt.imshow(x)


def main():
    # load image
    x = imread("earth128.jpg").astype(np.float32).transpose((2, 0, 1)) / 255
    b = 50
    x = zoom(x, (1, 2 * b / x.shape[1], 2 * b / x.shape[2]))
    x = torch.tensor(x, dtype=torch.float, device="cuda")
    x = x.view(1, 3, 2 * b, 2 * b)

    # equivariant transformation
    s2_grid = s2_near_identity_grid(max_beta=0.1, n_alpha=4, n_beta=1)
    s2_conv = S2Convolution(nfeature_in=3, nfeature_out=3, b_in=b, b_out=b, grid=s2_grid)
    s2_conv.cuda()

    so3_grid = so3_near_identity_grid(max_beta=0.1, n_alpha=4, n_beta=1)
    so3_conv = SO3Convolution(nfeature_in=3, nfeature_out=3, b_in=b, b_out=b, grid=so3_grid)
    so3_conv.cuda()

    def phi(x):
        x = s2_conv(x)
        x = torch.nn.functional.softplus(x)
        x = so3_conv(x)
        return x

    # test equivariance
    abc = (0, 0.5, 0)  # rotation angles

    y1 = phi(s2_rotation(x, *abc))
    y2 = so3_rotation(phi(x), *abc)
    print((y1 - y2).std().item(), y1.std().item())

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plot(x)

    plt.subplot(2, 3, 2)
    plot(s2_rotation(x, *abc))

    plt.subplot(2, 3, 3)
    plot(phi(x))

    plt.subplot(2, 3, 4)
    plot(phi(s2_rotation(x, *abc)))

    plt.subplot(2, 3, 5)
    plot(so3_rotation(phi(x), *abc))

    plt.tight_layout()
    plt.savefig("fig.jpeg")


if __name__ == "__main__":
    main()
