'''
Aim: see that L_R Phi(x) = Phi(L_R x)

Where Phi is a composition of a S^2 convolution and a SO(3) convolution

For simplicity, R is a rotation around the Z axis.
'''

#pylint: disable=C,R,E1101,W0621
import torch

from s2cnn import s2_equatorial_grid, S2Convolution
from s2cnn import so3_equatorial_grid, SO3Convolution

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the two convolutions
s2_grid = s2_equatorial_grid(max_beta=0, n_alpha=64, n_beta=1)
s2_conv = S2Convolution(nfeature_in=12, nfeature_out=15, b_in=64, b_out=32, grid=s2_grid)
s2_conv.to(device)

so3_grid = so3_equatorial_grid(max_beta=0, max_gamma=0, n_alpha=64, n_beta=1, n_gamma=1)
so3_conv = SO3Convolution(nfeature_in=15, nfeature_out=21, b_in=32, b_out=24, grid=so3_grid)
so3_conv.to(device)

def phi(x):
    x = s2_conv(x)
    x = torch.nn.functional.relu(x)
    x = so3_conv(x)
    return x

def rot(x, angle):
    # rotate the signal around the Z axis
    n = round(x.size(3) * angle / 360)
    return torch.cat([x[:, :, :, n:], x[:, :, :, :n]], dim=3)

# Create random input
x = torch.randn(1, 12, 128, 128).to(device) # [batch, feature, beta, alpha]

y = phi(x)
y1 = rot(phi(x), angle=45)
y2 = phi(rot(x, angle=45))

relative_error = torch.std(y1.data - y2.data) / torch.std(y.data)

print('relative error = {}'.format(relative_error))
