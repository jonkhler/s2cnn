#pylint: disable=C,R,E1101,W0621
import numpy as np
import torch
from torch.autograd import Variable

from s2cnn.nn.soft.so3_rotation import so3_rotation
from s2cnn.nn.soft.so3_conv import SO3Convolution
from s2cnn.ops.so3_localft import equatorial_grid
bandwidth = 30

layers = 3
number_of_features = 10

# Pick random Euler angles
alpha = np.random.rand() * 2 * np.pi
beta = np.arccos(2 * np.random.rand() - 1)
gamma = np.random.rand() * 2 * np.pi


x = Variable(torch.randn(1, number_of_features, 2 * bandwidth, 2 * bandwidth, 2 * bandwidth), volatile=True).cuda()

grid = equatorial_grid(max_beta=0, max_gamma=0, n_alpha=2 * bandwidth, n_beta=1, n_gamma=1)
convs = [SO3Convolution(number_of_features, number_of_features, bandwidth, bandwidth, grid) for _ in range(layers)]

for conv in convs:
    conv.cuda()

def foo(x):
    for conv in convs:
        x = conv(x)
    return x

# \Phi(x)
y = foo(x)

# L_R \Phi(x)
y1 = so3_rotation(y, alpha, beta, gamma)

# \Phi(L_R x)
y2 = foo(so3_rotation(x, alpha, beta, gamma))


y = y.data.cpu().numpy()
y1 = y1.data.cpu().numpy()
y2 = y2.data.cpu().numpy()

relative_error = np.std(y1 - y2) / np.std(y)

print('relative error = {}'.format(relative_error))
