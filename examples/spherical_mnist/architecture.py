#pylint: disable=E1101,R,C
import torch.nn as nn
from s2cnn.nn.soft.so3_conv import SO3Convolution
from s2cnn.nn.soft.s2_conv import S2Convolution
from s2cnn.nn.soft.so3_integrate import so3_integrate
from sphere_cnn.ops.so3_localft import equatorial_grid as so3_equatorial_grid
from sphere_cnn.ops.s2_localft import equatorial_grid as s2_equatorial_grid
import torch.nn.functional as F


class Mnist_Classifier(nn.Module):
    '''Very simple S(2) CNN for classifying spherical MNIST signals.'''

    def __init__(self):
        super(Mnist_Classifier, self).__init__()

        # number of filters on each layer
        k_input = 1
        k_l1 = 100
        k_l2 = 200
        k_output = 10

        # bandwidth on each layer
        b_in = 30
        b_l1 = 10
        b_out = 5

        # grid for the s2 convolution
        grid_s2 = s2_equatorial_grid(
            max_beta=0, n_alpha=2 * b_in, n_beta=1)

        # grid for the so3 convolution
        grid_so3 = so3_equatorial_grid(
                max_beta=0, max_gamma=0, n_alpha=2 * b_l1, n_beta=1, n_gamma=1)

        # first layer is a S(2) convolution
        self.s2_conv = S2Convolution(
            nfeature_in=k_input,
            nfeature_out=k_l1,
            b_in=b_in,
            b_out=b_l1,
            grid=grid_s2)

        # second layer is a SO(3) convolution
        self.so3_conv = SO3Convolution(
            nfeature_in=k_l1,
            nfeature_out=k_l2,
            b_in=b_l1,
            b_out=b_out,
            grid=grid_so3)

        # output layer is a linear regression on the filters
        self.out_layer = nn.Linear(k_l2, k_output)

    def forward(self, x):
        ''' Return logits for the given input signal '''

        # first layer
        x = self.s2_conv(x)
        x = F.relu(x)

        # second layer
        x = self.so3_conv(x)
        x = F.relu(x)

        # integrate out gamma dimension
        # projecting the SO(3) signal
        # back onto S(2)
        x = so3_integrate(x)

        # linear regression for the logits
        x = self.out_layer(x)

        return x
