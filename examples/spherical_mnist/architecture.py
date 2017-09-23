#pylint: disable=E1101,R,C
import torch.nn as nn
from s2cnn.nn.soft.so3_conv import SO3Convolution
from s2cnn.nn.soft.s2_conv import S2Convolution
from s2cnn.nn.soft.so3_integrate import so3_integrate
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

        # size of convolution kernel for each layer
        ks_in = 6
        ks_l1 = 2

        # first layer is a S(2) convolution
        self.s2_conv = S2Convolution(
            in_channels=k_input,
            out_channels=k_l1,
            in_b=b_in,
            out_b=b_l1,
            size=ks_in)

        # second layer is a SO(3) convolution
        self.so3_conv = SO3Convolution(
            in_channels=k_l1,
            out_channels=k_l2,
            in_b=b_l1,
            out_b=b_out,
            size=ks_l1)

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
