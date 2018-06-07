# pylint: disable=E1101,R,C
import torch
import torch.nn as nn
import torch.nn.functional as F
from s2cnn.soft.so3_conv import SO3Convolution
from s2cnn.soft.s2_conv import S2Convolution
from s2cnn.soft.so3_integrate import so3_integrate
from s2cnn.so3_grid import so3_near_identity_grid
from s2cnn.s2_grid import s2_near_identity_grid


nonlinearity = F.relu
AFFINE = True


class S2Block(nn.Module):
    """ simple s2 convolution block """

    def __init__(self, b_in, b_out, f_in, f_out):
        """ b_in/b_out: bandwidth of input/output signals
            f_in/f_out: filters in input/output signals """

        super(S2Block, self).__init__()

        self.grid_s2 = s2_near_identity_grid(
            n_alpha=2*b_in, n_beta=2)

        self.cnn = S2Convolution(
            nfeature_in=f_in,
            nfeature_out=f_out,
            b_in=b_in,
            b_out=b_out,
            grid=self.grid_s2)

        self.bn = nn.BatchNorm3d(f_out, affine=AFFINE)

    def forward(self, x):
        x = self.cnn(x)
        x = self.bn(x)
        x = nonlinearity(x)
        return x


class So3Block(nn.Module):
    """ simple so3 convolution block """

    def __init__(self, b_in, b_out, f_in, f_out):
        """ b_in/b_out: bandwidth of input/output signals
            f_in/f_out: filters in input/output signals """

        super(So3Block, self).__init__()

        self.grid_so3 = so3_near_identity_grid(
            n_alpha=2*b_in, n_beta=2, n_gamma=2)

        self.cnn = SO3Convolution(
            nfeature_in=f_in,
            nfeature_out=f_out,
            b_in=b_in,
            b_out=b_out,
            grid=self.grid_so3)

        self.bn = nn.BatchNorm3d(f_out, affine=AFFINE)

    def forward(self, x):
        x = self.cnn(x)
        x = self.bn(x)
        x = nonlinearity(x)
        return x


class DeepSet(nn.Module):
    """ deep set block """

    def __init__(self, f, h1, h_latent, h2, n_objs):
        """ f:         input filters
            h1, h2:    hidden units for encoder/decoder mlps
            h_latent:  dimensions
            n_objs:    of objects to aggregate in latent space """

        super(DeepSet, self).__init__()
        self.f = f
        self.h1 = h1
        self.h3 = h2
        self.n_objs = n_objs

        # encoder
        self.emb_h = nn.Linear(f, h1)
        self.emb_rep = nn.Linear(h1, h_latent)

        # decoder
        self.proj_h = nn.Linear(h_latent, h2)
        self.proj = nn.Linear(h2, 1)

        self.bn1 = nn.BatchNorm1d(h1, affine=AFFINE)
        self.bn2 = nn.BatchNorm1d(h_latent, affine=AFFINE)
        self.bn3 = nn.BatchNorm1d(h2, affine=AFFINE)

    def forward(self, x, mask):

        # encode atoms
        x = self.emb_h(x)
        x = self.bn1(x)
        x = nonlinearity(x)
        x = self.emb_rep(x)
        x = self.bn2(x)
        x = nonlinearity(x)

        # reshape (batch * atoms, features) -> (batch, atoms, features)
        n, h_latent = x.size()
        x = x.view(n // self.n_objs, self.n_objs, h_latent)

        # sum over latent atoms, filter out NULL atoms with mask
        x = torch.sum(x * mask, dim=1)

        # decode to final energy
        x = self.proj_h(x)
        x = self.bn3(x)
        x = nonlinearity(x)
        x = self.proj(x)

        return x


class S2CNNRegressor(nn.Module):
    """ approximate energy using spherical representations """

    def __init__(self):
        super(S2CNNRegressor, self).__init__()

        # number of atoms in a molecule
        n_objs = 23

        self.blocks = [
                S2Block(b_in=10, f_in=5, b_out=8, f_out=8),
                So3Block(b_in=8, b_out=6, f_in=8, f_out=16),
                So3Block(b_in=6, b_out=4, f_in=16, f_out=32),
                So3Block(b_in=4, b_out=2, f_in=32, f_out=64),
        ]

        # TODO: replace with nn.Sequential or similar
        for i, block in enumerate(self.blocks):
            setattr(self, "block{0}".format(i), block)

        self.ds = DeepSet(64, 256, 64, 512, n_objs)

    def forward(self, x, atom_types):

        n_batch, n_atoms, n_features, bandwidth, _ = x.size()

        # compute mask of atoms which are present
        # this prevents from the need to learn NULL atoms
        mask = (atom_types > 0).view(n_batch, n_atoms, 1).float()

        # push atoms to batch dimension
        x = x.view(n_batch * n_atoms, n_features, bandwidth, bandwidth)

        # propagate through convolutions
        for block in self.blocks:
            x = block(x)

        # integrate over SO(3)
        x = so3_integrate(x)

        # combine atom representations to final energy
        y = self.ds(x, mask)

        return y
