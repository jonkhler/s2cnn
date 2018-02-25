# pylint: disable=E1101,R,C
import torch
import torch.nn.functional as F
import torchvision

from s2cnn.ops.s2_localft import equatorial_grid as s2_equatorial_grid
from s2cnn.nn.soft.s2_conv import S2Convolution
from s2cnn.ops.so3_localft import equatorial_grid as so3_equatorial_grid
from s2cnn.nn.soft.so3_conv import SO3Convolution

import time

from dataset import Shrec17, CacheNPY, ToMesh, ProjectOnSphere


class Model(torch.nn.Module):
    def __init__(self, nclasses):
        super().__init__()

        self.features = [6,  20, 60, 100, nclasses]
        self.bandwidths = [64, 20, 10, 7]

        assert len(self.bandwidths) == len(self.features) - 1

        sequence = []

        # S2 layer
        grid = s2_equatorial_grid(max_beta=0, n_alpha=2 * self.bandwidths[0], n_beta=1)
        sequence.append(S2Convolution(self.features[0], self.features[1], self.bandwidths[0], self.bandwidths[1], grid))

        # SO3 layers
        for l in range(1, len(self.features) - 2):
            nfeature_in = self.features[l]
            nfeature_out = self.features[l + 1]
            b_in = self.bandwidths[l]
            b_out = self.bandwidths[l + 1]

            sequence.append(torch.nn.BatchNorm3d(nfeature_in, affine=True))
            sequence.append(torch.nn.ReLU())
            grid = so3_equatorial_grid(max_beta=0, max_gamma=0, n_alpha=2 * b_in, n_beta=1, n_gamma=1)
            sequence.append(SO3Convolution(nfeature_in, nfeature_out, b_in, b_out, grid))

        sequence.append(torch.nn.BatchNorm3d(self.features[-2], affine=True))
        sequence.append(torch.nn.ReLU())

        self.sequential = torch.nn.Sequential(*sequence)

        # Output layer
        output_features = self.features[-2]
        self.bn_out2 = torch.nn.BatchNorm1d(output_features, affine=False)
        self.out_layer = torch.nn.Linear(output_features, self.features[-1])

    def forward(self, x):  # pylint: disable=W0221
        x = self.sequential(x)  # [batch, feature, beta, alpha, gamma]
        x = x.view(x.size(0), x.size(1), -1).max(-1)[0]  # [batch, feature]

        x = self.bn_out2(x.contiguous())
        x = self.out_layer(x)
        return F.log_softmax(x, dim=1)


def main():
    torch.backends.cudnn.benchmark = True

    # Increasing `repeat` will generate more cached files
    transform = CacheNPY(prefix="b64_", repeat=2, transform=torchvision.transforms.Compose(
        [
            ToMesh(random_rotations=True, random_translation=0.1),
            ProjectOnSphere(bandwidth=64)
        ]
    ))

    def target_transform(x):
        classes = ['02691156', '02747177', '02773838', '02801938', '02808440', '02818832', '02828884', '02843684', '02871439', '02876657',
                   '02880940', '02924116', '02933112', '02942699', '02946921', '02954340', '02958343', '02992529', '03001627', '03046257',
                   '03085013', '03207941', '03211117', '03261776', '03325088', '03337140', '03467517', '03513137', '03593526', '03624134',
                   '03636649', '03642806', '03691459', '03710193', '03759954', '03761084', '03790512', '03797390', '03928116', '03938244',
                   '03948459', '03991062', '04004475', '04074963', '04090263', '04099429', '04225987', '04256520', '04330267', '04379243',
                   '04401088', '04460130', '04468005', '04530566', '04554684']
        return classes.index(x[0])

    train_set = Shrec17("data", "train", perturbed=True, download=True, transform=transform, target_transform=target_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    model = Model(55)
    model.cuda()

    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    print("{} paramerters in the last layer".format(sum(x.numel() for x in model.out_layer.parameters())))

    optimizer = torch.optim.SGD(model.parameters(), lr=0, momentum=0.9)

    def train_step(data, target):
        model.train()
        data, target = data.cuda(), target.cuda()
        data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)

        prediction = model(data)
        loss = F.nll_loss(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.data[0]

    def get_learning_rate(epoch):
        limits = [30, 50, 100]
        lrs = [0.5, 0.05, 0.005, 0.0005]
        assert len(lrs) == len(limits) + 1
        for lim, lr in zip(limits, lrs):
            if epoch < lim:
                return lr
        return lrs[-1]

    for epoch in range(300):

        lr = get_learning_rate(epoch)
        for p in optimizer.param_groups:
            p['lr'] = lr

        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            time_start = time.perf_counter()
            loss = train_step(data, target)

            total_loss += loss

            print("[{}:{}/{}] LOSS={:.2} <LOSS>={:.2} time={:.2}".format(
                epoch, batch_idx, len(train_loader), loss, total_loss / (batch_idx + 1), time.perf_counter() - time_start))

        torch.save(model.state_dict(), "state.pkl")


if __name__ == "__main__":
    main()
