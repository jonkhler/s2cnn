# pylint: disable=E1101,R,C
import torch
import numpy as np
import joblib


class IndexBatcher:

    def __init__(self, indices, n_batch, cuda=None):
        self.indices = indices.astype(np.int64)
        self.n_batch = n_batch
        self.pos = 0
        self.cuda = cuda
        self.internal_indices = np.arange(len(indices)).astype(np.int64)
        np.random.shuffle(self.internal_indices)

    def __iter__(self):
        return self

    def reset(self):
        self.pos = 0
        np.random.shuffle(self.internal_indices)

    def __next__(self):
        start = self.pos
        end = np.minimum(self.pos + self.n_batch, len(self.indices))
        self.pos += self.n_batch
        if self.pos >= len(self.indices):
            self.reset()
            raise StopIteration
        tensor = torch.LongTensor(
            self.indices[self.internal_indices[start:end]])
        if self.cuda is not None:
            tensor.cuda(self.cuda)
        return tensor

    def num_iterations(self):
        return len(self.indices) // self.n_batch

    next = __next__


def to_one_hot(x, n):
    x_ = torch.unsqueeze(x, 2)
    dims = (*x.size(), n)
    one_hot = torch.FloatTensor(*dims).zero_()
    one_hot.scatter_(2, x_, 1)
    return one_hot


def load_data(path, test_strat_id=None, cuda=None):
    '''
    Loads the data

        path:           path to the molecule .gz
        batch_size:     size of a mini batch
        test_strat_id:  id of strat being used as test set
    '''
    data = joblib.load(path)

    # map charges to type indices
    # TODO refactor to individual function
    # TODO make less reliant on individual data dict structure
    type_remap = -np.ones(int(data["features"]["atom_types"].max())+1)
    unique_types = np.unique(data["features"]["atom_types"]).astype(int)
    type_remap[unique_types] = np.arange(len(unique_types))
    data["features"]["atom_types"] = type_remap[
        data["features"]["atom_types"].astype(int)]

    # wrap features as torch tensors
    data["features"]["geometry"] = torch.FloatTensor(
        data["features"]["geometry"].astype(np.float32))
    data["features"]["atom_types"] = torch.LongTensor(
        data["features"]["atom_types"].astype(np.int64))
    data["targets"] = torch.from_numpy(data["targets"])

    if cuda is not None:
        data["features"]["geometry"].cuda(cuda)
        data["features"]["atom_types"].cuda(cuda)
        data["targets"].cuda(cuda)

    train = np.ndarray((0))
    test = np.ndarray((0))

    # split in train and test set according to used strat
    # TODO this should be solved in a less ugly/ad-hoc fashion!
    if not test_strat_id:
        test_strat_id = np.random.randint(len(data["strats"]))
    for i in range(len(data["strats"])):
        if i != test_strat_id:
            train = np.concatenate((train, data["strats"][i]))
        else:
            test = np.concatenate((test, data["strats"][i]))

    return data, train, test


def exp_lr_scheduler(optimizer, epoch, init_lr=5e-3, lr_decay_epoch=40):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))
    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return optimizer


def count_params(model):
    return sum([np.prod(p.size())
                for p in model.parameters()
                if p.requires_grad])
