#pylint: disable=E1101,R,C
import torch
import torch.utils.data as data_utils
import gzip, pickle
import numpy as np

def load_data(path, batch_size):
    '''
    Loads the data

        path:           path to the spherical MNIST .gz
        batch_size:     size of a mini batch
    '''
    with gzip.open(path, 'rb') as f:
        dataset = pickle.load(f)

    train_data = torch.from_numpy(
        dataset["train"]["images"][:,None,:,:].astype(np.float32))
    train_labels = torch.from_numpy(
        dataset["train"]["labels"].astype(np.int64))

    mean = train_data.mean()
    stdv = train_data.std()
    train_data = (train_data - mean) / stdv

    train_dataset = data_utils.TensorDataset(train_data, train_labels)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_data = torch.from_numpy(
        dataset["test"]["images"][:,None,:,:].astype(np.float32))
    test_labels = torch.from_numpy(
        dataset["test"]["labels"].astype(np.int64))

    test_data = (test_data - mean) / stdv

    test_dataset = data_utils.TensorDataset(test_data, test_labels)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, train_dataset, test_dataset

