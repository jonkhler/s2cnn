import os
import numpy as np
import shutil
import requests
import zipfile
from train import Model
from dataset import Shrec17, CacheNPY, ToMesh, ProjectOnSphere
from subprocess import check_output
import torch
import torchvision


class KeepName:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, file_name):
        return file_name, self.transform(file_name)


def main():
    print(check_output(["nodejs", "--version"]))

    torch.backends.cudnn.benchmark = True

    # Increasing `repeat` will generate more cached files
    transform = CacheNPY(prefix="b64_", repeat=1, transform=torchvision.transforms.Compose(
        [
            ToMesh(random_rotations=True, random_translation=0.1),
            ProjectOnSphere(bandwidth=64)
        ]
    ))
    transform = KeepName(transform)

    resdir = "test_perturbed"
    dataset, perturbed = resdir.split("_")
    perturbed = (perturbed == "perturbed")

    test_set = Shrec17("data", dataset, perturbed=perturbed, download=True, transform=transform)

    loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    model = Model(55)
    model.cuda()

    model.load_state_dict(torch.load("state.pkl"))

    if os.path.isdir(resdir):
        shutil.rmtree(resdir)
    os.mkdir(resdir)

    predictions = []
    ids = []

    for batch_idx, data in enumerate(loader):
        model.eval()

        if dataset != "test":
            data = data[0]

        file_names, data = data

        data = data.cuda()
        data = torch.autograd.Variable(data, volatile=True)

        predictions.append(model(data).data.cpu().numpy())
        ids.extend([x.split("/")[-1].split(".")[0] for x in file_names])

        print("[{}/{}]      ".format(batch_idx, len(loader)))

    predictions = np.concatenate(predictions)

    ex = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
    softmax = ex / np.sum(ex, axis=1, keepdims=True)

    predictions_class = np.argmax(predictions, axis=1)

    for i in range(len(ids)):
        print("{}/{}    ".format(i, len(ids)), end="\r")
        idfile = os.path.join(resdir, ids[i])

        retrieved = [(softmax[j, predictions_class[j]], ids[j]) for j in range(len(ids)) if predictions_class[j] == predictions_class[i]]
        retrieved = sorted(retrieved, reverse=True)
        threshold = 0
        retrieved = [i for prob, i in retrieved if prob > threshold]

        with open(idfile, "w") as f:
            f.write("\n".join(retrieved))

    url = "https://shapenet.cs.stanford.edu/shrec17/code/evaluator.zip"
    file_path = "evaluator.zip"

    r = requests.get(url, stream=True)
    with open(file_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=16 * 1024 ** 2):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                f.flush()

    zip_ref = zipfile.ZipFile(file_path, 'r')
    zip_ref.extractall(".")
    zip_ref.close()

    print(check_output(["nodejs", "evaluate.js", "../"], cwd="evaluator"))


if __name__ == "__main__":
    main()
