#pylint: disable=C
import os
from setuptools import setup, find_packages

try:
    import torch
    if not torch.cuda.is_available():
        print("torch.cuda.is_available() returned False")
        exit()
except ImportError:
    print("Cannot import torch")
    exit()

setup(
    name='s2cnn',
    version="1.0.0",
    author="Mario Geiger, Taco Cohen, Jonas Koehler",
    description=("SO(3) equivariant CNNs for PyTorch."),
    license="MIT",
    keywords="so3 equivariant cnn pytorch",
    url="https://github.com/AMLab-Amsterdam/s2cnn",
    long_description=open(os.path.join(os.path.dirname(__file__), "README.md"), encoding='utf-8').read(),
    install_requires=["cffi>=1.0.0"],
    setup_requires=["cffi>=1.0.0"],
    packages=find_packages(exclude=["build"]),
    cffi_modules=[os.path.join(os.path.dirname(__file__), "build.py:ffi_plan_cufft")],
)
