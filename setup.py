#pylint: disable=C
import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

this_dir = os.path.dirname(__file__)

cffi_modules = []

try:
    import torch
    if torch.cuda.is_available():
        print("try to build CUDA depdendencies")
        cffi_modules.append(os.path.join(this_dir, "build.py:ffi_plan_cufft"))
    else:
        print("CUDA is not available on your system.")
except ImportError:
    print("PyTorch is not available on your system.")

setup(
    name='s2cnn',
    version = "1.0.0",
    author = "Mario Geiger, Taco Cohen, Jonas Koehler",
    description = ("SO(3) equivariant CNNs for PyTorch."),
    license = "MIT",
    keywords = "so3 equivariant cnn pytorch",
    url = "https://github.com/AMLab-Amsterdam/s2cnn",
    long_description=read('README'),
    install_requires=["cffi>=1.0.0"],
    setup_requires=["cffi>=1.0.0"],
    packages=find_packages(exclude=["build"]),
    cffi_modules=cffi_modules,
)
