#pylint: disable=C
import os
from setuptools import setup, find_packages

this_dir = os.path.dirname(__file__)

cffi_modules = []

try:
    import torch
    if torch.cuda.is_available():
        cffi_modules.append(os.path.join(this_dir, "build.py:ffi_plan_cufft"))
except ImportError:
    pass

setup(
    name='s2cnn',
    install_requires=["cffi>=1.0.0"],
    setup_requires=["cffi>=1.0.0"],
    packages=find_packages(exclude=["build"]),
    cffi_modules=cffi_modules,
)
