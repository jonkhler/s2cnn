#pylint: disable=C
import os
from setuptools import setup, find_packages

setup(
    name='s2cnn',
    version="1.0.0",
    author="Mario Geiger, Taco Cohen, Jonas Koehler",
    description=("SO(3) equivariant CNNs for PyTorch."),
    license="MIT",
    keywords="so3 equivariant cnn pytorch",
    url="https://github.com/AMLab-Amsterdam/s2cnn",
    long_description=open(os.path.join(os.path.dirname(__file__), "README.md"), encoding='utf-8').read(),
    packages=find_packages(exclude=["build"]),
)
