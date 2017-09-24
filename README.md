# s2cnn - SO(3) equivariant CNNs for PyTorch

## Overview
This library contains a PyTorch implementation of the SO(3) equivariant CNNs for spherical signals (e.g. omnidirectional cameras, signals on the globe) as presented in [[1]](https://arxiv.org/abs/1709.04893).

## Dependencies

* __PyTorch__: http://pytorch.org/
* __cupy__: https://github.com/cupy/cupy
* __lie_learn__: https://github.com/AMLab-Amsterdam/lie_learn
* __pynvrtc__: https://github.com/NVIDIA/pynvrtc

## Installation

To install, run

```bash
$ python setup.py install
```

## Structure
* [__nn__](s2cnn/nn): PyTorch nn.Modules for the S(2) and SO(3) CNN layers
* [__ops__](s2cnn/ops): Low-level operations used for computing the FFT
* [__examples__](examples): Example code for using the library within a PyTorch project
    - [__spherical_mnist__](s2cnn/examples/spherical_mnist): Data and example code for applying s2cnn on MNIST digits projected on a sphere

## Usage
Please have a look into the [examples](s2cnn/examples).

Please cite [1] in your work when using this library in your experiments.

## Feedback
For questions and comments, feel free to contact [Taco Cohen](http://ta.co.nl).

## License
MIT

## References

```
[1] Taco Cohen, Mario Geiger, Jonas Köhler, Max Welling (2017). 
Convolutional Networks for Spherical Signals. 
In ICML Workshop on Principled Approaches to Deep Learning.
```
