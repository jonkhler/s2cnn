# Spherical CNNs
## Equivariant CNNs for the sphere and SO(3) implemented in PyTorch

![Equivariance](https://github.com/jonas-koehler/s2cnn/raw/master/examples/equivariance_plot/fig.jpeg)

## Overview
This library contains a PyTorch implementation of the rotation equivariant CNNs for spherical signals (e.g. omnidirectional images, signals on the globe) as presented in [[1]](https://arxiv.org/abs/1801.10130). Equivariant networks for the plane are available [here](https://github.com/tscohen/GrouPy).

## Dependencies

* __PyTorch__: http://pytorch.org/ (>= 0.4.0)
* __cupy__: https://github.com/cupy/cupy
* __lie_learn__: https://github.com/AMLab-Amsterdam/lie_learn
* __pynvrtc__: https://github.com/NVIDIA/pynvrtc

(commands to install all the dependencies on a new conda environment)
```bash
conda create --name cuda9 python=3.6 
conda activate cuda9

# s2cnn deps
conda install pytorch torchvision cuda90 -c pytorch  
conda install -c anaconda cupy  
pip install pynvrtc  

# lie_learn deps
conda install -c anaconda cython  
conda install -c anaconda requests  

# shrec17 example dep
conda install -c anaconda scipy  
conda install -c conda-forge rtree shapely  
conda install -c conda-forge pyembree  
pip install "trimesh[easy]"  
```

## Installation

To install, run

```bash
$ python setup.py install
```

## Usage
Please have a look at the [examples](examples).

Please cite [1] in your work when using this library in your experiments.

## Feedback
For questions and comments, feel free to contact us: taco.cohen (gmail), geiger.mario (gmail), jonas (argmin.xyz).


## License
MIT

## References

[1] Taco S. Cohen, Mario Geiger, Jonas Köhler, Max Welling,
[Spherical CNNs](https://arxiv.org/abs/1801.10130).
International Conference on Learning Representations (ICLR), 2018.

[2] Taco S. Cohen, Mario Geiger, Jonas Köhler, Max Welling,
[Convolutional Networks for Spherical Signals](https://arxiv.org/abs/1709.04893).
ICML Workshop on Principled Approaches to Deep Learning, 2017.

[3] Taco S. Cohen, Mario Geiger, Maurice Weiler,
[Intertwiners between Induced Representations (with applications to the theory of equivariant neural networks)](https://arxiv.org/abs/1803.10743),
ArXiv preprint 1803.10743, 2018.
