# s2cnn - SO(3) equivariant CNNs for PyTorch

## Example: MNIST on a sphere

### Description of the data

![Projected 4 in phi, theta coordinates"](figures/soft4.jpeg?raw=true).
![Projected 4 in cartesian 3d coordinates](figures/sphere4.png?raw=true).
![Projected 9 in phi, theta coordinates](figures/soft9.jpeg?raw=true).
![Projected 9 in cartesian 3d coordinates](figures/sphere9.png?raw=true).

This is the classic MNIST data set projected on the sphere, using a stereographic projection from the north pole. The sphere is parameterized by its angles phi and theta. In the current setup we use the SOFT[1] grid with a bandwidth b for representing the sphere. To reduce artifacts we use bilinear sampling to map the projection on the grid.

## Dependencies

* __Tensorflow__

### Getting the data

To download and preprocess the data run
```bash
$ python3 data/generate_s2mnist.py
```
it will generate a `s2_mnist.gz` file, that contains all the projected data.

Optional parameters for the data generation are:

* __bandwidth__: Controls the bandwidth b of the signal. Defaults to b=30.
* __noise__: Controls the amount of noise, by which the digits are rotated. Setting the noise to 0 will create spheres, where the digit is always centered at the northpole. Setting it to 1 will create spheres where uniformily sampled rotations are applied on. Defaults to 1.
* __chunk_size__: Controls the number of digits that are processed in a chunk/batch, where a random rotation is uniformily applied on all digits of a chunk. A big chunk size speeds up the preprocessing step at the cost of coarser sampling of random rotations. A small chunk size will give a dense sampling, but takes a longer time. Defaults to 50.
* __mnist_data_folder__: Sets the folder where the raw MNIST digits are stored. Defaults to `MNIST_data`.
* __output_file__: Sets the output file for generated data. Defaults to `s2_mnist.gz`.

### A simple CNN architecture to solve the digit classification

For our example we use a very simple architecture (`architecture.py`):

1. Use a S(2)-convolution for the raw S(2) input signal, which will yield a K_0-dimensional filter bank of SO(3) signals.
2. Use a SO(3)-convolution on these filter banks, which yields a K_1-dimensional filter bank of SO(3) signals.
3. Use a 10-dimensional linear layer on these filters to produce our logits.
4. Use a softmax for the actual classification.

```python

# architecture.py [...] 

# number of filters on each layer
k_input = 1
k_l1 = 100
k_l2 = 200
k_output = 10

# bandwidth on each layer
b_in = 30
b_l1 = 10
b_out = 5

# size of convolution kernel for each layer
ks_in = 6
ks_l1 = 2

# first layer is a S(2) convolution
self.s2_conv = S2Convolution(
    in_channels=k_input,
    out_channels=k_l1,
    in_b=b_in,
    out_b=b_l1,
    size=ks_in)

# second layer is a SO(3) convolution
self.so3_conv = SO3Convolution(
    in_channels=k_l1,
    out_channels=k_l2,
    in_b=b_l1,
    out_b=b_out,
    size=ks_l1)

# output layer is a linear regression on the filters
self.out_layer = nn.Linear(k_l2, k_output)
```

### Running the experiment

After getting the data and setting up the architecture we now run the experiment (`run.py`). For the given setup of hyper params the final results should achieve an accuracy of ~99%.

```sh
$ python3 run.py
```