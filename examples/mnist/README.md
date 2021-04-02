# Spherical MNIST example

## Generate the spherical MNIST data set

- __NR__: non rotated
- __R__: randomly rotated

##### train: __NR__ - test: __NR__
```bash
python3 gendata.py --no_rotate_train --no_rotate_test
```

##### train: __R__ - test: __R__
```bash
python3 gendata.py
```

##### train: __NR__ - test: __R__
```bash
python3 gendata.py --no_rotate_train
```

This will generate a `s2_mnist.gz` in the same folder containing the compressed generated dataset.

To get more information about other params for the data generation (noise magnitude, number of images having the same random rotations etc.):
```bash
python3 gendata.py --help
```

## Run the models

(Apologies for the ugly global constants regarding hyperparams. I will add some nice argparse at a later point. - jonas)

### Simple 2D CNN

```bash
python3 run_classic.py
```

### Run S2CNNs

To run the original S2CNN architecture reported in the paper simply call
```bash
python3 run.py
```
or
```bash
python3 run.py --network=original
```

An improved model can be selected by calling
```bash
python3 run.py --network=deep
```
This architecture served as baseline for the the Icosahedral CNN [[1]](https://arxiv.org/pdf/1902.04615.pdf) (in the baseline run of [[1]](https://arxiv.org/pdf/1902.04615.pdf) slightly different hyperparameters like the bandwidth, learning rate decay and batch size were used).
It achieves an accuracy of ~99.2%.


## References

[1] Taco S. Cohen, Maurice Weiler, Berkay Kicanaoglu, Max Welling,
[Gauge Equivariant Convolutional Networks and the Icosahedral CNN](https://arxiv.org/pdf/1902.04615.pdf).
International Conference on Machine Learning (ICML), 2019.
