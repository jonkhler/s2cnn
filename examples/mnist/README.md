# Spherical MNIST example

## Generate the sphericla MNIST data set

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

### Run the S2CNN

```bash
python3 run.py
```
