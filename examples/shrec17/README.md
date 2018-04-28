# SHREC17

## Dependencies

To load the dataset SHREC17 we use the libaray `trimesh` and make it work efficiently you will aslo need to install `pyembree`.
(See on [the trimesh github](https://github.com/mikedh/trimesh#basic-installation) how to install them)

## Training
```
python train.py --model_path model.py --log_dir my_run --dataset train --batch_size 32 --learning_rate 0.5 --augmentation 5
```

## Validation
```
python test.py --log_dir my_run --dataset val --batch_size 32 --augmentation 5
cat my_run/summary.csv | grep micro
```

## Original Code
The files here are not the original code used for the article. They have been recoded from scratch.
These files a simpler and clearer to read.
The model has also been simplified.
These simplifications gives *very similar results*.

### Model
The results reported in the article where produced by the model `model_original.py`.
The file `model.py` is a simplification of the original model we used, here is the list of the differences:
- `model.py` has one less layer than `model_original.py` (2 hidden layers vs 3 hidden layers)
- `model.py` has 1M less parameters than `model_original.py` (1.4M vs 400k)
- `model.py` takes as input a bandwidth of 64 instead of 128 (smaller input images on the sphere)
- To get rid of the spacial dimensions `model.py` computes the integral instead of taking the maximum

### Training
Originaly (for the article) we used Adam with a complicated learning rate schedule.
Here `train.py` used SGD with momentum and the learning rate is divided by 10 every 100 epoch.

### Data Augmentation
Originaly the data augmentation (rotation and translation of the 3D model before the raytracing) where fixed to 7 augmentations,
we did one translation in each direction in 3D: `(0,0,0), (+1,0,0), (-1,0,0), (0,+1,0), (0,-1,0), (0,0,+1), (0,0,-1)`.
Now to make things more flexible and easy, the code in `dataset.py` perform random translations such that the number of augmentation can be set arbitratily.
