# SHREC17

## Dependencies

To load the dataset SHREC17 we use the libaray `trimesh` and make it work efficiently you will aslo need to install `pyembree`.


## Training
```
python train.py --model_path model.py --log_dir my_run --dataset train --batch_size 32 --augmentation 4
```

## Validation
```
python test.py --log_dir my_run --dataset val --batch_size 32 --augmentation 4
cat my_run/summary.csv | grep micro
```
