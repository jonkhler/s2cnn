# Molecule Example

## Get dataset

Download original QM7 Matlab file

```bash
wget http://quantum-machine.org/data/qm7.mat
```

Run preprocessing script
```bash
python3 datagen.py
```

## Run experiments

```bash
./run_all_experiments.sh
```
### Remark about results

This version is not the exact architecture as explained in the paper but a much simpler one which instead uses only a fraction of parameters, runs faster, is more stable and produces much better results. When run correctly, it should produce a RMSE in the ~5 regime.

If you have of any questions about this experiment feel free to contact jonas (at) argmin (dot) xyz.
