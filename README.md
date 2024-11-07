# High-dimensional Safe Optimization with local optimistic exploration (HdSafeBO)

This is the code repository of the paper "Safe Bayesian Optimization for the Control of High-Dimensional Embodied Systems".

To replicate the experiment, run the following command:

## Synthetic Function

```
python optimization_gpfun.py --algo HdSafeBO --latent_opt 1
```
Replace "HdSafeBO" with other algorithms and switch "latent_opt" argument between 0 and 1 to optimize on the original/latent space.

## Musculoskeletal Model Control

```
python optimization_muscle.py --algo HdSafeBO --latent_opt 1
```


## Spinal Cord Neuromodulation experiment

To replicate the experiment, download data file in the following link and put the folder in ./task/scs

https://cloud.tsinghua.edu.cn/f/f04d18c7730743f68e13/?dl=1

Then run the following command:

```
python optimization_muscle.py --algo HdSafeBO --latent_opt 1
```