[gan]: data/figures/gan_training.png
[gangan]: data/figures/gangan_sample.png

# GAN You Do the GAN GAN? [arXiv](https://arxiv.org/abs/1904.00724)
This repository contains the [Paper PDF](suarez_gangan.pdf), full source code, and sample models for the short paper "GAN You Do the GAN GAN?" in which we train a GAN over 100 snapshots from each of 35 trained GANs.

## Quickstart
We assume a standard Anaconda setup with GPU Pytorch installed. This is a minimal project with standard libraries -- if you are missing anything, snag it with pip. This project was conducted on a single desktop with a 6 core i7 and a GTX 1080 Ti. If training from scratch, expect to generate ~20GB of data.

# Recommended Setup:
```
#Train all the GANS (overnight this)
python train.py gan

#Train the GAN-GAN over GANs (lunch break this)
python train.py gangan

#Reproduce the paper figures (using prebuilt models)
python figures.py
```

Note: GANs are notoriously difficult to train. While the GAN training code is stable. the GAN-GAN code is not. Stability and convergence speed vary dramatically with random seed. While most runs yield something reasonable, you should not expect to reproduce our results exactly without a bit of patience and tweaking.

# TL;DR

GAN You Do the GAN-GAN? Yes you GAN! Generative models can model other generative models in an interpretable and meaningful latent space.

# Summary

This is an April 1 paper, but the results are real. This is what the samples from a small MNIST GAN look like over the course of training:

<video width="320" height="240" controls>
  <source src="data/gan/0/demo.mp4" type="video/mp4">
</video>

(Image in case the video does not load)![][gan]

And this is what the images (horizontal) sampled from the GANs sampled from the 1D latent space (vertical) of a GAN-GAN look like. Not only are they somewhat better than the images sampled from the GAN, but the latent space of the GAN-GAN appears to have a structure that roughly corresponds to the different training stages of a GAN.
![][gangan]
