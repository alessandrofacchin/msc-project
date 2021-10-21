# Latent Variables Related to Behaviour in Neural Activity

[![Python Tests](https://github.com/alessandrofacchin/msc-project/actions/workflows/python-tests.yml/badge.svg?branch=main)](https://github.com/alessandrofacchin/msc-project/actions/workflows/python-tests.yml) [![codecov](https://codecov.io/gh/alessandrofacchin/msc-project/branch/main/graph/badge.svg?token=nqcEwTGBbE)](https://codecov.io/gh/alessandrofacchin/msc-project) ![TensorFlow Requirement: 2.x](https://img.shields.io/badge/TensorFlow%20Requirement-2.x-brightgreen)

This repository documents the results and contributions of my dissertation, which concludes a two years Part-Time Master of Science in Artificial Intelligence at the University of Edinburgh. The methods developed are derived from the work of Chethan Pandarinath et al. on LFADS [[1]](#1) and the work of [@colehurwitz]( https://github.com/colehurwitz), [@mhhennig](https://github.com/mhhennig) and [@NinelK](https://github.com/NinelK) on TNDM [[2]](#2).

The full dissertation is available at [this link](https://alefacchin.com/assets/documents/dissertation.pdf).

## Abstract

Human brain encodes information through activity patterns in populations of neurons. Recent advancements in multi-electrode interfaces allow researchers to record large clusters of neurons simultaneously with a single neuron precision. Latent variables models have been shown effective in extracting signals from these populations while disregarding the background noise, typical of in neural recordings. More recently, a few models have tried to represent both neural activity and behaviour through latent variables extracted from neural recordings. In this project, we focus our attention on LFADS [[1]](#1), the state-of-the-art method for the extraction of latent variables from neu- ral recordings. We then compare the performance of LFADS with TNDM [[2]](#2), a recent extension of the same model that takes into account behaviour. We show TNDM out- performs LFADS for small training samples, that it can decode wrist electromyography (EMG) signals when applied to the primary motor cortex (M1) and that it is able of disentangling behaviour-relevant and behaviour-irrelevant latent variables. Among the contributions of this project, a new Python implementation for both algorithms based on TensorFlow2. The novel implementation offers faster training, a simpler interface and a cleaner, shorter codebase, which will be easier to maintain. This implementation was employed for all the experiments provided in this project.

## References

<a id="1">[1]</a> 
Chethan Pandarinath et al. Inferring single-trial neural population dynamics using sequential auto-encoders. June 2017. DOI: 10.1101/152884.

<a id="2">[2]</a> 
Cole Hurwitz et al. “Targeted Neural Dynamical Modeling”. In: Advances in Neural Information Processing Systems 34. Unpublished, submitted.
