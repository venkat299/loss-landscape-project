# Optimizer Study

## Overview

This report compares SGD and Adam optimizers across datasets, architectures, and activations.

## Model performance summary

| Optimizer | Dataset(s) | Activation(s) | Optimizer(s) | Mean Test Loss | Mean Test Accuracy |
| --------- | ---------- | ------------- | ----------- | -------------- | ------------------- |
| adam | circles, gaussians, moons, xor | gelu, relu, tanh | adam | 0.0049 | 0.9993 |
| sgd | circles, gaussians, moons, xor | gelu, relu, tanh | sgd | 0.0748 | 0.9693 |