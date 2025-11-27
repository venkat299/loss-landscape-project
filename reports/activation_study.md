# Activation Study

## Overview

This report compares activation functions (ReLU, Tanh, GELU) across architectures and optimizers using final test metrics.

## Model performance summary

| Activation | Dataset(s) | Activation(s) | Optimizer(s) | Mean Test Loss | Mean Test Accuracy |
| --------- | ---------- | ------------- | ----------- | -------------- | ------------------- |
| gelu | circles, gaussians, moons, xor | gelu | adam, sgd | 0.0177 | 0.9951 |
| relu | circles, gaussians, moons, xor | relu | adam, sgd | 0.0163 | 0.9969 |
| tanh | circles, gaussians, moons, xor | tanh | adam, sgd | 0.0855 | 0.9609 |