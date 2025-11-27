# Depth Study

## Overview

This report compares models grouped by depth (number of hidden layers) using final test performance statistics.

## Model performance summary

| Hidden Layers | Dataset(s) | Activation(s) | Optimizer(s) | Mean Test Loss | Mean Test Accuracy |
| --------- | ---------- | ------------- | ----------- | -------------- | ------------------- |
| 1 | circles, gaussians, moons, xor | gelu, relu, tanh | adam, sgd | 0.0742 | 0.9693 |
| 2 | circles, gaussians, moons, xor | gelu, relu, tanh | adam, sgd | 0.0265 | 0.9921 |
| 4 | circles, gaussians, moons, xor | gelu, relu, tanh | adam, sgd | 0.0122 | 0.9954 |