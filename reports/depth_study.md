# Depth Study

## Overview

This report compares models grouped by depth (number of hidden layers) using final test performance statistics.

## Model performance summary

| Hidden Layers | Dataset(s) | Activation(s) | Optimizer(s) | Mean Test Loss | Mean Test Accuracy |
| --------- | ---------- | ------------- | ----------- | -------------- | ------------------- |
| 1 | moons | gelu, relu, tanh | adam, sgd | 0.0887 | 0.9662 |
| 2 | moons | gelu, relu, tanh | adam, sgd | 0.0448 | 0.9848 |
| 4 | moons | gelu, relu, tanh | adam, sgd | 0.0214 | 0.9916 |