# Activation Study

## Overview

This report compares activation functions (ReLU, Tanh, GELU) across architectures and optimizers using final test metrics.

## Model performance summary

| Activation | Dataset(s) | Activation(s) | Optimizer(s) | Mean Test Loss | Mean Test Accuracy |
| --------- | ---------- | ------------- | ----------- | -------------- | ------------------- |
| gelu | moons | gelu | adam, sgd | 0.0305 | 0.9908 |
| relu | moons | relu | adam, sgd | 0.0280 | 0.9941 |
| tanh | moons | tanh | adam, sgd | 0.1004 | 0.9554 |