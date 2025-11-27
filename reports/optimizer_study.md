# Optimizer Study

## Overview

This report compares SGD and Adam optimizers across datasets, architectures, and activations.

## Model performance summary

| Optimizer | Dataset(s) | Activation(s) | Optimizer(s) | Mean Test Loss | Mean Test Accuracy |
| --------- | ---------- | ------------- | ----------- | -------------- | ------------------- |
| adam | moons | gelu, relu, tanh | adam | 0.0043 | 0.9997 |
| sgd | moons | gelu, relu, tanh | sgd | 0.1016 | 0.9605 |