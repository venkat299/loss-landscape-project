# Width Study

## Overview

This report compares models grouped by hidden layer width using final test performance statistics.

## Model performance summary

| Hidden Size | Dataset(s) | Activation(s) | Optimizer(s) | Mean Test Loss | Mean Test Accuracy |
| --------- | ---------- | ------------- | ----------- | -------------- | ------------------- |
| 100 | moons | gelu, relu, tanh | adam, sgd | 0.0246 | 0.9922 |
| 250 | moons | gelu, relu, tanh | adam, sgd | 0.0383 | 0.9836 |
| 50 | moons | gelu, relu, tanh | adam, sgd | 0.0946 | 0.9630 |
| 500 | moons | gelu, relu, tanh | adam, sgd | 0.0828 | 0.9694 |