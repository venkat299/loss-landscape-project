# Width Study

## Overview

This report compares models grouped by hidden layer width using final test performance statistics.

## Model performance summary

| Hidden Size | Dataset(s) | Activation(s) | Optimizer(s) | Mean Test Loss | Mean Test Accuracy |
| --------- | ---------- | ------------- | ----------- | -------------- | ------------------- |
| 100 | circles, gaussians, moons, xor | gelu, relu, tanh | adam, sgd | 0.0149 | 0.9959 |
| 250 | circles, gaussians, moons, xor | gelu, relu, tanh | adam, sgd | 0.0210 | 0.9911 |
| 50 | circles, gaussians, moons, xor | gelu, relu, tanh | adam, sgd | 0.0627 | 0.9807 |
| 500 | circles, gaussians, moons, xor | gelu, relu, tanh | adam, sgd | 0.0856 | 0.9578 |