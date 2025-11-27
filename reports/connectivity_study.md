# Connectivity Study

## Overview

Connectivity experiments (linear and curved paths between modes) are summarized via barrier-height annotated plots saved under `reports/figures/.../connectivity/` for each (dataset, architecture, activation, optimizer, seed-pair) combination.

## Connectivity summary

| Dataset | Architecture | Activation | Optimizer | Num Pairs | Mean Train Barrier | Mean Test Barrier | Max Train Barrier | Max Test Barrier |
| ------- | ------------ | ---------- | --------- | --------- | ------------------- | ------------------ | ------------------- | ------------------ |
| moons | 1x500 | gelu | adam | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| moons | 1x500 | gelu | sgd | 3 | 0.0167 | 0.0160 | 0.0285 | 0.0271 |
| moons | 1x500 | relu | adam | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| moons | 1x500 | relu | sgd | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| moons | 1x500 | tanh | adam | 3 | 0.0054 | 0.0046 | 0.0094 | 0.0076 |
| moons | 1x500 | tanh | sgd | 3 | 0.0000 | 0.0002 | 0.0000 | 0.0005 |
| moons | 1x50 | gelu | adam | 3 | 0.0005 | 0.0000 | 0.0015 | 0.0000 |
| moons | 1x50 | gelu | sgd | 3 | 0.0386 | 0.0409 | 0.0633 | 0.0589 |
| moons | 1x50 | relu | adam | 3 | 0.0039 | 0.0002 | 0.0118 | 0.0007 |
| moons | 1x50 | relu | sgd | 3 | 0.0350 | 0.0330 | 0.0589 | 0.0518 |
| moons | 1x50 | tanh | adam | 3 | 0.0710 | 0.0598 | 0.1458 | 0.1185 |
| moons | 1x50 | tanh | sgd | 3 | 0.0105 | 0.0141 | 0.0266 | 0.0332 |
| moons | 2x100 | gelu | adam | 3 | 0.1896 | 0.1736 | 0.2822 | 0.3017 |
| moons | 2x100 | gelu | sgd | 3 | 0.3337 | 0.3309 | 0.5068 | 0.5205 |
| moons | 2x100 | relu | adam | 3 | 0.1397 | 0.1315 | 0.1783 | 0.1679 |
| moons | 2x100 | relu | sgd | 3 | 0.0937 | 0.0906 | 0.1327 | 0.1250 |
| moons | 2x100 | tanh | adam | 3 | 1.4092 | 1.4407 | 1.5695 | 1.6204 |
| moons | 2x100 | tanh | sgd | 3 | 0.0745 | 0.0748 | 0.1157 | 0.1192 |
| moons | 4x100 | gelu | adam | 3 | 0.0361 | 0.0358 | 0.0793 | 0.0824 |
| moons | 4x100 | gelu | sgd | 3 | 0.0000 | 0.0000 | 0.0001 | 0.0000 |
| moons | 4x100 | relu | adam | 3 | 0.2662 | 0.2513 | 0.3776 | 0.3689 |
| moons | 4x100 | relu | sgd | 3 | 0.1411 | 0.1327 | 0.2905 | 0.2597 |
| moons | 4x100 | tanh | adam | 3 | 0.8408 | 0.8503 | 0.9665 | 0.9623 |
| moons | 4x100 | tanh | sgd | 3 | 0.1145 | 0.1056 | 0.1909 | 0.1833 |
| moons | 4x250 | gelu | adam | 3 | 0.2504 | 0.2674 | 0.4032 | 0.4618 |
| moons | 4x250 | gelu | sgd | 3 | 0.0463 | 0.0568 | 0.0927 | 0.0866 |
| moons | 4x250 | relu | adam | 3 | 0.3471 | 0.3409 | 0.3904 | 0.3762 |
| moons | 4x250 | relu | sgd | 3 | 0.2100 | 0.2056 | 0.3335 | 0.3192 |
| moons | 4x250 | tanh | adam | 3 | 1.4574 | 1.5731 | 2.0690 | 2.2866 |
| moons | 4x250 | tanh | sgd | 3 | 0.1034 | 0.1036 | 0.1390 | 0.1415 |