# DCT (Discrete Cosine Transform) for pytorch

[![Build Status](https://travis-ci.com/zh217/torch-dct.svg?branch=master)](https://travis-ci.com/zh217/torch-dct)
[![codecov](https://codecov.io/gh/zh217/torch-dct/branch/master/graph/badge.svg)](https://codecov.io/gh/zh217/torch-dct)
[![PyPI version](https://img.shields.io/pypi/v/torch-dct.svg)](https://pypi.python.org/pypi/torch-dct/)
[![PyPI version](https://img.shields.io/pypi/pyversions/torch-dct.svg)](https://pypi.python.org/pypi/torch-dct/)
[![PyPI status](https://img.shields.io/pypi/status/torch-dct.svg)](https://pypi.python.org/pypi/torch-dct/)
[![GitHub license](https://img.shields.io/github/license/zh217/torch-dct.svg)](https://github.com/zh217/torch-dct/blob/master/LICENSE)


This library implements DCT in terms of the built-in FFT operations in pytorch so that
back propagation works through it, on both CPU and GPU. For more information on
DCT and the algorithms used here, see 
[Wikipedia](https://en.wikipedia.org/wiki/Discrete_cosine_transform) and the paper by
[J. Makhoul](https://ieeexplore.ieee.org/document/1163351/). This
[StackExchange article](https://dsp.stackexchange.com/questions/2807/fast-cosine-transform-via-fft)
might also be helpful.

The following are currently implemented:

* 1-D DCT-I and its inverse (which is a scaled DCT-I)
* 1-D DCT-II and its inverse (which is a scaled DCT-III)
* 2-D DCT-II and its inverse (which is a scaled DCT-III)
* 3-D DCT-II and its inverse (which is a scaled DCT-III)

## Install

```
pip install torch-dct
```

Requires `torch>=0.4.1` (lower versions are probably OK but I haven't tested them).

You can run test by getting the source and run `pytest`. To run the test you also
need `scipy` installed.

## Usage

```python
import torch
import torch_dct as dct

x = torch.randn(200)
X = dct.dct(x)   # DCT-II done through the last dimension
y = dct.idct(X)  # scaled DCT-III done through the last dimension
assert (torch.abs(x - y)).sum() < 1e-10  # x == y within numerical tolerance
```

`dct.dct1` and `dct.idct1` are for DCT-I and its inverse. The usage is the same.

Just replace `dct` and `idct` by `dct_2d`, `dct_3d`, `idct_2d`, `idct_3d`, etc
to get the multidimensional versions.
