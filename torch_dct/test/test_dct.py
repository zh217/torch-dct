import torch_dct as dct
import scipy.fftpack as fftpack
import numpy as np
import torch

np.random.seed(1)

EPS = 1e-10


def test_dct1():
    for N in [2, 5, 32, 111]:
        x = np.random.normal(size=(1, N,))
        ref = fftpack.dct(x, type=1)
        act = dct.dct1(torch.tensor(x)).numpy()
        assert np.abs(ref - act).max() < EPS, ref

    for d in [2, 3, 4]:
        x = np.random.normal(size=(2,) * d)
        ref = fftpack.dct(x, type=1)
        act = dct.dct1(torch.tensor(x)).numpy()
        assert np.abs(ref - act).max() < EPS, ref


def test_idct1():
    for N in [2, 5, 32, 111]:
        x = np.random.normal(size=(1, N))
        X = dct.dct1(torch.tensor(x))
        y = dct.idct1(X).numpy()
        assert np.abs(x - y).max() < EPS, x


def test_dct():
    for N in [2, 5, 32, 111]:
        x = np.random.normal(size=(1, N,))
        ref = fftpack.dct(x, type=2)
        act = dct.dct(torch.tensor(x)).numpy()
        assert np.abs(ref - act).max() < EPS, ref

    for d in [2, 3, 4]:
        x = np.random.normal(size=(2,) * d)
        ref = fftpack.dct(x, type=2)
        act = dct.dct(torch.tensor(x)).numpy()
        assert np.abs(ref - act).max() < EPS, ref


def test_idct():
    for N in [5, 2, 32, 111]:
        x = np.random.normal(size=(1, N))
        X = dct.dct(torch.tensor(x))
        y = dct.idct(X).numpy()
        assert np.abs(x - y).max() < EPS, x
