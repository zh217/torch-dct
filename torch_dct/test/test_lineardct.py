import torch_dct
import scipy.fftpack as fftpack
import numpy as np
import torch

np.random.seed(1)

EPS = 1e-3
# THIS IS NOT HOW THESE LAYERS SHOULD BE USED IN PRACTICE
# only written this way for testing convenience
dct1 = lambda x: torch_dct.LinearDCT(x.size(1), type='dct1')(x).data
idct1 = lambda x: torch_dct.LinearDCT(x.size(1), type='idct1')(x).data
def dct(x, norm=None):
    return torch_dct.LinearDCT(x.size(1), type='dct', norm=norm)(x).data
def idct(x, norm=None):
    return torch_dct.LinearDCT(x.size(1), type='idct', norm=norm)(x).data

dct_2d = lambda x: torch_dct.apply_linear_2d(x, torch_dct.LinearDCT(x.size(1), type='dct')).data
dct_3d = lambda x: torch_dct.apply_linear_3d(x, torch_dct.LinearDCT(x.size(1), type='dct')).data
idct_2d = lambda x: torch_dct.apply_linear_2d(x, torch_dct.LinearDCT(x.size(1), type='idct')).data
idct_3d = lambda x: torch_dct.apply_linear_3d(x, torch_dct.LinearDCT(x.size(1), type='idct')).data

def test_dct1():
    for N in [2, 5, 32, 111]:
        x = np.random.normal(size=(1, N,))
        ref = fftpack.dct(x, type=1)
        act = dct1(torch.tensor(x).float()).numpy()
        assert np.abs(ref - act).max() < EPS, ref

    for d in [2, 3, 4]:
        x = np.random.normal(size=(2,) * d)
        ref = fftpack.dct(x, type=1)
        act = dct1(torch.tensor(x).float()).numpy()
        assert np.abs(ref - act).max() < EPS, ref


def test_idct1():
    for N in [2, 5, 32, 111]:
        x = np.random.normal(size=(1, N))
        X = dct1(torch.tensor(x).float())
        y = idct1(X).numpy()
        assert np.abs(x - y).max() < EPS, x


def test_dct():
    for norm in [None, 'ortho']:
        for N in [2, 3, 5, 32, 111]:
            x = np.random.normal(size=(1, N,))
            ref = fftpack.dct(x, type=2, norm=norm)
            act = dct(torch.tensor(x).float(), norm=norm).numpy()
            assert np.abs(ref - act).max() < EPS, (norm, N)

        for d in [2, 3, 4, 11]:
            x = np.random.normal(size=(2,) * d)
            ref = fftpack.dct(x, type=2, norm=norm)
            act = dct(torch.tensor(x).float(), norm=norm).numpy()
            assert np.abs(ref - act).max() < EPS, (norm, d)


def test_idct():
    for norm in [None, 'ortho']:
        for N in [5, 2, 32, 111]:
            x = np.random.normal(size=(1, N))
            X = dct(torch.tensor(x).float(), norm=norm)
            y = idct(X, norm=norm).numpy()
            assert np.abs(x - y).max() < EPS, x

def test_dct_2d():
    for N1 in [2, 5, 32]:
        x = np.random.normal(size=(1, N1, N1))
        ref = fftpack.dct(x, axis=2, type=2)
        ref = fftpack.dct(ref, axis=1, type=2)
        act = dct_2d(torch.tensor(x).float()).numpy()
        assert np.abs(ref - act).max() < EPS, (ref, act)


def test_idct_2d():
    for N1 in [2, 5, 32]:
        x = np.random.normal(size=(1, N1, N1))
        X = dct_2d(torch.tensor(x).float())
        y = idct_2d(X).numpy()
        assert np.abs(x - y).max() < EPS, x


def test_dct_3d():
    for N1 in [2, 5, 32]:
        x = np.random.normal(size=(1, N1, N1, N1))
        ref = fftpack.dct(x, axis=3, type=2)
        ref = fftpack.dct(ref, axis=2, type=2)
        ref = fftpack.dct(ref, axis=1, type=2)
        act = dct_3d(torch.tensor(x).float()).numpy()
        assert np.abs(ref - act).max() < EPS, (ref, act)


def test_idct_3d():
    for N1 in [2, 5, 32]:
            x = np.random.normal(size=(1, N1, N1, N1))
            X = dct_3d(torch.tensor(x).float())
            y = idct_3d(X).numpy()
            assert np.abs(x - y).max() < EPS, x
