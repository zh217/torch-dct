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
    for norm in [None, 'ortho']:
        for N in [2, 3, 5, 32, 111]:
            x = np.random.normal(size=(1, N,))
            ref = fftpack.dct(x, type=2, norm=norm)
            act = dct.dct(torch.tensor(x), norm=norm).numpy()
            assert np.abs(ref - act).max() < EPS, (norm, N)

        for d in [2, 3, 4, 11]:
            x = np.random.normal(size=(2,) * d)
            ref = fftpack.dct(x, type=2, norm=norm)
            act = dct.dct(torch.tensor(x), norm=norm).numpy()
            assert np.abs(ref - act).max() < EPS, (norm, d)


def test_idct():
    for norm in [None, 'ortho']:
        for N in [5, 2, 32, 111]:
            x = np.random.normal(size=(1, N))
            X = dct.dct(torch.tensor(x), norm=norm)
            y = dct.idct(X, norm=norm).numpy()
            assert np.abs(x - y).max() < EPS, x


def test_cuda():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

        for N in [2, 5, 32, 111]:
            x = np.random.normal(size=(1, N,))
            ref = fftpack.dct(x, type=1)
            act = dct.dct1(torch.tensor(x, device=device)).cpu().numpy()
            assert np.abs(ref - act).max() < EPS, ref

        for d in [2, 3, 4]:
            x = np.random.normal(size=(2,) * d)
            ref = fftpack.dct(x, type=1)
            act = dct.dct1(torch.tensor(x, device=device)).cpu().numpy()
            assert np.abs(ref - act).max() < EPS, ref

        for norm in [None, 'ortho']:
            for N in [2, 3, 5, 32, 111]:
                x = np.random.normal(size=(1, N,))
                ref = fftpack.dct(x, type=2, norm=norm)
                act = dct.dct(torch.tensor(x, device=device), norm=norm).cpu().numpy()
                assert np.abs(ref - act).max() < EPS, (norm, N)

            for d in [2, 3, 4, 11]:
                x = np.random.normal(size=(2,) * d)
                ref = fftpack.dct(x, type=2, norm=norm)
                act = dct.dct(torch.tensor(x, device=device), norm=norm).cpu().numpy()
                assert np.abs(ref - act).max() < EPS, (norm, d)

            for N in [5, 2, 32, 111]:
                x = np.random.normal(size=(1, N))
                X = dct.dct(torch.tensor(x, device=device), norm=norm)
                y = dct.idct(X, norm=norm).cpu().numpy()
                assert np.abs(x - y).max() < EPS, x

def test_dct_2d():
    for N1 in [2, 5, 32]:
        for N2 in [2, 5, 32]:
            x = np.random.normal(size=(1, N1, N2))
            ref = fftpack.dct(x, axis=2, type=2)
            ref = fftpack.dct(ref, axis=1, type=2)
            act = dct.dct_2d(torch.tensor(x)).numpy()
            assert np.abs(ref - act).max() < EPS, (ref, act)


def test_idct_2d():
    for N1 in [2, 5, 32]:
        for N2 in [2, 5, 32]:
            x = np.random.normal(size=(1, N1, N2))
            X = dct.dct_2d(torch.tensor(x))
            y = dct.idct_2d(X).numpy()
            assert np.abs(x - y).max() < EPS, x


def test_dct_3d():
    for N1 in [2, 5, 32]:
        for N2 in [2, 5, 32]:
            for N3 in [2, 5, 32]:
                x = np.random.normal(size=(1, N1, N2, N3))
                ref = fftpack.dct(x, axis=3, type=2)
                ref = fftpack.dct(ref, axis=2, type=2)
                ref = fftpack.dct(ref, axis=1, type=2)
                act = dct.dct_3d(torch.tensor(x)).numpy()
                assert np.abs(ref - act).max() < EPS, (ref, act)


def test_idct_3d():
    for N1 in [2, 5, 32]:
        for N2 in [2, 5, 32]:
            for N3 in [2, 5, 32]:
                x = np.random.normal(size=(1, N1, N2, N3))
                X = dct.dct_3d(torch.tensor(x))
                y = dct.idct_3d(X).numpy()
                assert np.abs(x - y).max() < EPS, x
