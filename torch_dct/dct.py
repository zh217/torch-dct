import numpy as np
import torch


def dct1(x):
    """
    Discrete Cosine Transform, Type I

    :return:
    """
    ndim = len(x.shape)

    return torch.rfft(torch.cat([x, x.flip([ndim - 1])[:, 1:-1]], dim=ndim - 1), 1)[:, :, 0]


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I

    :param X:
    :return:
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    :return:
    """
    ndim = len(x.shape)

    N = x.shape[ndim - 1]

    v = torch.cat([x[::2], x[1::2].flip()], dim=ndim - 1)

    V = torch.rfft(v, 1)[:, :, 0]

    k = torch.arange(N)[None, None, :]

    return V * 2 * torch.exp(-1j * np.pi * k / (2 * N))


def idct():
    """
    Discrete Cosine Transform, Type III (a.k.a. the inverse DCT)

    :return:
    """

    pass
