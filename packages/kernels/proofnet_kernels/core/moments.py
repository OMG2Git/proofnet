"""Parallel moment / co-moment merging (Chan et al.). NumPy + stdlib only.

A moment state is (n, mean, M2) with M2 = sum((x - mean)^2) per feature. States with n == 0 are
valid identity elements.
"""

import numpy as np


def moments(x: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Two-pass (n, mean, M2) over axis 0 of a 2-D array. Empty input -> zeros."""
    n = x.shape[0]
    if n == 0:
        z = np.zeros(x.shape[1])
        return 0, z, z.copy()
    mean = x.mean(axis=0)
    d = x - mean
    return n, mean, (d * d).sum(axis=0)


def merge_moments(
    a: tuple[int, np.ndarray, np.ndarray], b: tuple[int, np.ndarray, np.ndarray]
) -> tuple[int, np.ndarray, np.ndarray]:
    na, ma, m2a = a
    nb, mb, m2b = b
    if na == 0:
        return nb, mb.copy(), m2b.copy()
    if nb == 0:
        return na, ma.copy(), m2a.copy()
    n = na + nb
    delta = mb - ma
    mean = ma + delta * (nb / n)
    m2 = m2a + m2b + delta * delta * (na * nb / n)
    return n, mean, m2


def comoments(
    x: np.ndarray, y: np.ndarray
) -> tuple[int, np.ndarray, float, np.ndarray, np.ndarray, float]:
    """(n, mean_x, mean_y, Sxx, Sxy, Syy) with centered cross-products. Empty -> zeros."""
    n, d = x.shape
    if n == 0:
        return 0, np.zeros(d), 0.0, np.zeros((d, d)), np.zeros(d), 0.0
    mx = x.mean(axis=0)
    my = float(y.mean())
    xc = x - mx
    yc = y - my
    return n, mx, my, xc.T @ xc, xc.T @ yc, float(yc @ yc)


def merge_comoments(
    a: tuple[int, np.ndarray, float, np.ndarray, np.ndarray, float],
    b: tuple[int, np.ndarray, float, np.ndarray, np.ndarray, float],
) -> tuple[int, np.ndarray, float, np.ndarray, np.ndarray, float]:
    na, mxa, mya, sxxa, sxya, syya = a
    nb, mxb, myb, sxxb, sxyb, syyb = b
    if na == 0:
        return nb, mxb.copy(), myb, sxxb.copy(), sxyb.copy(), syyb
    if nb == 0:
        return na, mxa.copy(), mya, sxxa.copy(), sxya.copy(), syya
    n = na + nb
    f = na * nb / n
    dx = mxb - mxa
    dy = myb - mya
    return (
        n,
        mxa + dx * (nb / n),
        mya + dy * (nb / n),
        sxxa + sxxb + f * np.outer(dx, dx),
        sxya + sxyb + f * dx * dy,
        syya + syyb + f * dy * dy,
    )
