import numpy as np
from itertools import product

def iqr(X):
    return np.percentile(X, 75) - np.percentile(X, 25)


def pilot_h(dists):
    return 0.9 * np.minimum(dists.std(), iqr(dists) / 1.34) * len(dists) ** (-1 / 5)


def lattice(a, b, step=1, dim=2, array=True):
    side = np.arange(a, b, step)
    if len(side) ** dim > 1e6:
        raise ValueError(
            f"Too many points ({len(side) ** dim:.2e} > 1e6). Try a bigger step or a smaller dim."
        )
    gen = product(*[side] * dim)
    return np.array([*gen]) if array else gen


