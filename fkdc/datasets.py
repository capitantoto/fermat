import numpy as np
from scipy import stats
from seaborn import load_dataset as sns_load_dataset
from seaborn import scatterplot as sns_scatterplot
from sklearn.datasets import (
    fetch_openml,
    load_digits,
    load_iris,
    load_wine,
    make_circles,
    make_moons,
)
from sklearn.utils import Bunch

from fkdc.eyeglasses import eyeglasses


def hacer_espiral_fermat(a=1, n_samples=200, turns=2, noise=None):
    phi = np.arange(n_samples) / n_samples * 2 * np.pi * turns
    arm = np.vstack([a * np.sqrt(phi) * np.cos(phi), a * np.sqrt(phi) * np.sin(phi)]).T
    X = np.vstack([arm, -arm])
    if noise:
        X += stats.norm(scale=noise).rvs(size=(n_samples * 2, 2))
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    return X, y


def hacer_anteojos(n_sample=400, noise=0, separation=3, **eye_kwarg):
    eye_kwarg["n"] = n_samples // 2
    eye_kwarg["separation"] = separation
    X0 = eyeglasses(**eye_kwarg)
    if noise:
        X0 += stats.norm(scale=noise).rvs(size=X0.shape)
    n_eyes = n_samples // 4
    X1 = np.vstack(
        [
            stats.norm(-separation / 2, 1 / 6).rvs(n_eyes),
            stats.norm(0, 2 / 6).rvs(n_eyes),
        ]
    ).T
    X2 = np.vstack(
        [
            stats.norm(separation / 2, 1 / 6).rvs(n_eyes),
            stats.norm(0, 2 / 6).rvs(n_eyes),
        ]
    ).T
    X = np.vstack([X0, X1, X2])
    y = np.hstack(
        [np.zeros(n_samples // 2), np.ones(n_eyes), 2 * np.ones(n_eyes)]
    ).astype(str)
    return X, y


class Dataset(Bunch):
    def __init__(self, nombre, X=None, y=None):
        self.nombre = nombre
        self.X = X
        self.y = y
        self.n, self.p = X.shape
        self.k = len(np.unique(y))

    def __str__(self):
        return f"Dataset('{self.nombre}', n={self.n}, p={self.p}, k={self.k})"

    def scatter(self, x=0, y=1, ax=None):
        sns_scatterplot(x=self.X[:, x], y=self.X[:, y], hue=self.y, ax=ax)


n_samples = 400
datasets = [
    ("iris", *load_iris(return_X_y=True)),
    ("lunas", *make_moons(n_samples=n_samples)),
    ("circulos", *make_circles(n_samples=n_samples)),
    ("espirales", *hacer_espiral_fermat(n_samples=n_samples)),
    ("mnist", *fetch_openml("mnist_784", version=1, return_X_y=True)),
    ("vino", *load_wine(return_X_y=True)),
    (f"noisy_lunas_{n_samples}", *make_moons(n_samples=n_samples, noise=0.35)),
    (f"noisy_circulos_{n_samples}", *make_circles(n_samples=n_samples, noise=0.1)),
    (
        f"noisy_espirales_{n_samples}",
        *hacer_espiral_fermat(n_samples=n_samples // 2, noise=0.15),
    ),
    (f"2noisy_lunas_{n_samples}", *make_moons(n_samples=n_samples, noise=0.525)),
    (f"2noisy_circulos_{n_samples}", *make_circles(n_samples=n_samples, noise=0.15)),
    (
        f"2noisy_espirales_{n_samples}",
        *hacer_espiral_fermat(n_samples=n_samples // 2, noise=0.225),
    ),
    ("digitos", *load_digits(return_X_y=True)),
]
datasets = Bunch(**{nombre: Dataset(nombre, X, y) for nombre, X, y in datasets})

penguins = sns_load_dataset("penguins").dropna()
penguins_keep = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
datasets.pinguinos = Dataset(
    "pinguinos", penguins[penguins_keep].values, penguins.species.values
)
datasets.anteojos = Dataset(
    "anteojos", *hacer_anteojos(n_sample=n_samples, bridge_height=0.6, noise=0.15)
)
