import pickle
from numbers import Number

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from seaborn import load_dataset as sns_load_dataset
from seaborn import scatterplot as sns_scatterplot
from sklearn.datasets import (  # fetch_openml,
    load_digits,
    load_iris,
    load_wine,
    make_circles,
    make_moons,
)
from sklearn.utils import Bunch
from sklearn.utils import shuffle as sk_shuffle
from sklearn.utils.multiclass import unique_labels

from fkdc.eyeglasses import eyeglasses


def _dos_muestras(n_samples, random_state=None, shuffle=False):
    rng = np.random.default_rng(random_state)
    if isinstance(n_samples, int):
        n0 = n1 = n_samples // 2
        n0 += n_samples % 2
    elif isinstance(n_samples, tuple) and len(n_samples) == 2:
        assert all(isinstance(n, int) for n in n_samples)
        n0, n1 = n_samples
        n_samples = sum(n_samples)
    else:
        raise ValueError("`n_samples` debe ser un entero o una 2-tupla de enteros")
    y = np.hstack([np.zeros(n0, int), np.ones(n1, int)])
    if shuffle:
        y = sk_shuffle(y, random_state=rng)
    return y


def hacer_espirales(
    a=1, n_samples=200, turns=2, noise=None, random_state=None, shuffle=False
):
    rng = np.random.default_rng(random_state)
    y = _dos_muestras(n_samples, rng, shuffle)
    phi = stats.uniform(0, 2 * np.pi * turns).rvs(len(y), random_state=rng)
    X = np.vstack([a * np.sqrt(phi) * np.cos(phi), a * np.sqrt(phi) * np.sin(phi)]).T
    if noise:
        X += rng.normal(scale=noise, size=X.shape)
    X[y == 1] *= -1
    return X, y


def hacer_anteojos(
    n_samples=400,
    separation=3,
    noise=None,
    random_state=None,
    shuffle=False,
    **eye_kwarg,
):
    rng = np.random.default_rng(random_state)
    if isinstance(n_samples, int):  # 1/2 en anteojos, 1/4 en c/ojo
        n_anteojos = n_ojos = n_samples // 2
        n_anteojos += n_samples % 2
        n_oi = n_od = n_ojos // 2
        n_oi += n_ojos % 2
    elif isinstance(n_samples, tuple) and len(n_samples) == 3:
        assert all(isinstance(n, int) for n in n_samples)
        n_anteojos, n_oi, n_od = n_samples
        n_ojos = n_oi + n_od
        n_samples = sum(n_samples)
    else:
        raise ValueError("`n_samples` debe ser un entero o una 3-tupla de enteros")
    eye_kwarg["n"] = n_anteojos
    eye_kwarg["separation"] = separation
    X_anteojos = eyeglasses(**eye_kwarg, random_state=rng)
    if noise:
        X_anteojos += rng.normal(scale=noise, size=X_anteojos.shape)
    X_ojos = np.vstack(
        [
            stats.norm(separation / 2, 1 / 6).rvs(n_ojos, random_state=rng),
            stats.norm(0, 2 / 6).rvs(n_ojos, random_state=rng),
        ]
    ).T
    y_ojos = np.hstack([np.ones(n_oi, int), 2 * np.ones(n_od, int)])
    X_ojos[y_ojos == 1, 0] *= -1
    X = np.vstack([X_anteojos, X_ojos])
    y = np.hstack([np.zeros(n_anteojos, int), y_ojos])
    if shuffle:
        X, y = sk_shuffle(X, y, random_state=rng)
    return X, y


def hacer_helices(
    n_samples=400,
    centro=(0, 0),
    radio=1,
    altura=1,
    vueltas=5,
    noise=None,
    random_state=None,
    shuffle=False,
):
    rng = np.random.default_rng(random_state)
    y = _dos_muestras(n_samples, rng, shuffle)
    seeds = rng.uniform(size=len(y)) * vueltas * 2 * np.pi
    X_z = seeds * altura / (2 * np.pi)
    X_x = centro[0] + radio * np.cos(seeds + np.pi * y)  # `np * y` rota clase 1 180º
    X_y = centro[1] + radio * np.sin(seeds + np.pi * y)  # sobre el eje Z
    X = np.vstack([X_x, X_y, X_z]).T
    if noise:
        X += rng.normal(scale=noise, size=X.shape)
    return X, y


def hacer_eslabones(
    n_samples=200,
    centro=(0, 0),
    radio=1,
    noise=None,
    random_state=None,
    shuffle=False,
):
    rng = np.random.default_rng(random_state)
    y = _dos_muestras(n_samples, rng, shuffle)
    seeds = rng.uniform(size=len(y)) * 2 * np.pi
    X_x = centro[0] + radio * np.cos(seeds)
    X_y = centro[1] + radio * np.sin(seeds)
    X_z = np.zeros_like(seeds)
    X = np.vstack([X_x, X_y, X_z]).T
    mask = y == 1  # "mascara" para observaciones del eslabon 1
    X[mask, 0] += radio  # traslada `1 * radio` el eje x, como el logo de cierta TC
    X[mask, 1], X[mask, 2] = X[mask, 2], X[mask, 1]  # roto 90º sobre el eje X
    if noise:
        X += rng.normal(scale=noise, size=X.shape)
    return X, y


def hacer_hueveras(
    n_samples=200, limites=(10, 10), noise=None, random_state=None, shuffle=False
):
    rng = np.random.default_rng(random_state)
    y = _dos_muestras(n_samples, rng, shuffle)
    X_x = rng.uniform(0, limites[0], size=len(y))
    X_y = rng.uniform(0, limites[1], size=len(y))
    X_z = np.sin(X_x) * np.sin(X_y)
    X_z[y == 1] *= -1  # "doy vuelta" la huevera de la clase 1
    X = np.vstack([X_x, X_y, X_z]).T
    if noise:
        X += rng.normal(scale=noise, size=X.shape)
    return X, y


def hacer_pionono(
    n_samples=400,
    esquinas=(0.3, 0.7),
    desvios=4,
    noise=0.01,
    random_state=None,
    shuffle=False,
):
    # TODO: Credit Sapienza & lle.py de noseque libro con swissroll original
    assert (len(esquinas) == 2) and all(isinstance(n, Number) for n in esquinas)
    rng = np.random.default_rng(random_state)
    if isinstance(n_samples, int):  # 1/2 en anteojos, 1/4 en c/ojo
        base, resto = n_samples // 4, n_samples % 4
        ns_clase = base + np.array([i < resto for i in range(4)], np.int32)
    elif isinstance(n_samples, tuple) and len(n_samples) == 4:
        assert all(isinstance(n, int) for n in n_samples)
        ns_clase = n_samples
        n_samples = sum(n_samples)
    else:
        raise ValueError("`n_samples` debe ser un entero o una 3-tupla de enteros")
    y = np.concatenate([np.ones(n_cls, int) * i for i, n_cls in enumerate(ns_clase)])
    if shuffle:
        y = sk_shuffle(y, random_state=rng)
    centros = [(x, y) for x in esquinas for y in esquinas]
    seeds = rng.normal(scale=(esquinas[1] - esquinas[0]) / desvios, size=(n_samples, 2))
    for i in range(4):
        seeds[y == i] += centros[i]
    t = 2 * np.pi * (1 + 2 * seeds[:, 0])
    X = np.vstack([t * np.cos(t), 21 * seeds[:, 1], t * np.sin(t)]).T
    if noise:
        X += rng.normal(scale=noise, size=X.shape)
    return X, y


def agregar_dims_ruido(X, ndims=None, scale=None, random_state=None):
    rng = np.random.default_rng(random_state)
    scale = scale or np.std(X)
    ndims = ndims or (3 * X.shape[1])
    ruido = rng.normal(scale=scale, size=(X.shape[0], ndims))
    return np.hstack([X, ruido])


class Dataset:
    def __init__(self, X=None, y=None, nombre=None):
        self.X = X
        self.y = y.astype(str)
        self.nombre = nombre
        self.n, self.p = X.shape
        self.labels = unique_labels(self.y)
        self.k = len(self.labels)

    def de_fabrica(factory, factory_params=None, ruido_params=None, nombre=None):
        params = Bunch(**(factory_params or {}))
        X, y = factory(**params)
        if ruido_params:
            X = agregar_dims_ruido(X, **({} if ruido_params is True else ruido_params))
        params.factory = factory.__name__
        params.ruido = ruido_params
        ds = Dataset(X, y, nombre)
        ds.params = params
        return ds

    def __str__(self):
        return f"Dataset('{self.nombre}', n={self.n}, p={self.p}, k={self.k})"

    def __hash__(self):
        return hash(tuple(tuple(row) for row in self.X)) * hash(tuple(self.y)) % 2**64

    def scatter(self, x=0, y=1, ax=None, **plot_kws):
        sns_scatterplot(x=self.X[:, x], y=self.X[:, y], hue=self.y, ax=ax, **plot_kws)

    def scatter_3d(self, x=0, y=1, z=2, **plot_kws):
        if self.p < 3:
            raise ValueError(f"{self.nombre} tiene sólo {self.p} dimensiones")
        ax = plt.gcf().add_subplot(projection="3d")
        for i, lbl in enumerate(self.labels):
            X_lbl = self.X[self.y == lbl]
            ax.scatter(X_lbl[:, x], X_lbl[:, y], X_lbl[:, z], c=f"C{i}", label=str(lbl))
        ax.legend(title="Clase")

    def guardar(self, archivo=None):
        with open(archivo or f"{hash(self)}.pkl", "wb") as file:
            pickle.dump(self, file)


n_samples = 400
datasets = [
    ("iris", *load_iris(return_X_y=True)),
    ("lunas", *make_moons(n_samples=n_samples)),
    ("circulos", *make_circles(n_samples=n_samples)),
    ("espirales", *hacer_espirales(n_samples=n_samples)),
    # ("mnist", *fetch_openml("mnist_784", version=1, return_X_y=True)),
    ("vino", *load_wine(return_X_y=True)),
    (f"noisy_lunas_{n_samples}", *make_moons(n_samples=n_samples, noise=0.35)),
    (f"noisy_circulos_{n_samples}", *make_circles(n_samples=n_samples, noise=0.1)),
    (
        f"noisy_espirales_{n_samples}",
        *hacer_espirales(n_samples=n_samples // 2, noise=0.15),
    ),
    (f"2noisy_lunas_{n_samples}", *make_moons(n_samples=n_samples, noise=0.525)),
    (f"2noisy_circulos_{n_samples}", *make_circles(n_samples=n_samples, noise=0.15)),
    (
        f"2noisy_espirales_{n_samples}",
        *hacer_espirales(n_samples=n_samples // 2, noise=0.225),
    ),
    ("digitos", *load_digits(return_X_y=True)),
]
datasets = Bunch(**{nombre: Dataset(X, y, nombre) for nombre, X, y in datasets})

penguins = sns_load_dataset("penguins").dropna()
penguins_keep = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
datasets.pinguinos = Dataset(
    penguins[penguins_keep].values, penguins.species.values, "pinguinos"
)
datasets.anteojos = Dataset(
    *hacer_anteojos(n_samples=n_samples, bridge_height=0.6, noise=0.15), "anteojos"
)
