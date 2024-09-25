import pickle
from numbers import Number

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from seaborn import scatterplot as sns_scatterplot
from sklearn.utils import Bunch
from sklearn.utils import shuffle as sk_shuffle
from sklearn.utils.multiclass import unique_labels

from fkdc.utils import eyeglasses


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
    if scale is None:
        scale = np.std(X)
    elif isinstance(scale, callable):
        scale = scale(X)
    assert isinstance(scale, Number), "`scale` debe ser None, un callable o un escalar"
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

    def de_fabrica(factory, ruido=None, nombre=None, **factory_params):
        params = Bunch(**(factory_params or {}))
        X, y = factory(**params)
        if ruido:
            X = agregar_dims_ruido(X, **({} if ruido is True else ruido))
        params.factory = factory.__name__
        params.ruido = ruido
        ds = Dataset(X, y, nombre)
        ds.params = params
        return ds

    def __str__(self):
        return f"Dataset('{self.nombre}', n={self.n}, p={self.p}, k={self.k})"

    def __hash__(self):
        return hash(tuple(tuple(row) for row in self.X)) * hash(tuple(self.y)) % 2**64

    def scatter(self, x=0, y=1, ax=None, **plot_kws):
        sns_scatterplot(x=self.X[:, x], y=self.X[:, y], hue=self.y, ax=ax, **plot_kws)

    def scatter_3d(self, x=0, y=1, z=2, ax=None, **plot_kws):
        if self.p < 3:
            raise ValueError(f"{self.nombre} tiene sólo {self.p} dimensiones")
        ax = ax or plt.gcf().add_subplot(projection="3d")
        assert ax.name == "3d", "El eje (`ax`) debe tener proyección 3d"
        for i, lbl in enumerate(self.labels):
            X_lbl = self.X[self.y == lbl]
            X, Y, Z = X_lbl[:, x], X_lbl[:, y], X_lbl[:, z]
            ax.scatter(X, Y, Z, c=f"C{i}", label=str(lbl), **plot_kws)
        ax.legend(title="Clase")

    def guardar(self, archivo=None):
        with open(archivo or f"{hash(self)}.pkl", "wb") as file:
            pickle.dump(self, file)
