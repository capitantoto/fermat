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
    y = np.hstack([np.zeros(n0), np.ones(n1)]).astype(int)
    if shuffle:
        y = sk_shuffle(y, random_state=rng)
    return y


def hacer_espiral_fermat(
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
    X_anteojos = eyeglasses(**eye_kwarg)
    if noise:
        X_anteojos += rng.normal(scale=noise, size=X_anteojos.shape)
    X_ojos = np.vstack(
        [
            stats.norm(separation / 2, 1 / 6).rvs(n_ojos, random_state=rng),
            stats.norm(0, 2 / 6).rvs(n_ojos, random_state=rng),
        ]
    ).T
    y_ojos = np.hstack([np.ones(n_oi), 2 * np.ones(n_od)])
    X_ojos[y_ojos == 1, 0] *= -1
    X = np.vstack([X_anteojos, X_ojos])
    y = np.hstack([np.zeros(n_anteojos), y_ojos]).astype(int)
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
    X_x = centro[0] + radio * np.cos(seeds + np.pi * y)
    X_y = centro[1] + radio * np.sin(seeds + np.pi * y)
    X = np.stack([X_x, X_y, X_z], axis=1)
    if noise:
        X += rng.normal(scale=noise, size=X.shape)
    return X, y


def make_eslabones(
    n_samples=200,
    centro=(0, 0),
    radio=1,
    noise=None,
    random_state=None,
    shuffle=False,
):
    rng = np.random.default_rng(random_state)
    y = _dos_muestras(n_samples, rng, shuffle)
    seeds = np.random.uniform(size=len(y)) * 2 * np.pi
    X = [
        centro[0] + radio * np.cos(seeds),
        centro[1] + radio * np.sin(seeds),
        np.zeros_like(seeds),
    ]
    X[y == 1, 0] += radio  # traslada eslabon 1 `radio`, como el logo de cierta TC
    X[y == 1, [1, 2]] = X[y == 1, [2, 1]]  # roto eslabon 1, queda perpendicular al 0
    if noise:
        X += rng.normal(scale=noise, size=X.shape)
    return X, y


class Dataset(Bunch):
    def __init__(self, nombre, X=None, y=None):
        self.nombre = nombre
        self.X = X
        self.y = y.astype(str)
        self.n, self.p = X.shape
        self.k = len(np.unique(y))

    def __str__(self):
        return f"Dataset('{self.nombre}', n={self.n}, p={self.p}, k={self.k})"

    def scatter(self, x=0, y=1, ax=None, **plot_kws):
        sns_scatterplot(x=self.X[:, x], y=self.X[:, y], hue=self.y, ax=ax, **plot_kws)


n_samples = 400
datasets = [
    ("iris", *load_iris(return_X_y=True)),
    ("lunas", *make_moons(n_samples=n_samples)),
    ("circulos", *make_circles(n_samples=n_samples)),
    ("espirales", *hacer_espiral_fermat(n_samples=n_samples)),
    # ("mnist", *fetch_openml("mnist_784", version=1, return_X_y=True)),
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
    "anteojos", *hacer_anteojos(n_samples=n_samples, bridge_height=0.6, noise=0.15)
)
