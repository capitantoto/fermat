import datetime as dt
import logging
import pickle
from numbers import Number
from pathlib import Path
from typing import TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from scipy import stats
from seaborn import load_dataset, pairplot, scatterplot
from sklearn.datasets import (
    fetch_openml,
    load_digits,
    load_iris,
    load_wine,
    make_circles,
    make_moons,
)
from sklearn.decomposition import PCA
from sklearn.utils import Bunch
from sklearn.utils import shuffle as sk_shuffle
from sklearn.utils.multiclass import unique_labels

from fkdc import config
from fkdc.utils import anteojos, muestra

T = TypeVar("T")

datasets_sinteticos = [
    ## D=2, d=1, k=2, "ruido bajo"
    "lunas_lo",
    "circulos_lo",
    "espirales_lo",
    ### ibíd, "ruido alto"
    "lunas_hi",
    "circulos_hi",
    "espirales_hi",
    ## D=3, d=1, k=2, 0 dims ruido
    "eslabones_0",
    "helices_0",
    ## D=3, d=2, k=2
    "hueveras_0",
    ## D=3, d=2, k=4
    "pionono_0",
    ### ibíd, 12 dims ruido
    "eslabones_12",
    "helices_12",
    "hueveras_12",
    "pionono_12",
    # OG es real, esto es una reducción PCA(n=~90) que captura >90% varianza
    "mnist",
]
datasets_reales = [
    ## k=3, d=?
    ### D=2
    "anteojos",
    ### D=4
    "iris",
    "pinguinos",
    ## D=11, k=?
    "vino",
    ## D "grande", k=10
    "digitos",
]
datasets = [*datasets_sinteticos, *datasets_reales]


def _dos_muestras(n_samples, random_state=None, shuffle=False):
    """Genera etiquetas binarias para dos muestras equilibradas."""
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
    a=1, n_samples=200, vueltas=2, noise=None, random_state=None, shuffle=False
):
    """Genera dos espirales entrelazadas en 2D."""
    rng = np.random.default_rng(random_state)
    y = _dos_muestras(n_samples, rng, shuffle)
    phi = stats.uniform(0, 2 * np.pi * vueltas).rvs(len(y), random_state=rng)
    X = np.vstack([a * np.sqrt(phi) * np.cos(phi), a * np.sqrt(phi) * np.sin(phi)]).T
    if noise:
        X += rng.normal(scale=noise, size=X.shape)
    X[y == 1] *= -1
    return X, y


def hacer_anteojos(
    n_samples: int | tuple[int, int, int] = 400,
    separacion: float = 3,
    noise=None,
    random_state=None,
    shuffle=False,
    **kw_ojo,
):
    """Genera dataset de anteojos: marco + dos pupilas."""
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
    kw_ojo["n"] = n_anteojos
    kw_ojo["separacion"] = separacion
    X_anteojos = anteojos(**kw_ojo, random_state=rng)
    if noise:
        X_anteojos += rng.normal(scale=noise, size=X_anteojos.shape)
    X_ojos = np.vstack(
        [
            stats.norm(separacion / 2, 1 / 6).rvs(n_ojos, random_state=rng),
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
    """Genera dos hélices entrelazadas en 3D."""
    rng = np.random.default_rng(random_state)
    y = _dos_muestras(n_samples, rng, shuffle)
    semillas = rng.uniform(size=len(y)) * vueltas * 2 * np.pi
    X_z = semillas * altura / (2 * np.pi)
    X_x = centro[0] + radio * np.cos(semillas + np.pi * y)  # `np * y` rota clase 1 180°
    X_y = centro[1] + radio * np.sin(semillas + np.pi * y)  # sobre el eje Z
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
    """Genera dos eslabones (anillos) enlazados en 3D."""
    rng = np.random.default_rng(random_state)
    y = _dos_muestras(n_samples, rng, shuffle)
    semillas = rng.uniform(size=len(y)) * 2 * np.pi
    X_x = centro[0] + radio * np.cos(semillas)
    X_y = centro[1] + radio * np.sin(semillas)
    X_z = np.zeros_like(semillas)
    X = np.vstack([X_x, X_y, X_z]).T
    mascara = y == 1  # máscara para observaciones del eslabón 1
    X[mascara, 0] += radio  # traslada `1 * radio` el eje x, como el logo de cierta TC
    X[mascara, 1], X[mascara, 2] = (
        X[mascara, 2],
        X[mascara, 1],
    )  # roto 90° sobre el eje X
    if noise:
        X += rng.normal(scale=noise, size=X.shape)
    return X, y


def hacer_hueveras(
    n_samples=200, limites=(10, 10), noise=None, random_state=None, shuffle=False
):
    """Genera dos superficies sinusoidales opuestas en 3D."""
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
    """Genera un pionono (swiss roll) con 4 clases en 3D."""
    # TODO: Crédito a Sapienza & lle.py del libro con swissroll original
    assert (len(esquinas) == 2) and all(isinstance(n, Number) for n in esquinas)
    rng = np.random.default_rng(random_state)
    if isinstance(n_samples, int):  # 1/4 por clase
        base, resto = n_samples // 4, n_samples % 4
        ns_clase = base + np.array([i < resto for i in range(4)], np.int32)
    elif isinstance(n_samples, tuple) and len(n_samples) == 4:
        assert all(isinstance(n, int) for n in n_samples)
        ns_clase = n_samples
        n_samples = sum(n_samples)
    else:
        raise ValueError("`n_samples` debe ser un entero o una 4-tupla de enteros")
    y = np.concatenate([np.ones(n_cls, int) * i for i, n_cls in enumerate(ns_clase)])
    if shuffle:
        y = sk_shuffle(y, random_state=rng)
    centros = [(x, y) for x in esquinas for y in esquinas]
    semillas = rng.normal(
        scale=(esquinas[1] - esquinas[0]) / desvios, size=(n_samples, 2)
    )
    for i in range(4):
        semillas[y == i] += centros[i]
    t = 2 * np.pi * (1 + 2 * semillas[:, 0])
    X = np.vstack([t * np.cos(t), 21 * semillas[:, 1], t * np.sin(t)]).T
    if noise:
        X += rng.normal(scale=noise, size=X.shape)
    return X, y


def agregar_dims_ruido(X, ndims=None, scale=None, random_state=None):
    """Agrega dimensiones de ruido gaussiano a un dataset."""
    rng = np.random.default_rng(random_state)
    if scale is None:
        scale = np.std(X)
    elif isinstance(scale, callable):
        scale = scale(X)
    assert isinstance(scale, Number), "`scale` debe ser None, un callable o un escalar"
    ndims = ndims or (3 * X.shape[1])
    ruido = rng.normal(scale=scale, size=(X.shape[0], ndims))
    return np.hstack([X, ruido])


class ConjuntoDatos:
    """Conjunto de datos con etiquetas, dimensiones y métodos de visualización."""

    def __init__(self, X, y, nombre=None):
        self.X = X
        self.y = y.astype(str)
        self.nombre = nombre
        self.n, self.p = X.shape
        self.etiquetas = unique_labels(self.y)
        self.k = len(self.etiquetas)

    @staticmethod
    def de_fabrica(fabrica, ruido=None, nombre=None, **params_fabrica):
        """Crea un ConjuntoDatos a partir de una función de fábrica."""
        params = Bunch(**(params_fabrica or {}))
        X, y = fabrica(**params)
        if ruido:
            X = agregar_dims_ruido(X, **({} if ruido is True else ruido))
        params.factory = fabrica.__name__
        params.ruido = ruido
        ds = ConjuntoDatos(X, y, nombre)
        ds.params = params
        return ds

    def __str__(self):
        return f"ConjuntoDatos('{self.nombre}', n={self.n}, p={self.p}, k={self.k})"

    def __hash__(self):
        return hash(tuple(tuple(row) for row in self.X)) * hash(tuple(self.y)) % 2**64

    def dispersar(self, x=0, y=1, ax=None, **plot_kws):
        """Gráfico de dispersión 2D."""
        scatterplot(x=self.X[:, x], y=self.X[:, y], hue=self.y, ax=ax, **plot_kws)

    def dispersar_3d(self, x=0, y=1, z=2, ax=None, **plot_kws):
        """Gráfico de dispersión 3D."""
        if self.p < 3:
            raise ValueError(f"{self.nombre} tiene solo {self.p} dimensiones")
        ax = ax or plt.gcf().add_subplot(projection="3d")
        assert ax.name == "3d", "El eje (`ax`) debe tener proyección 3d"
        for i, lbl in enumerate(self.etiquetas):
            X_lbl = self.X[self.y == lbl]
            X, Y, Z = X_lbl[:, x], X_lbl[:, y], X_lbl[:, z]
            ax.scatter(X, Y, Z, c=f"C{i}", label=str(lbl), **plot_kws)
        ax.legend(title="Clase")

    def grafico_pares(self, dims: list[int] | None = None, **plot_kws):
        """Gráfico de pares (pairplot) de las dimensiones seleccionadas."""
        if dims is None:
            dims = list(range(self.p))
        datos = pd.DataFrame(self.X[:, dims])
        datos["y"] = self.y
        return pairplot(data=datos, hue="y", **plot_kws)

    def guardar(self, archivo: Path | None = None):
        """Guarda el conjunto de datos como pickle."""
        with open(archivo or f"{hash(self)}.pkl", "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def cargar(cls: T, path: Path) -> T:
        """Carga un conjunto de datos desde un archivo pickle."""
        return pickle.load(open(path, "rb"))


def hacer_datasets(
    n_muestras: int = config.n_muestras,
    semilla_principal: int | None = config.semilla_principal,
    repeticiones: int = config.repeticiones,
    dir_datos: Path | None = None,
):
    """Genera todos los datasets y los guarda como pickles."""
    dir_datos = dir_datos or Path.cwd() / "datasets"
    dir_datos.mkdir(parents=True, exist_ok=True)
    np.random.default_rng(semilla_principal)
    # Semillas grandes devuelven error
    semilla_principal = semilla_principal or (hash(dt.datetime.now()) % 2**32)
    semillas = config._obtener_semillas(semilla_principal, repeticiones)
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    logger.info("Instanciando datasets 2D")
    config_2d = Bunch(
        lunas=Bunch(factory=make_moons, noise_levels=Bunch(lo=0.25, hi=0.5)),
        circulos=Bunch(factory=make_circles, noise_levels=Bunch(lo=0.08, hi=0.2)),
        espirales=Bunch(factory=hacer_espirales, noise_levels=Bunch(lo=0.1, hi=0.2)),
    )
    datasets_2d = {
        (f"{nombre}_{nivel_ruido}", semilla): ConjuntoDatos.de_fabrica(
            cfg.factory, n_samples=n_muestras, noise=ruido, random_state=semilla
        )
        for nombre, cfg in config_2d.items()
        for nivel_ruido, ruido in cfg.noise_levels.items()
        for semilla in semillas
    }

    logger.info("Instanciando datasets Multi-K")
    X_pinguinos = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
    y_pinguinos = "species"
    pinguinos = load_dataset("penguins")[X_pinguinos + [y_pinguinos]].dropna()
    datasets_multik = {
        "anteojos": ConjuntoDatos.de_fabrica(
            hacer_anteojos,
            n_samples=n_muestras,
            noise=0.1,
            random_state=semilla_principal,
        ),
        "iris": ConjuntoDatos.de_fabrica(load_iris, return_X_y=True),
        "vino": ConjuntoDatos.de_fabrica(load_wine, return_X_y=True),
        "pinguinos": ConjuntoDatos(
            pinguinos[X_pinguinos].values, pinguinos.species.values
        ),
        "digitos": ConjuntoDatos.de_fabrica(load_digits, return_X_y=True),
    }

    logger.info("Instanciando datasets 3D")
    config_3d = Bunch(
        eslabones=Bunch(factory=hacer_eslabones, noise=0.15),
        helices=Bunch(factory=hacer_helices, noise=0.05),
        hueveras=Bunch(factory=hacer_hueveras, noise=0.05),
        pionono=Bunch(factory=hacer_pionono, noise=0.5),
    )
    datasets_3d = {
        (f"{nombre}_{ndims}", semilla): ConjuntoDatos.de_fabrica(
            cfg.factory,
            n_samples=n_muestras,
            noise=cfg.noise,
            random_state=semilla,
            ruido=Bunch(random_state=semilla_principal, ndims=ndims)
            if ndims
            else False,
        )
        for nombre, cfg in config_3d.items()
        for ndims in [0, 12]
        for semilla in semillas
    }

    logger.info("Instanciando datasets MNIST")
    n_componentes = 96
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    pca = PCA(n_componentes).fit(X)
    _X = pca.transform(X)
    datasets_mnist = {
        ("mnist", semilla): ConjuntoDatos(
            *muestra(_X, y, n_muestras=n_muestras, random_state=semilla)
        )
        for semilla in semillas
    }
    logger.info("Guardando datasets")
    for clave, dataset in {
        **datasets_2d,
        **datasets_3d,
        **datasets_multik,
        **datasets_mnist,
    }.items():
        clave = "-".join(map(str, clave)) if isinstance(clave, tuple) else clave
        pickle.dump(dataset, open(dir_datos / (f"{clave}.pkl"), "wb"))


if __name__ == "__main__":
    typer.run(hacer_datasets)
