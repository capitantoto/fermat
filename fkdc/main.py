"""
Principal script ejecutor de las Tareas de la tesis
En cuerpo:
1. Defino Datasets participantes
2. Defino algoritmos participantes
3. Creo Tareas
En main:
4. Ejecuto y evaluo.
"""

import numpy as np
from seaborn import load_dataset as sns_load_dataset
from sklearn.datasets import (  # fetch_openml,
    load_digits,
    load_iris,
    load_wine,
    make_circles,
    make_moons,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import Bunch

from fkdc.datasets import (  # hacer_eslabones,; hacer_helices,; hacer_hueveras,; hacer_pionono,
    Dataset,
    hacer_anteojos,
    hacer_espirales,
)
from fkdc.fermat import FermatKNeighborsClassifier, KDClassifier

# %%

espacio_kn = {
    "n_neighbors": np.unique(np.logspace(0, np.log10(200), num=12, dtype=int)),
    "weights": ["uniform", "distance"],
}
clasificadores = Bunch(
    fkdc=(
        KDClassifier(metric="fermat"),
        {
            "alpha": np.linspace(1, 2.5, 9),
            "bandwidth": np.logspace(-2, 2, 31),
        },
    ),
    kdc=(KDClassifier(metric="euclidean"), {"bandwidth": np.logspace(-2, 6, 101)}),
    gnb=(GaussianNB(), {"var_smoothing": np.logspace(-10, -2, 17)}),
    kn=(KNeighborsClassifier(), espacio_kn),
    fkn=(FermatKNeighborsClassifier(), espacio_kn),
    lr=(
        LogisticRegression(solver="saga", penalty="elasticnet"),
        {"C": np.logspace(-2, 2, 11), "l1_ratio": np.linspace(0, 1, 3)},
    ),
    svc=(
        SVC(),
        {
            "C": np.logspace(-3, 3, 21),
            "gamma": ["scale", "auto", *np.logspace(-3, 2, 11)],
            "kernel": ["linear", "rbf", "sigmoid"],
        },
    ),
)


n_samples = 400
datasets = [
    ("iris", *load_iris(return_X_y=True)),
    ("lunas", *make_moons(n_samples=n_samples)),
    ("circulos", *make_circles(n_samples=n_samples)),
    ("espirales", *hacer_espirales(n_samples=n_samples)),
    # ("mnist", *fetch_openml("mnist_784", version=1, return_X_y=True)),
    ("vino", *load_wine(return_X_y=True)),
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
