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
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import Bunch

from fkdc.fermat import BaseKDClassifier, FermatKDClassifier, FermatKNeighborsClassifier

# %%

espacio_kn = {
    "n_neighbors": np.unique(np.logspace(0, np.log10(200), num=12, dtype=int)),
    "weights": ["uniform", "distance"],
}
clasificadores = Bunch(
    fkdc=(
        FermatKDClassifier(),
        {
            "alpha": np.linspace(1, 2.5, 9),
            "bandwidth": np.logspace(-2, 2, 31),
        },
    ),
    kdc=(BaseKDClassifier(), {"bandwidth": np.logspace(-2, 6, 101)}),
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
