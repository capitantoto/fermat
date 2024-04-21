import numpy as np
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import Bunch

from fkdc.fermat import BaseKDEClassifier, FermatKDEClassifier


class Algoritmo:
    preprocesador_base = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA()),
        ]
    )
    espacio_pre = {
        "scaler": [StandardScaler(), "passthrough"],
        "pca": [*[PCA(ratio) for ratio in [0.8, 0.9, 0.95]], "passthrough"],
    }

    def __init__(self, nombre, clf, espacio_clf=None, pre=None, espacio_pre=None):
        self.nombre = nombre
        self.clf = clf
        self.pre = pre or self.preprocesador_base
        self.espacio = {
            **{"pre__%s" % k: v for k, v in (espacio_pre or {}).items()},
            **{"clf__%s" % k: v for k, v in (espacio_clf or {}).items()},
        }
        self.pipe = Pipeline([("pre", self.pre), ("clf", self.clf)])


# %%
clasificadores = Bunch(
    fkde=FermatKDEClassifier(),
    kde=BaseKDEClassifier(),
    gnb=GaussianNB(),
    kn=KNeighborsClassifier(),
    lgbm=LGBMClassifier(),
    lr=LogisticRegression(solver="saga"),
    svc=SVC(),
)
# max_neighbors = np.floor(min(Ni) * (1 - inner_test_size)).astype(int)


espacios_clf = {
    "kde": {"bandwidth": np.logspace(-2, 6, 101)},
    "fkde": (
        {
            "alpha": np.linspace(1, 2.5, 17),
            "bandwidth": np.logspace(-2, 6, 61),
        }
    ),
    "gnb": {"var_smoothing": np.logspace(-9, -1, 17)},
    "kn": {
        "n_neighbors": np.linspace(1, 200, dtype=int),
        "weights": ["uniform", "distance"],
    },
    "svc": {
        "C": np.logspace(-4, 3, 21),
        "gamma": np.logspace(-4, 2, 21),
        "kernel": [
            "linear",
            "rbf",
            "sigmoid",
        ],  # excluding 'poly' which sometimes takes forever to finish
    },
    "lr": {
        "C": np.logspace(-4, 4, 20),
        "penalty": ["elasticnet"],
        "l1_ratio": np.linspace(0, 1, 20),
    },
    "lgbm": {
        "n_estimators": [50, 200, 500],
        "learning_rate": np.logspace(-4, -2, 5),
        "num_leaves": [3, 5, 8],
        "reg_alpha": np.logspace(-3, -1, 3),
        "reg_lambda": np.logspace(-3, -1, 3),
        "colsample_bytree": [1 / 3, 1 / 2],
    },
}

algoritmos = Bunch(
    **{
        nombre: Algoritmo(nombre, clf, espacio_clf=espacios_clf.get(nombre))
        for nombre, clf in clasificadores.items()
    }
)
