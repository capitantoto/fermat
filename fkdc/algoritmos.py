import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import Bunch

from fkdc.fermat import BaseKDClassifier, FermatKDClassifier, FermatKNeighborsClassifier


class Algoritmo:
    preprocesador_base = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA()),
        ]
    )
    espacio_pre_base = {
        "scaler": [StandardScaler(), "passthrough"],
        "pca": [*[PCA(ratio) for ratio in [0.8, 0.9, 0.95]], "passthrough"],
    }

    def __init__(self, nombre, clf, espacio_clf=None, pre=None, espacio_pre=None):
        self.nombre = nombre
        self.clf = clf
        self.pre = self.preprocesador_base if pre is None else pre
        espacio_pre = self.espacio_pre_base if espacio_pre is None else espacio_pre
        self.espacio = {
            **{"pre__%s" % k: v for k, v in espacio_pre.items()},
            **{"clf__%s" % k: v for k, v in (espacio_clf or {}).items()},
        }
        self.pipe = Pipeline([("pre", self.pre), ("clf", self.clf)])


# %%
clasificadores = Bunch(
    fkdc=FermatKDClassifier(),
    kdc=BaseKDClassifier(),
    gnb=GaussianNB(),
    kn=KNeighborsClassifier(),
    fkn=FermatKNeighborsClassifier(),
    lr=LogisticRegression(solver="saga"),
    svc=SVC(),
)
# max_neighbors = np.floor(min(Ni) * (1 - inner_test_size)).astype(int)

espacio_kn = {
    "n_neighbors": np.unique(np.logspace(0, np.log10(200), num=12, dtype=int)),
    "weights": ["uniform", "distance"],
}
espacios_clf = Bunch(
    kdc={"bandwidth": np.logspace(-2, 6, 101)},
    fkdc=(
        {
            "alpha": np.linspace(1, 2.5, 9),
            "bandwidth": np.logspace(-2, 2, 31),
        }
    ),
    gnb={"var_smoothing": np.logspace(-10, -2, 17)},
    kn=espacio_kn,
    fkn=espacio_kn,
    svc={
        "C": np.logspace(-3, 3, 21),
        "gamma": ["scale", "auto", *np.logspace(-3, 2, 11)],
        "kernel": ["linear", "rbf", "sigmoid"],
        # excluding 'poly' kernel which sometimes takes forever to finish
    },
    lr={
        "C": np.logspace(-2, 2, 11),
        "penalty": ["elasticnet"],
        "l1_ratio": np.linspace(0, 1, 3),
    },
)

algoritmos = Bunch(
    **{
        nombre: Algoritmo(nombre, clf, espacio_clf=espacios_clf.get(nombre))
        for nombre, clf in clasificadores.items()
    }
)
