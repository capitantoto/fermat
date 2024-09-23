import logging
import pickle
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils import Bunch

from fkdc.datasets import datasets

logger = logging.basicConfig()


def _renormalizar_probas(probas, eps=1e-100):
    X = probas + eps
    return (X.T / X.sum(axis=1)).T


class Tarea:
    def __init__(
        self,
        dataset,
        algoritmos,
        busqueda_factory=GridSearchCV,
        busqueda_params=Bunch(
            refit=True, return_train_score=True, cv=5, scoring="accuracy"
        ),
        split_evaluacion=0.8,
        seed=None,
    ):
        self.dataset = ds = (
            pickle.load(open(dataset, "rb")) if isinstance(dataset, str) else dataset
        )
        if isinstance(algoritmos, list):
            self.algoritmos = {
                clf.__name__: Bunch(clf=clf, espacio=espacio)
                for (clf, espacio) in algoritmos
            }
        elif isinstance(algoritmos, dict):
            self.algoritmos = {
                nombre: Bunch(clf=clf, espacio=espacio)
                for nombre, (clf, espacio) in algoritmos.items()
            }
        else:
            raise ValueError(
                "`algoritmos` debe ser una lista o un dict de 2-tuplas (clf, espacio)"
            )
        self.busqueda_factory = busqueda_factory
        self.busqueda_params = busqueda_params or {}
        self.seed = seed or np.random.randint(0, int(1e10))
        self.split_evaluacion = split_evaluacion
        self._fitted = False

        self.X_train, self.X_eval, self.y_train, self.y_eval = train_test_split(
            ds.X, ds.y, test_size=self.split_evaluacion, random_state=self.seed
        )
        self.info = Bunch()

    def __str__(self):
        return f"Tarea({self.dataset.nombre}, [{', '.join(self.algoritmos)}])"

    def entrenar(self):
        self.clasificadores = Bunch()
        for nombre, algo in self.algoritmos.items():
            self.info[nombre] = info = Bunch()
            logger.info(f"Entrenar {nombre} en {self.dataset.nombre}")
            logger.info(f"Espacio de búsqueda: {algo.espacio}")
            busqueda = self.busqueda_factory(
                algo.clf, algo.espacio, **self.busqueda_params
            )
            t0 = time()
            busqueda.fit(self.X_train, self.y_train)
            info.t_entrenar = time() - t0
            self.clasificadores[nombre] = busqueda.best_estimator_
            info.busqueda = busqueda
        self._fitted = True

    def evaluar(self, verosimilitud=True, forzar_entrenamiento=True):
        if not self._fitted:
            if forzar_entrenamiento:
                self.entrenar()
            else:
                raise ValueError("Aún no se ha(n) ajustado el(los) modelo(s)")
        self.info.base = base = Bunch(
            probas=pd.Series(self.y_train).value_counts(normalize=True)
        )
        if verosimilitud:
            base.logvero = base.probas.loc[self.y_eval].apply("log").sum()
        for nombre, clf in self.clasificadores.items():
            logger.info(f"Evaluando {nombre}")
            info = self.info[nombre]
            if verosimilitud and hasattr(clf, "predict_proba"):
                info.probas = clf.predict_proba(self.X_eval)
                info.preds_proba = clf.classes_[info.probas.argmax(axis=1)]
                info.logvero = -log_loss(
                    _renormalizar_probas(info.probas),
                    self.y_eval,
                    normalize=False,
                    labels=self.dataset.labels,
                )
                info.r2 = 1 - info.logvero / base.logvero
            t0 = time()
            info.preds = clf.predict(self.X_eval)
            info.t_evaluar = time() - t0
            info.accuracy = accuracy_score(info.preds, self.y_eval)

    def guardar(self, path=None):
        if path is None:
            params = (self.dataset.nombre, self.seed, self.split_evaluacion)
            path = "%s-%s-%s.pkl" % params
        pickle.dump(self, open(path, "wb"))


tareas = [
    Tarea(dataset, [], split_evaluacion=0.5, seed=1991) for dataset in datasets.values()
]

if __name__ == "__main__":
    import pickle as pkl

    for tarea in tareas:
        tarea.entrenar()
        tarea.evaluar()

    resumenes = Bunch()

    for tarea in tareas:
        logger.info("==== %s ====" % tarea.dataset.nombre.upper())
        busquedas = tarea.info.pop("busquedas")
        train = (
            pd.concat(
                [
                    pd.DataFrame(busqueda.cv_results_)[
                        ["mean_test_score", "mean_train_score"]
                    ]
                    for busqueda in busquedas.values()
                ],
                keys=busquedas.keys(),
                names=["algo", "_drop"],
            )
            .reset_index()
            .drop(columns="_drop")
        )
        idx = train.groupby("algo").mean_test_score.idxmax()
        train_scores = train.loc[idx].set_index("algo")
        resumen = pd.concat(
            [
                pd.DataFrame({**tarea.info, "mean_eval_score": tarea.puntajes}),
                train_scores,
            ],
            axis=1,
        ).sort_values("mean_eval_score", ascending=False)
        logger.info(resumen)
    pkl.dump(resumenes, open("resumenes.pkl", "wb"))
    pkl.dump(tareas, open("tareas.pkl", "wb"))
