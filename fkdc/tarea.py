import logging
import pickle
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils import Bunch


logger = logging.getLogger(__name__)
# Read warnings, and log them at the debug level
warnings_logger = logging.getLogger('py.warnings')
warnings_logger.setLevel(logging.DEBUG)
logging.captureWarnings(capture=True)

class Tarea:
    def __init__(
        self,
        dataset,
        algoritmos,
        scoring="accuracy",
        busqueda_factory=GridSearchCV,
        busqueda_params=Bunch(refit=True, return_train_score=True, cv=5),
        split_evaluacion=0.2,
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
        # Si `scoring` esta explícitamente seteado en busqueda_params, respétese.
        self.busqueda_params.setdefault("scoring", scoring)
        self.split_evaluacion = split_evaluacion
        self.seed = seed or np.random.randint(0, 2**32)
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
            logger.info("Entrenando %s", nombre)
            self.info[nombre] = info = Bunch()
            if "n_neighbors" in algo.espacio:  # Evita n_neighbors > train size
                _, conteos = np.unique(self.y_train, return_counts=True)
                algo.espacio["n_neighbors"] = [
                    n for n in algo.espacio["n_neighbors"] if n <= min(conteos)
                ]
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
        base.accuracy = (self.y_eval == base.probas.idxmax()).mean()
        if verosimilitud:
            conteos = pd.Series(self.y_eval).value_counts()
            base.logvero = (np.log(base.probas) * conteos).sum()
            base.r2 = 0
        for nombre, clf in self.clasificadores.items():
            logger.info("Evaluando %s", nombre)
            try:
                info = self.info[nombre]
                if verosimilitud and hasattr(clf, "predict_proba"):
                    info.probas = clf.predict_proba(self.X_eval)
                    info.preds_proba = clf.classes_[info.probas.argmax(axis=1)]
                    info.logvero = -log_loss(
                        self.y_eval,
                        info.probas,
                        normalize=False,
                        labels=self.dataset.labels,
                    )
                    info.r2 = 1 - info.logvero / base.logvero
                t0 = time()
                info.preds = clf.predict(self.X_eval)
                info.t_evaluar = time() - t0
                info.accuracy = accuracy_score(info.preds, self.y_eval)
            except Exception as exc:
                logger.warning(exc, exc_info=True)

    def guardar(self, path=None):
        if path is None:
            params = (self.dataset.nombre, self.seed, self.split_evaluacion)
            path = "%s-%s-%s.pkl" % params
        pickle.dump(self, open(path, "wb"))
