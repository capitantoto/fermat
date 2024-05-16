import pickle
from time import time

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils import Bunch

from fkdc.algoritmos import algoritmos
from fkdc.datasets import datasets


class Tarea:
    def __init__(
        self,
        dataset,
        algoritmos,
        busqueda_factory=GridSearchCV,
        busqueda_params=Bunch(refit=True, return_train_score=True),
        cv=5,
        split_evaluacion=0.8,
        scorer=accuracy_score,
        seed=42,
    ):
        self.dataset = ds = datasets[dataset] if isinstance(dataset, str) else dataset
        self.algoritmos = Bunch(**{algo.nombre: algo for algo in algoritmos})
        self.busqueda_factory = busqueda_factory
        self.busqueda_params = busqueda_params or {}
        self.busqueda_params.setdefault("cv", cv)
        self.cv = cv
        self.scorer = scorer
        self.seed = seed
        self.split_evaluacion = split_evaluacion
        self._fitted = False

        self.X_train, self.X_eval, self.y_train, self.y_eval = train_test_split(
            ds.X, ds.y, test_size=self.split_evaluacion, random_state=self.seed
        )
        self.debug = Bunch()

    def __str__(self):
        return f"Tarea({self.dataset.nombre}, [{', '.join(self.algoritmos)}])"

    def entrenar(self):
        self.clasificadores = Bunch()
        self.debug.t_entrenar = Bunch()
        self.debug.busquedas = Bunch()
        for nombre, algo in self.algoritmos.items():
            print(f"Entrenar {nombre} en {self.dataset.nombre}")
            print(f"Espacio de búsqueda: {algo.espacio}")
            self.debug.busquedas[nombre] = busqueda = self.busqueda_factory(
                algo.pipe, algo.espacio, **self.busqueda_params
            )
            t0 = time()
            busqueda.fit(self.X_train, self.y_train)
            self.debug.t_entrenar[nombre] = time() - t0
            self.clasificadores[nombre] = busqueda.best_estimator_
        self._fitted = True

    def evaluar(self, forzar_entrenamiento=True):
        if not self._fitted:
            if forzar_entrenamiento:
                self.entrenar()
            else:
                raise ValueError("No se ha(n) ajustado el(los) modelo(s) aún")

        self.debug.t_evaluar = Bunch()
        self.puntajes = Bunch()
        for nombre, clasificador in self.clasificadores.items():
            print(f"Evaluar {nombre} en {self.dataset.nombre}")
            t0 = time()
            self.puntajes[nombre] = self.scorer(
                clasificador.predict(self.X_eval), self.y_eval
            )
            self.debug.t_evaluar[nombre] = time() - t0

    def guardar(self, path=None):
        if path is None:
            params = (self.dataset.nombre, self.seed, self.split_evaluacion)
            path = "%s-%s-%s.pkl" % params
        pickle.dump(self, open(path, "wb"))


tareas = [
    Tarea(
        dataset,
        [algoritmos[algo] for algo in ["fkdc", "kdc", "gnb", "kn", "lr"]],
        split_evaluacion=0.5,
        seed=1991,
    )
    for dataset in datasets.values()
]

if __name__ == "__main__":
    import pickle as pkl

    for tarea in tareas:
        tarea.entrenar()
        tarea.evaluar()

    resumenes = Bunch()

    for tarea in tareas:
        print("==== %s ====" % tarea.dataset.nombre.upper())
        busquedas = tarea.debug.pop("busquedas")
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
                pd.DataFrame({**tarea.debug, "mean_eval_score": tarea.puntajes}),
                train_scores,
            ],
            axis=1,
        ).sort_values("mean_eval_score", ascending=False)
        print(resumen)
    pkl.dump(resumenes, open("resumenes.pkl", "wb"))
    pkl.dump(tareas, open("tareas.pkl", "wb"))
