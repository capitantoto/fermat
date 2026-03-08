import logging
from math import floor
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils import Bunch

from fkdc import dir_raiz
from fkdc.fermat import FermatKNeighborsClassifier, KDClassifier
from fkdc.utils import yaml

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

espacio_kn = {
    "n_neighbors": np.unique(np.logspace(0, np.log10(500), num=21, dtype=int)),
    "weights": ["uniform", "distance"],
}
grillas = {
    "fkdc": {"alpha": np.linspace(1, 4, 13), "bandwidth": np.logspace(-5, 6, 45)},
    "kdc": {"bandwidth": np.logspace(-5, 6, 136)},
    "gnb": {"var_smoothing": np.logspace(-12, 1, 40)},
    "kn": espacio_kn,
    "fkn": {**espacio_kn, "alpha": np.linspace(1, 4, 13)},
    "lr": {"C": np.logspace(-5, 2, 36)},
    "slr": {"logreg__C": np.logspace(-5, 2, 36)},
    "svc": {"C": np.logspace(-4, 6, 61), "gamma": ["scale", "auto"]},
    "gbt": {"learning_rate": [0.025, 0.05, 0.1], "max_depth": [3, 5, 8, 13]},
}
clasificadores = Bunch(
    fkdc=KDClassifier(metric="fermat"),
    kdc=KDClassifier(metric="euclidean"),
    gnb=GaussianNB(),
    kn=KNeighborsClassifier(),
    fkn=FermatKNeighborsClassifier(),
    lr=LogisticRegression(max_iter=50_000),
    slr=Pipeline(
        [("scaler", StandardScaler()), ("logreg", LogisticRegression(max_iter=50_000))]
    ),
    svc=SVC(),
    gbt=HistGradientBoostingClassifier(max_features=0.5),
)
n_muestras = 800
semilla_principal = 1312
split_evaluacion = 0.5
cv = 5
puntuacion = "neg_log_loss"
repeticiones = 25
dir_ejecucion = dir_raiz / "sandbox/v5/infos"


def _obtener_semillas(semilla_principal=semilla_principal, repeticiones=repeticiones):
    """Genera semillas reproducibles a partir de una semilla principal."""
    rng = np.random.default_rng(semilla_principal)
    return sorted(rng.integers(1000, 10000, size=repeticiones).tolist())


def hacer_configuraciones(
    semilla_principal: int = semilla_principal,
    split_evaluacion: float = split_evaluacion,
    cv: int = cv,
    semillas: list[int] | None = None,
    repeticiones: int = repeticiones,
    dir_datasets: Path = Path("datasets"),
    dir_configs: Path = Path("configs"),
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
):
    """Genera configs YAML para cada combinación dataset × clf."""
    dir_configs.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.info("Generando configuraciones")

    datasets = {ds.stem: ds for ds in dir_datasets.glob("*.pkl")}
    for nombre_clf, clf in clasificadores.items():
        grilla_base = {  # Evita usar np.array que serializa feo
            param: valores if isinstance(valores, list) else valores.tolist()
            for param, valores in grillas[nombre_clf].items()
        }
        for nombre_dataset, ruta_dataset in datasets.items():
            grilla_hipers = grilla_base.copy()
            if n_neighbors := grilla_hipers.get("n_neighbors"):
                ds = Dataset.cargar(ruta_dataset)
                n_muestras_ajuste = floor(ds.n * (cv - 1) / cv * (1 - split_evaluacion))
                n_neighbors = [n for n in n_neighbors if n < n_muestras_ajuste]
                grilla_hipers["n_neighbors"] = n_neighbors
            config_base = {
                "dataset": str(ruta_dataset),
                "clasificador": nombre_clf,
                "grilla_hipers": grilla_hipers,
                "cv": cv,
                "split_evaluacion": split_evaluacion,
            }
            partes = nombre_dataset.split("-")
            semillas = semillas or _obtener_semillas(semilla_principal, repeticiones)
            if len(partes) == 2:
                nombre_dataset, semilla_dataset = partes
                semilla_dataset = int(semilla_dataset)
                semillas_tarea = (
                    [semilla_principal] if semilla_dataset in semillas else semillas
                )
            elif len(partes) == 1:
                nombre_dataset = partes[0]
                semilla_dataset = None
                semillas_tarea = semillas
            else:
                raise ValueError(f"Nombre de dataset inválido: {nombre_dataset}")
            for semilla_tarea in semillas_tarea:
                puntuacion_tarea = (
                    "neg_log_loss" if hasattr(clf, "predict_proba") else "accuracy"
                )
                configuracion = dict(
                    **config_base, seed=semilla_tarea, scoring=puntuacion_tarea
                )
                clave = (
                    nombre_dataset,
                    semilla_dataset,
                    nombre_clf,
                    semilla_tarea,
                    puntuacion_tarea,
                )
                nombre_config = "-".join(map(str, clave)) + ".yaml"
                logger.debug("Generando configuración %s", nombre_config)
                yaml.dump(configuracion, open(dir_configs / nombre_config, "w"))


if __name__ == "__main__":
    from fkdc.datasets import Dataset

    typer.run(hacer_configuraciones)
