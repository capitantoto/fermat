import logging
from itertools import product
from pathlib import Path
from typing import List, Optional

import numpy as np
import typer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import Bunch
from typing_extensions import Annotated

from fkdc.fermat import FermatKNeighborsClassifier, KDClassifier
from fkdc.utils import yaml

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

espacio_kn = {
    "n_neighbors": np.unique(np.logspace(0, np.log10(300), num=15, dtype=int)),
    "weights": ["uniform", "distance"],
}
grillas = dict(
    fkdc={"alpha": np.linspace(1, 2.5, 9), "bandwidth": np.logspace(-3, 2, 21)},
    kdc={"bandwidth": np.logspace(-3, 5, 101)},
    gnb={"var_smoothing": np.logspace(-10, -2, 17)},
    kn=espacio_kn,
    fkn={**espacio_kn, "alpha": np.linspace(1, 2.5, 7)},
    lr={"C": np.logspace(-2, 2, 11), "l1_ratio": [0, 0.5, 1]},
    svc={"C": np.logspace(-3, 5, 51), "gamma": ["scale", "auto"]},
    lsvc={"C": np.logspace(-3, 3, 11), "fit_intercept": [True, False]},
)
clasificadores = Bunch(
    fkdc=KDClassifier(metric="fermat"),
    kdc=KDClassifier(metric="euclidean"),
    gnb=GaussianNB(),
    kn=KNeighborsClassifier(),
    fkn=FermatKNeighborsClassifier(),
    lr=LogisticRegression(
        solver="saga", penalty="elasticnet", max_iter=200, l1_ratio=0
    ),
    svc=SVC(),
    lsvc=LinearSVC(dual="auto"),
)
n_samples = 800
main_seed = 2024
max_runtime = 120
clf_lentos = (KDClassifier, FermatKNeighborsClassifier)
split_evaluacion = 0.5
cv = 5
repetitions = 16


def _get_run_seeds(main_seed=main_seed, repetitions=repetitions):
    rng = np.random.default_rng(main_seed)
    return sorted(rng.integers(1000, 10000, size=repetitions).tolist())


def make_configs(
    main_seed: int = main_seed,
    max_runtime: float = max_runtime,
    split_evaluacion: float = split_evaluacion,
    cv: int = cv,
    run_seeds: Optional[List[int]] = None,
    repetitions: int = repetitions,
    datasets_dir: Path = Path("datasets"),
    configs_dir: Path = Path("configs"),
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
):
    configs_dir.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.info("Generando configuraciones")

    datasets = {
        ds.stem: ds
        for ds in datasets_dir.glob("*.pkl")
        if any(
            nombre in ds.stem
            for nombre in ["anteojos", "vino", "pinguinos", "iris", "digitos"]
        )
    }
    for nombre_clf, clf in clasificadores.items():
        scores = ["accuracy"]
        if hasattr(clf, "predict_proba"):
            scores.append("neg_log_loss")
        grilla_hipers = {
            param: values if isinstance(values, list) else values.tolist()
            for param, values in grillas[nombre_clf].items()
        }
        for nombre_dataset, dataset in datasets.items():
            base_config = dict(
                dataset=str(dataset),
                clasificador=nombre_clf,
                grilla_hipers=grilla_hipers,
                cv=cv,
                split_evaluacion=split_evaluacion,
                max_runtime=max_runtime * (5 if isinstance(clf, clf_lentos) else 1),
            )
            partes = nombre_dataset.split("-")
            run_seeds = run_seeds or _get_run_seeds(main_seed, repetitions)
            if len(partes) == 2:
                nombre_dataset, semilla_dataset = partes
                semilla_dataset = int(semilla_dataset)
                task_seeds = [main_seed] if semilla_dataset in run_seeds else run_seeds
            elif len(partes) == 1:
                nombre_dataset = partes[0]
                semilla_dataset = None
                task_seeds = run_seeds
            else:
                raise ValueError("Nombre de dataset invalido: %s" % nombre_dataset)
            for task_seed, score in product(task_seeds, scores):
                config = dict(**base_config, seed=task_seed, scoring=score)
                clave = (nombre_dataset, semilla_dataset, nombre_clf, task_seed, score)
                nombre_config = "-".join(map(str, clave)) + ".yaml"
                logger.debug("Generando configuracion %s", nombre_config)
                yaml.dump(config, open(configs_dir / nombre_config, "w"))


if __name__ == "__main__":
    typer.run(make_configs)
