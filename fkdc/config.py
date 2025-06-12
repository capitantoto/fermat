import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
import typer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import Bunch
from typing_extensions import Annotated

from fkdc.datasets import Dataset
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
grillas = dict(
    fkdc={"alpha": np.linspace(1, 4, 13), "bandwidth": np.logspace(-3, 6, 37)},
    kdc={"bandwidth": np.logspace(-3, 6, 118)},
    gnb={"var_smoothing": np.logspace(-12, 1, 40)},
    kn=espacio_kn,
    fkn={**espacio_kn, "alpha": np.linspace(1, 4, 13)},
    lr={"C": np.logspace(-5, 2, 36)},
    slr={"C": np.logspace(-5, 2, 36)},
    svc={"C": np.logspace(-4, 6, 61), "gamma": ["scale", "auto"]},
    gbt={"learning_rate": [0.025, 0.05, 0.1], "max_depth": [3, 5, 8, 13]},
)
clasificadores = Bunch(
    fkdc=KDClassifier(metric="fermat"),
    kdc=KDClassifier(metric="euclidean"),
    gnb=GaussianNB(),
    kn=KNeighborsClassifier(),
    fkn=FermatKNeighborsClassifier(),
    lr=LogisticRegression(max_iter=50_000),
    slr=Pipeline([("scaler", StandardScaler()), ("logreg", LogisticRegression(max_iter=50_000))]),
    svc=SVC(),
    gbt=HistGradientBoostingClassifier(max_features=0.5),
)
n_samples = 800
main_seed = 1312
split_evaluacion = 0.5
cv = 5
scoring="neg_log_loss"
repetitions = 16


def _get_run_seeds(main_seed=main_seed, repetitions=repetitions):
    rng = np.random.default_rng(main_seed)
    return sorted(rng.integers(1000, 10000, size=repetitions).tolist())


def make_configs(
    main_seed: int = main_seed,
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

    datasets = {ds.stem: ds for ds in datasets_dir.glob("*.pkl")}
    for nombre_clf, clf in clasificadores.items():
        grilla_hipers = {
            param: values if isinstance(values, list) else values.tolist()
            for param, values in grillas[nombre_clf].items()
        }
        for nombre_dataset, dataset_path in datasets.items():
            ds = Dataset.cargar(dataset_path)
            if n_neighbors := grilla_hipers.get("n_neighbors"):
                grilla_hipers["n_neighbors"] = [n for n in n_neighbors if n <= ds.n * (cv - 1) / cv]
            base_config = dict(
                dataset=str(dataset_path),
                clasificador=nombre_clf,
                grilla_hipers=grilla_hipers,
                cv=cv,
                split_evaluacion=split_evaluacion,
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
            for task_seed in task_seeds:
                scoring = (
                    "neg_log_loss" if hasattr(clf, "predict_proba") else "accuracy"
                )
                config = dict(**base_config, seed=task_seed, scoring=scoring)
                clave = (
                    nombre_dataset,
                    semilla_dataset,
                    nombre_clf,
                    task_seed,
                    scoring,
                )
                nombre_config = "-".join(map(str, clave)) + ".yaml"
                logger.debug("Generando configuracion %s", nombre_config)
                yaml.dump(config, open(configs_dir / nombre_config, "w"))


if __name__ == "__main__":
    typer.run(make_configs)
