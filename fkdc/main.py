"""
Principal script ejecutor de las Tareas de la tesis
En cuerpo:
1. Defino Datasets participantes
2. Defino algoritmos participantes
3. Creo Tareas
En main:
4. Ejecuto y evaluo.
"""

import datetime as dt
import logging
import pickle
from time import time

import numpy as np
import pandas as pd
from seaborn import load_dataset as sns_load_dataset
from sklearn.datasets import (
    fetch_openml,
    load_digits,
    load_iris,
    load_wine,
    make_circles,
    make_moons,
)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import Bunch

from fkdc.datasets import (
    Dataset,
    hacer_anteojos,
    hacer_eslabones,
    hacer_espirales,
    hacer_helices,
    hacer_hueveras,
    hacer_pionono,
)
from fkdc.fermat import FermatKNeighborsClassifier, KDClassifier
from fkdc.tarea import Tarea
from fkdc.utils import MAX_SEED, sample

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

main_seed = hash(dt.datetime.now()) % MAX_SEED
rng = np.random.default_rng(main_seed)
repetitions = 16
run_seeds = rng.integers(1000, 10000, size=repetitions)
n_samples = 800

# %%

espacio_kn = {
    "n_neighbors": np.unique(np.logspace(0, np.log10(300), num=15, dtype=int)),
    "weights": ["uniform", "distance"],
}
clasificadores = Bunch(
    fkdc=(
        KDClassifier(metric="fermat"),
        {
            "alpha": np.linspace(1, 2.5, 9),
            "bandwidth": np.logspace(-3, 2, 21),
        },
    ),
    kdc=(KDClassifier(metric="euclidean"), {"bandwidth": np.logspace(-3, 5, 101)}),
    gnb=(GaussianNB(), {"var_smoothing": np.logspace(-10, -2, 17)}),
    kn=(KNeighborsClassifier(), espacio_kn),
    fkn=(FermatKNeighborsClassifier(), espacio_kn),
    lr=(
        LogisticRegression(solver="saga", penalty="elasticnet", max_iter=200),
        {"C": np.logspace(-2, 2, 11), "l1_ratio": [0, 0.5, 1]},
    ),
    svc=(SVC(), {"C": np.logspace(-3, 5, 51), "gamma": ["scale", "auto"]}),
    lsvc=(
        LinearSVC(dual="auto"),
        {"C": np.logspace(-3, 3, 11), "fit_intercept": [True, False]},
    ),
)


config_2d = Bunch(
    lunas=Bunch(factory=make_moons, noise_levels=Bunch(lo=0.25, hi=0.5)),
    circulos=Bunch(factory=make_circles, noise_levels=Bunch(lo=0.08, hi=0.2)),
    espirales=Bunch(factory=hacer_espirales, noise_levels=Bunch(lo=0.1, hi=0.2)),
)
datasets_2d = {
    (nombre, seed, noise_lvl): Dataset.de_fabrica(
        cfg.factory, n_samples=n_samples, noise=noise, random_state=seed
    )
    for nombre, cfg in config_2d.items()
    for noise_lvl, noise in cfg.noise_levels.items()
    for seed in run_seeds
}
pinguinos = sns_load_dataset("penguins")
X_pinguinos = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
datasets_multik = {
    "anteojos": Dataset.de_fabrica(
        hacer_anteojos, n_samples=n_samples, noise=0.1, random_state=main_seed
    ),
    "iris": Dataset.de_fabrica(load_iris, return_X_y=True),
    "vino": Dataset.de_fabrica(load_wine, return_X_y=True),
    "pinguinos": Dataset(pinguinos[X_pinguinos].values, pinguinos.species.values),
    "digitos": Dataset.de_fabrica(load_digits, return_X_y=True),
}
ruido_ndims = [0, 12]
config_3d = Bunch(
    eslabones=Bunch(factory=hacer_eslabones, noise=0.15),
    helices=Bunch(factory=hacer_helices, noise=0.05),
    hueveras=Bunch(factory=hacer_hueveras, noise=0.05),
    pionono=Bunch(factory=hacer_pionono, noise=0.5),
)
datasets_3d = {
    (nombre, ndims): Dataset.de_fabrica(
        cfg.factory,
        n_samples=n_samples,
        noise=cfg.noise,
        random_state=main_seed,
        ruido=Bunch(random_state=main_seed, ndims=ndims) if ndims else False,
    )
    for nombre, cfg in config_3d.items()
    for ndims in ruido_ndims
}
p_mnist, n_mnist = 96, 4000  # 1.5x dims, ~2x sample size versus digits
X_mnist, y_mnist = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
mnist_pca = PCA(p_mnist).fit(X_mnist)
X_mnist_pca = mnist_pca.transform(X_mnist)
datasets_mnist = {
    ("mnist", seed): Dataset(
        *sample(X_mnist_pca, y_mnist, n_samples=n_mnist, random_state=seed)
    )
    for seed in run_seeds
}

split_tareas = 0.5

tareas_ds_fijo = {
    (nombre, seed): Tarea(ds, clasificadores, split_evaluacion=split_tareas, seed=seed)
    for nombre, ds in {**datasets_multik, **datasets_3d}.items()
    for seed in run_seeds
}
tareas_ds_variable = {
    nombre: Tarea(ds, clasificadores, split_evaluacion=split_tareas, seed=main_seed)
    for nombre, ds in {**datasets_2d, **datasets_mnist}.items()
}
if __name__ == "__main__":
    from pathlib import Path

    dir = Path(f"run-{main_seed}")
    dir.mkdir(parents=True, exist_ok=True)
    pickle.dump(run_seeds, open(dir / f"{main_seed}-run_seeds.pkl", "wb"))
    for nombre, tarea in {**tareas_ds_fijo, **tareas_ds_variable}.items():
        logger.info("Entrenando y evaluando %s", nombre)
        t0 = time()
        tarea.evaluar()
        logger.info("Tomó %f.2 segundos" % (time() - t0))
        tarea.dataset.guardar(dir / f"dataset-{nombre}.pkl")
        tarea.guardar(dir / f"tarea-{nombre}.pkl")
        logger.info("Resumen resultados")
        campos = ["logvero", "r2", "accuracy", "t_entrenar", "t_evaluar"]
        logger.info(pd.DataFrame(tarea.info).T[campos].to_markdown())
        pickle.dump(tarea.info, open(dir / f"info-{nombre}.pkl", "wb"))
