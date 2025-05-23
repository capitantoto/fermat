import logging
import pickle
import time
from math import floor
from pathlib import Path
from warnings import simplefilter

import pandas as pd
import typer
import yaml
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

from fkdc import config
from fkdc.datasets import Dataset
from fkdc.tarea import Tarea

simplefilter("ignore", category=ConvergenceWarning)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.info("Logging inicializado")


def main(
    config_file: Path = Path("config.yaml"),
    workdir: Path = Path.cwd(),
):
    workdir.mkdir(parents=True, exist_ok=True)
    logger.info("Leyendo config %s", config_file)
    cfg = yaml.safe_load(open(config_file, "r"))
    dataset = Dataset.cargar(cfg["dataset"])
    clf = cfg["clasificador"]
    clasificador = config.clasificadores.get(clf)
    if clasificador is None:
        try:
            clasificador = open(Path(clf), "rb")
        except Exception as exc:
            logger.error("Error leyendo Clasificador de %s", clf, exc_info=True)
            raise exc
    cv = cfg.get("cv", config.cv)
    busqueda_params = cfg.get(
        "busqueda_params",
        dict(refit=True, return_train_score=True, cv=cv, n_jobs=1),
    )
    busqueda_params.setdefault("scoring", cfg.get("scoring", "accuracy"))
    logger.info("Corrida de prueba")
    fold_runtime = time.time()
    split = cfg.get("split_evaluacion", config.split_evaluacion)
    seed = cfg.get("seed", config.main_seed)
    test_clf = clone(clasificador)
    X_train, X_eval, y_train, _ = train_test_split(
        dataset.X, dataset.y, test_size=(cv - 1) / cv, random_state=seed
    )
    train_time = time.time()
    test_clf.fit(X_train, y_train)
    logger.info("- Entrenamiento tomó %.2f segundos", time.time() - train_time)
    eval_time = time.time()
    test_clf.predict(X_eval)
    logger.info("- Evaluación tomó %.2f segundos", time.time() - eval_time)
    fold_runtime = time.time() - fold_runtime
    max_runtime = cfg.get("max_runtime", config.max_runtime)
    grilla_hipers = cfg.get("grilla_hipers", {})
    grid_iter = 1
    for values in grilla_hipers.values():
        grid_iter *= len(values)
    max_iter = int(floor(max_runtime / (fold_runtime * cv)))
    logger.info(
        "Fin Corrida de prueba. Tomó %.3f s, tiempo máximo de Tarea seteado en %.3f s.",
        fold_runtime,
        max_runtime,
    )
    if max_iter == 0:
        logger.warning(
            "La corrida de prubea tardó más que el tiempo total permitido. Abortando"
        )
        return
    elif grid_iter <= max_iter:
        logger.info(
            "Se realizará una búsqueda exhaustiva en %i hiperparametrizaciones",
            grid_iter,
        )
        busqueda_factory = GridSearchCV
    else:
        logger.info(
            "Se realizará una búsqueda aleatoria en %i hiperparametrizaciones", max_iter
        )
        busqueda_factory = RandomizedSearchCV
        busqueda_params["n_iter"] = max_iter
    task_time = time.time()
    tarea = Tarea(
        dataset,
        {clf: (clasificador, grilla_hipers)},
        busqueda_factory=busqueda_factory,
        busqueda_params=busqueda_params,
        split_evaluacion=split,
        seed=seed,
    )
    logger.info("Entrenamiento principal")
    task_time = time.time()
    tarea.evaluar()
    logger.info("- Tomó %.2fs", time.time() - task_time)
    if cfg.get("guardar_tarea", False):
        tarea.guardar(workdir / f"tarea-{config_file.stem}.pkl")
    logger.info("Resumen resultados")
    campos = ["logvero", "r2", "accuracy", "t_entrenar", "t_evaluar"]
    logger.info("\n%s" % pd.DataFrame(tarea.info).T[campos].to_markdown())
    pickle.dump(tarea.info, open(workdir / f"{config_file.stem}.pkl", "wb"))


if __name__ == "__main__":
    typer.run(main)
