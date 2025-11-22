import logging
import pickle
import time
from pathlib import Path
from warnings import simplefilter

import pandas as pd
import typer
import yaml
from sklearn.exceptions import ConvergenceWarning

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
    cfg = yaml.safe_load(open(config_file))
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
    scoring = cfg.get("scoring", config.scoring)
    logger.info("Corrida de prueba")
    split = cfg.get("split_evaluacion", config.split_evaluacion)
    seed = cfg.get("seed", config.main_seed)
    grilla_hipers = cfg.get("grilla_hipers", {})
    task_time = time.time()
    tarea = Tarea(
        dataset,
        {clf: (clasificador, grilla_hipers)},
        cv=cv,
        split_evaluacion=split,
        seed=seed,
        scoring=scoring,
    )
    logger.info("Entrenamiento principal")
    task_time = time.time()
    tarea.evaluar()
    logger.info("- Tomó %.2fs", time.time() - task_time)
    if cfg.get("guardar_tarea", False):
        tarea.guardar(workdir / f"tarea-{config_file.stem}.pkl")
    logger.info("Resumen resultados")
    campos = ["logvero", "r2", "accuracy", "t_entrenar", "t_evaluar"]
    logger.info(f"\n{pd.DataFrame(tarea.info).T[campos].to_markdown()}")
    pickle.dump(tarea.info, open(workdir / f"{config_file.stem}.pkl", "wb"))


if __name__ == "__main__":
    typer.run(main)
