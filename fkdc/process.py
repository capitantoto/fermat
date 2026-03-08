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
from fkdc.datasets import ConjuntoDatos
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
    archivo_config: Path = Path("config.yaml"),
    dir_trabajo: Path | None = None,
):
    """Ejecuta entrenamiento y evaluación desde un archivo de config."""
    dir_trabajo = dir_trabajo or Path.cwd()
    dir_trabajo.mkdir(parents=True, exist_ok=True)
    logger.info("Leyendo config %s", archivo_config)
    cfg = yaml.safe_load(open(archivo_config))
    dataset = ConjuntoDatos.cargar(cfg["dataset"])
    clf = cfg["clasificador"]
    clasificador = config.clasificadores.get(clf)
    if clasificador is None:
        try:
            clasificador = open(Path(clf), "rb")
        except Exception as exc:
            logger.error("Error leyendo clasificador de %s", clf, exc_info=True)
            raise exc
    cv = cfg.get("cv", config.cv)
    scoring = cfg.get("scoring", config.puntuacion)
    logger.info("Corrida de prueba")
    split = cfg.get("split_evaluacion", config.split_evaluacion)
    semilla = cfg.get("seed", config.semilla_principal)
    grilla_hipers = cfg.get("grilla_hipers", {})
    t_tarea = time.time()
    tarea = Tarea(
        dataset,
        {clf: (clasificador, grilla_hipers)},
        cv=cv,
        split_evaluacion=split,
        semilla=semilla,
        scoring=scoring,
    )
    logger.info("Entrenamiento principal")
    t_tarea = time.time()
    tarea.evaluar()
    logger.info("- Tomó %.2fs", time.time() - t_tarea)
    if cfg.get("guardar_tarea", False):
        tarea.guardar(dir_trabajo / f"tarea-{archivo_config.stem}.pkl")
    logger.info("Resumen resultados")
    campos = ["logvero", "r2", "accuracy", "t_entrenar", "t_evaluar"]
    logger.info(f"\n{pd.DataFrame(tarea.info).T[campos].to_markdown()}")
    pickle.dump(tarea.info, open(dir_trabajo / f"{archivo_config.stem}.pkl", "wb"))


if __name__ == "__main__":
    typer.run(main)
