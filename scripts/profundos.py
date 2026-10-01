"""
Barrido de los clasificadores de `fkdc.profundos` sobre las tareas de la tesis.

Comandos:

- `verificar`: arma la `Tarea` de algunas tareas de v5 tal como lo hace `barrer` y
  la coteja con los infos de la tesis: misma verosimilitud y mismas proporciones
  del clasificador trivial, y un `y_eval` que reproduce exactamente la
  log-verosimilitud y la exactitud que guardó fkdc.
- `barrer`: para cada tarea de la tesis (las de fkdc en `sandbox/v5/infos`: misma
  variante, semilla de dataset, semilla de tarea y estandarización) ajusta el
  clasificador pedido sobre el pliego de entrenamiento, sin búsqueda de
  hiperparámetros y con `random_state` igual a la semilla de la tarea, coteja la
  partición con v5 y guarda `tarea.info` en `sandbox/fundacionales/infos` con el
  nombre de la tesis. Recorre las tareas repetición por repetición, de modo que
  una corrida interrumpida deja resultados parejos; las ya guardadas se saltean.
- `resumir`: mediana y rango intercuartílico de R² y exactitud por variante de
  dataset, de estos clasificadores junto a los de la tesis, en
  `sandbox/fundacionales/resumen-profundos.csv`.

Uso, desde la raíz del repositorio:
    .venv/bin/python scripts/profundos.py verificar
    nice -n 10 .venv/bin/python scripts/profundos.py barrer tabicl --hilos 4
    .venv/bin/python scripts/profundos.py resumir
"""

import inspect
import logging
import pickle
import time
import warnings
from importlib import import_module
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils import Bunch

from fkdc import config, fundacionales, profundos

# Los datasets se guardaron corriendo `fkdc/datasets.py` como script, así que
# pickle busca `Dataset` en `__main__`.
from fkdc.datasets import Dataset, datasets
from fkdc.tarea import Tarea

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="GPU available but not used")
for nombre_logger in ("fkdc.tarea", "pytorch_lightning", "lightning"):
    logging.getLogger(nombre_logger).setLevel(logging.WARNING)
logger = logging.getLogger("profundos")
logger.setLevel(logging.INFO)

dir_v5 = config.dir_ejecucion.parent
dir_salida = fundacionales.dir_fundacionales
dir_infos = fundacionales.dir_infos
puntuacion = "neg_log_loss"
app = typer.Typer(add_completion=False)


def _tareas(patrones=("*",)) -> list[Bunch]:
    """Tareas de la tesis: las que tienen info de fkdc en `sandbox/v5/infos`."""
    tareas = []
    for ruta in fundacionales._rutas(config.dir_ejecucion, ["fkdc"], patrones):
        partes = fundacionales._partes(ruta)
        variante = partes["dataset"]
        nombre_ds = variante.removesuffix("_std")
        if partes["semilla_ds"] != "None":
            nombre_ds += f"-{partes['semilla_ds']}"
        semilla_tarea = int(partes["semilla_tarea"])
        semilla_ds = partes["semilla_ds"]
        tareas.append(
            Bunch(
                variante=variante,
                semilla_ds=semilla_ds,
                semilla_tarea=semilla_tarea,
                estandarizar=variante.endswith("_std"),
                ruta_dataset=dir_v5 / "datasets" / f"{nombre_ds}.pkl",
                # Repetición: la semilla del dataset o, si no la hay, la de la tarea
                repeticion=int(semilla_ds) if semilla_ds != "None" else semilla_tarea,
            )
        )
    return sorted(tareas, key=lambda t: (t.repeticion, datasets.index(t.variante)))


def _nombre_info(tarea: Bunch, clf: str) -> str:
    """`{variante}-{semilla_ds}-{clf}-{semilla_tarea}-{puntuacion}.pkl`, como en v5."""
    clave = (tarea.variante, tarea.semilla_ds, clf, tarea.semilla_tarea, puntuacion)
    return "-".join(map(str, clave)) + ".pkl"


def _armar_tarea(tarea: Bunch, dataset: Dataset, algoritmos: dict) -> Tarea:
    """La `Tarea` de la tesis, con la partición y estandarización de v5."""
    return Tarea(
        dataset,
        algoritmos,
        scoring=puntuacion,
        cv=config.cv,
        split_evaluacion=config.split_evaluacion,
        semilla=tarea.semilla_tarea,
        estandarizar=tarea.estandarizar,
    )


def _cotejar_con_v5(tarea_v5: Bunch, tarea: Tarea) -> dict:
    """Coteja la partición de `tarea` con el info de fkdc en v5 de la misma tarea."""
    info_v5 = pickle.load(
        open(config.dir_ejecucion / _nombre_info(tarea_v5, "fkdc"), "rb")
    )
    base, base_v5 = tarea.info.base, info_v5.base
    logvero_fkdc = -log_loss(
        tarea.y_eval,
        info_v5.fkdc.probas,
        normalize=False,
        labels=tarea.dataset.etiquetas,
    )
    return {
        "base_logvero": base.logvero == base_v5.logvero,
        "base_probas": base.probas.sort_index().equals(base_v5.probas.sort_index()),
        "y_eval_logvero_fkdc": logvero_fkdc == info_v5.fkdc.logvero,
        "y_eval_accuracy_fkdc": accuracy_score(info_v5.fkdc.preds, tarea.y_eval)
        == info_v5.fkdc.accuracy,
    }


@app.command()
def verificar(
    patron: Annotated[
        list[str] | None, typer.Option(help="Glob de infos de fkdc en v5")
    ] = None,
):
    """Coteja con v5 la partición que arma `barrer` (por omisión, tres tareas)."""
    patron = patron or [
        "lunas_lo-1075-fkdc-1312-*",
        "pinguinos-None-fkdc-1075-*",
        "digitos_std-None-fkdc-2048-*",
    ]
    for tarea_v5 in _tareas(patron):
        dataset = Dataset.cargar(tarea_v5.ruta_dataset)
        tarea = _armar_tarea(tarea_v5, dataset, {})
        tarea.evaluar()  # sin clasificadores: solo la base
        cotejo = _cotejar_con_v5(tarea_v5, tarea)
        estado = "OK" if all(cotejo.values()) else "DIFIERE"
        print(f"{estado} {_nombre_info(tarea_v5, '*')} {cotejo}")


@app.command()
def barrer(
    clf: str,
    dispositivo: Annotated[str, typer.Option(help="cpu, mps o auto")] = "cpu",
    hilos: Annotated[int, typer.Option(help="Hilos de torch")] = 4,
    patron: Annotated[
        list[str] | None, typer.Option(help="Glob de infos de fkdc en v5")
    ] = None,
    limite: Annotated[int | None, typer.Option(help="Máximo de tareas")] = None,
):
    """Evalúa `clf` en cada tarea de la tesis que aún no tiene info."""
    import torch

    torch.set_num_threads(hilos)
    nombre_modulo, nombre_fabrica = profundos.clasificadores[clf].fabrica.split(":")
    fabrica = getattr(import_module(nombre_modulo), nombre_fabrica)
    opciones = {"dispositivo": dispositivo}
    if "hilos" in inspect.signature(fabrica).parameters:
        opciones["hilos"] = hilos

    dir_infos.mkdir(parents=True, exist_ok=True)
    tareas = _tareas(patron or ["*"])
    pendientes = [t for t in tareas if not (dir_infos / _nombre_info(t, clf)).exists()]
    logger.info(
        "%s en %s: %d tareas, %d pendientes, %d en esta corrida",
        clf,
        dispositivo,
        len(tareas),
        len(pendientes),
        len(pendientes[:limite]),
    )
    pendientes = pendientes[:limite]
    t_inicio, fallas, discrepancias = time.time(), 0, 0
    for i, tarea_v5 in enumerate(pendientes, start=1):
        nombre = _nombre_info(tarea_v5, clf)
        try:
            dataset = Dataset.cargar(tarea_v5.ruta_dataset)
            estimador = profundos.instanciar(clf, tarea_v5.semilla_tarea, **opciones)
            tarea = _armar_tarea(tarea_v5, dataset, {clf: (estimador, {})})
            tarea.evaluar()
            info = tarea.info[clf]
            if "r2" not in info:  # `Tarea.evaluar` traga excepciones
                raise RuntimeError(f"Evaluación incompleta en {nombre}")
        except Exception:
            logger.exception("Falló %s", nombre)
            fallas += 1
            if fallas >= 3:
                logger.error("Tres fallas; me detengo")
                break
            continue
        cotejo = _cotejar_con_v5(tarea_v5, tarea)
        if not all(cotejo.values()):
            discrepancias += 1
            logger.error("La partición de %s no coincide con v5: %s", nombre, cotejo)
        with open(dir_infos / nombre, "wb") as fp:
            pickle.dump(tarea.info, fp)
        transcurrido = time.time() - t_inicio
        logger.info(
            "[%d/%d] %s: R²=%.3f, exactitud=%.3f, entrenar %.1fs, probas %.1fs; "
            "faltan ~%.0f min",
            i,
            len(pendientes),
            nombre,
            info.r2,
            info.accuracy,
            info.t_entrenar,
            info.t_probas,
            transcurrido / i * (len(pendientes) - i) / 60,
        )
    logger.info(
        "%s: %d tareas en %.1f min, %d fallas, %d particiones distintas de v5",
        clf,
        len(pendientes),
        (time.time() - t_inicio) / 60,
        fallas,
        discrepancias,
    )


@app.command()
def resumir(ruta_salida: Path = dir_salida / "resumen-profundos.csv"):
    """Mediana e IQR de R² y exactitud por variante y clasificador, con la tesis."""
    tesis = fundacionales._cargar_basicos(
        config.dir_ejecucion, list(config.clasificadores)
    )
    nuevos = fundacionales._cargar_basicos(dir_infos, list(profundos.clasificadores))
    clave = ["dataset", "semilla_ds", "semilla_tarea"]
    bases = tesis.drop_duplicates(clave).set_index(clave).base_logvero
    contraste = nuevos.join(bases, on=clave, rsuffix="_tesis")
    distintas = (contraste.base_logvero != contraste.base_logvero_tesis).sum()
    if distintas:
        logger.warning("%d tareas no reproducen la base.logvero de la tesis", distintas)
    else:
        logger.info("Las %d tareas reproducen la base.logvero de la tesis", len(nuevos))

    todos = pd.concat([tesis, nuevos], ignore_index=True)
    todos["t_inferencia"] = todos.t_probas.fillna(todos.t_evaluar)

    def iqr(serie):
        return serie.quantile(0.75) - serie.quantile(0.25)

    resumen = (
        todos.groupby(["dataset", "clf"])
        .agg(
            n=("accuracy", "size"),
            r2_mediana=("r2", "median"),
            r2_iqr=("r2", iqr),
            accuracy_mediana=("accuracy", "median"),
            accuracy_iqr=("accuracy", iqr),
            t_entrenar_mediana=("t_entrenar", "median"),
            t_inferencia_mediana=("t_inferencia", "median"),
        )
        .reset_index()
    )
    orden = {
        "dataset": {nombre: i for i, nombre in enumerate(datasets)},
        "clf": {
            nombre: i
            for i, nombre in enumerate(
                [*config.clasificadores, *profundos.clasificadores]
            )
        },
    }
    resumen = resumen.sort_values(
        ["dataset", "clf"], key=lambda col: col.map(orden[col.name])
    )
    resumen.to_csv(ruta_salida, index=False, float_format="%.4g")
    logger.info("Escribió %s", ruta_salida)
    tabla = resumen.pivot(index="dataset", columns="clf", values="r2_mediana")
    tabla = tabla.loc[[d for d in datasets if d in tabla.index]]
    print(tabla[[c for c in orden["clf"] if c in tabla.columns]].round(3).to_string())


if __name__ == "__main__":
    np.seterr(all="ignore")
    app()
