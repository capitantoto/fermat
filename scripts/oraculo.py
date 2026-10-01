"""
Barrido de los clasificadores oráculo (`fkdc.oraculo`) sobre las tareas sintéticas.

Comandos:

- `verificar`: control de cordura de φ por familia. Sobre datos sin ruido de los
  generadores comprueba que φ recupere exactamente la coordenada que define la
  clase, y sobre la variante de bajo ruido mide la exactitud de entrenamiento de
  una regresión logística casi sin penalizar en el espacio φ (separabilidad lineal).
- `barrer`: para cada tarea sintética de la tesis y cada clasificador pedido arma
  la `Tarea` con la misma semilla, partición y puntuación que en `sandbox/v5`, la
  evalúa y guarda `tarea.info` en `sandbox/fundacionales/infos` como
  `{clf}_oraculo`. Cada tarea se coteja con los infos de v5: misma verosimilitud
  del clasificador trivial, mismas observaciones de evaluación y mismas etiquetas.
- `resumir`: mediana y rango intercuartílico de R² y exactitud por dataset, de
  los oráculos junto a los clasificadores de la tesis, en `resumen-oraculo.csv`.

Uso: `nice -n 10 .venv/bin/python scripts/oraculo.py barrer --clf lr --clf kdc`.
"""

import logging
import pickle
import time
import warnings
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from joblib import parallel_config
from sklearn.datasets import make_circles, make_moons
from sklearn.exceptions import ConvergenceWarning, InconsistentVersionWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV

from fkdc import config, dir_raiz
from fkdc.datasets import (
    Dataset,
    datasets,
    hacer_anteojos,
    hacer_eslabones,
    hacer_espirales,
    hacer_helices,
    hacer_hueveras,
    hacer_pionono,
)
from fkdc.oraculo import clasificador_oraculo, familia, mapas_oraculo
from fkdc.tarea import Tarea

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
logging.getLogger("fkdc.tarea").setLevel(logging.WARNING)
logger = logging.getLogger("oraculo")
logger.setLevel(logging.INFO)

dir_v5 = dir_raiz / "sandbox/v5"
dir_salida = dir_raiz / "sandbox/fundacionales"
clfs_oraculo = ["lr", "gnb", "kdc", "kn", "svc", "gbt"]  # fkdc sobre φ: muy costoso
clfs_tesis = ["fkdc", "kdc", "fkn", "kn", "gnb", "svc", "gbt", "lr"]
opcion_clf = typer.Option(clfs_oraculo, help="Clasificadores a montar sobre φ")
opcion_variante = typer.Option(
    [], "--variante", help="Restringe el barrido a estas variantes"
)
opcion_hilos_internos = typer.Option(
    None,
    help="Hilos OpenMP/BLAS por proceso (p. ej. 1 para gbt en una máquina cargada)",
)
variantes = [
    d for d in datasets if not d.endswith("_std") and d.split("_")[0] in mapas_oraculo
]
app = typer.Typer(add_completion=False)


def _semillas_tarea(semilla_dataset: int | None) -> list[int]:
    """Semillas de tarea de cada dataset, como en `config.hacer_configuraciones`."""
    semillas = config._obtener_semillas(config.semilla_principal, config.repeticiones)
    if semilla_dataset is not None and semilla_dataset in semillas:
        return [config.semilla_principal]
    return semillas


def _tareas(dir_datasets: Path = dir_v5 / "datasets"):
    """Tareas sintéticas de v5: (variante, semilla_dataset, semilla_tarea, ruta)."""
    for variante in variantes:
        rutas = {None: dir_datasets / f"{variante}.pkl"}
        rutas.update(
            {
                int(ruta.stem.split("-")[1]): ruta
                for ruta in dir_datasets.glob(f"{variante}-*.pkl")
            }
        )
        for semilla_dataset, ruta in sorted(rutas.items(), key=lambda kv: kv[0] or 0):
            if not ruta.exists():
                continue
            for semilla_tarea in _semillas_tarea(semilla_dataset):
                yield variante, semilla_dataset, semilla_tarea, ruta


def _nombre_info(variante, semilla_dataset, clf, semilla_tarea, puntuacion):
    return f"{variante}-{semilla_dataset}-{clf}-{semilla_tarea}-{puntuacion}.pkl"


def _busqueda(estimador, espacio, *, n_jobs_max: int, n_jobs=None, **kwargs):
    """`GridSearchCV` con paralelismo acotado (el resultado no depende de n_jobs)."""
    return GridSearchCV(estimador, espacio, n_jobs=n_jobs_max, **kwargs)


def _cotejar_con_v5(tarea: Tarea, variante, semilla_dataset, semilla_tarea) -> dict:
    """Coteja la partición de `tarea` con la de los infos de v5 de la misma tarea.

    - `base`: proporciones de clase en entrenamiento y log-verosimilitud del
      clasificador trivial en evaluación, idénticas.
    - `X_eval`: la regresión logística ajustada en v5 reproduce, sobre `X_eval`,
      las probabilidades que guardó v5.
    - `y_eval`: con las probabilidades y predicciones guardadas de fkdc en v5,
      `y_eval` reproduce su log-verosimilitud y exactitud.
    """
    ruta = dir_v5 / "infos"
    info_lr, info_fkdc = (
        pickle.load(
            open(
                ruta
                / _nombre_info(
                    variante, semilla_dataset, clf, semilla_tarea, "neg_log_loss"
                ),
                "rb",
            )
        )
        for clf in ("lr", "fkdc")
    )
    base, base_v5 = tarea.info.base, info_lr.base
    probas_lr = info_lr.lr.busqueda.best_estimator_.predict_proba(tarea.X_eval)
    logvero_fkdc = -log_loss(
        tarea.y_eval,
        info_fkdc.fkdc.probas,
        normalize=False,
        labels=tarea.dataset.etiquetas,
    )
    return {
        "base_logvero": base.logvero == base_v5.logvero,
        "base_probas": base.probas.sort_index().equals(base_v5.probas.sort_index()),
        "X_eval_dif_max": float(np.abs(probas_lr - info_lr.lr.probas).max()),
        "y_eval_logvero_fkdc": logvero_fkdc == info_fkdc.fkdc.logvero,
        "y_eval_accuracy_fkdc": accuracy_score(info_fkdc.fkdc.preds, tarea.y_eval)
        == info_fkdc.fkdc.accuracy,
    }


@app.command()
def barrer(
    clf: list[str] = opcion_clf,
    variantes_elegidas: list[str] = opcion_variante,
    n_jobs: int = typer.Option(4, help="Procesos por búsqueda en grilla"),
    hilos_internos: int | None = opcion_hilos_internos,
    rehacer: bool = typer.Option(False, help="Recalcula los infos ya guardados"),
    invertir: bool = typer.Option(
        False, help="Recorre las tareas al revés (para repartirlas entre dos procesos)"
    ),
):
    """Evalúa los oráculos en cada tarea sintética de v5 y guarda `tarea.info`."""
    dir_infos = dir_salida / "infos"
    dir_infos.mkdir(parents=True, exist_ok=True)
    tareas = [
        tarea
        for tarea in _tareas()
        if not variantes_elegidas or tarea[0] in variantes_elegidas
    ]
    if invertir:
        tareas.reverse()
    logger.info("%d tareas en %d variantes; clfs: %s", len(tareas), len(variantes), clf)
    cotejos, t_inicio = {nombre: [] for nombre in clf}, time.time()
    for i, (variante, semilla_dataset, semilla_tarea, ruta) in enumerate(tareas, 1):
        dataset, cotejo = Dataset.cargar(ruta), None
        for nombre_base in clf:
            nombre_clf = f"{nombre_base}_oraculo"
            clasificador, grilla = clasificador_oraculo(
                variante, nombre_base, dataset.n
            )
            # Como en `config.hacer_configuraciones`: sin probabilidades, exactitud
            puntuacion = (
                "neg_log_loss" if hasattr(clasificador, "predict_proba") else "accuracy"
            )
            nombre = _nombre_info(
                variante, semilla_dataset, nombre_clf, semilla_tarea, puntuacion
            )
            if (dir_infos / nombre).exists() and not rehacer:
                continue
            tarea = Tarea(
                dataset,
                {nombre_clf: (clasificador, grilla)},
                busqueda_factory=partial(_busqueda, n_jobs_max=n_jobs),
                scoring=puntuacion,
                cv=config.cv,
                split_evaluacion=config.split_evaluacion,
                semilla=semilla_tarea,
            )
            with (
                parallel_config(backend="loky", inner_max_num_threads=hilos_internos)
                if hilos_internos
                else nullcontext()
            ):
                tarea.evaluar()
            # La partición depende solo de la semilla: basta cotejarla una vez
            cotejo = cotejo or _cotejar_con_v5(
                tarea, variante, semilla_dataset, semilla_tarea
            )
            cotejos[nombre_base].append({"tarea": nombre, **cotejo})
            if not all(v for k, v in cotejo.items() if k != "X_eval_dif_max"):
                logger.error(
                    "La partición de %s no coincide con v5: %s", nombre, cotejo
                )
            with open(dir_infos / nombre, "wb") as fp:
                pickle.dump(tarea.info, fp)
            info = tarea.info[nombre_clf]
            logger.info(
                "[%d/%d] %s: R²=%.3f, exactitud=%.3f, %s (%.1fs)",
                i,
                len(tareas),
                nombre,
                info.get("r2", np.nan),
                info.accuracy,
                info.busqueda.best_params_,
                info.t_entrenar,
            )
    for nombre_base, filas in cotejos.items():
        if not filas:
            continue
        # Se acumula con lo cotejado en corridas anteriores (o en paralelo)
        ruta_cotejo = dir_salida / f"cotejo-particiones-{nombre_base}_oraculo.csv"
        previas = [pd.read_csv(ruta_cotejo)] if ruta_cotejo.exists() else []
        filas = pd.concat([*previas, pd.DataFrame(filas)])
        filas = filas.drop_duplicates("tarea", keep="last")
        filas.to_csv(ruta_cotejo, index=False)
        logger.info(
            "Cotejo con v5 de %s_oraculo en %d tareas: coincide = %s; "
            "dif. máx. en X_eval = %.2g",
            nombre_base,
            len(filas),
            bool(filas.drop(columns=["tarea", "X_eval_dif_max"]).all().all()),
            filas.X_eval_dif_max.max(),
        )
    logger.info("Barrido completo en %.1f min", (time.time() - t_inicio) / 60)


def _muestras_sin_ruido(nombre: str, semilla: int, n: int = config.n_muestras):
    """Muestra sin ruido de la familia `nombre`, con los parámetros de la tesis."""
    fabricas = {
        "lunas": partial(make_moons, noise=None),
        "circulos": partial(make_circles, noise=None),
        "espirales": partial(hacer_espirales, noise=None),
        "helices": partial(hacer_helices, noise=None),
        "eslabones": partial(hacer_eslabones, noise=None),
        "hueveras": partial(hacer_hueveras, noise=None),
        "pionono": partial(hacer_pionono, noise=0),
        "anteojos": partial(hacer_anteojos, noise=None),
    }
    return fabricas[nombre](n_samples=n, random_state=semilla)


@app.command()
def verificar(c_max: float = typer.Option(1e6, help="C de la regresión logística")):
    """Control de cordura: separabilidad lineal de las clases en el espacio φ."""
    semillas = config._obtener_semillas()
    variantes_bajo_ruido = [
        v for v in variantes if v.endswith(("_lo", "_0")) or v == "anteojos"
    ]
    filas = []
    for variante in variantes_bajo_ruido:
        nombre = familia(variante)
        phi = mapas_oraculo[nombre]
        exactitudes = {"sin_ruido": [], "phi": [], "X": []}
        for semilla in semillas:
            X, y = _muestras_sin_ruido(nombre, semilla)
            lr = LogisticRegression(C=c_max, max_iter=50_000)
            exactitudes["sin_ruido"].append(lr.fit(phi(X), y).score(phi(X), y))
        rutas = sorted((dir_v5 / "datasets").glob(f"{variante}*.pkl"))
        for ruta in rutas:
            ds = Dataset.cargar(ruta)
            for clave, Z in (("phi", phi(ds.X)), ("X", ds.X)):
                lr = LogisticRegression(C=c_max, max_iter=50_000)
                exactitudes[clave].append(lr.fit(Z, ds.y).score(Z, ds.y))
        filas.append(
            {
                "variante": variante,
                "exactitud_sin_ruido_min": min(exactitudes["sin_ruido"]),
                "exactitud_phi_mediana": np.median(exactitudes["phi"]),
                "exactitud_phi_min": min(exactitudes["phi"]),
                "exactitud_X_mediana": np.median(exactitudes["X"]),
                "n_datasets": len(rutas),
            }
        )
    tabla = pd.DataFrame(filas).set_index("variante")
    logger.info(
        "Exactitud de entrenamiento de LR(C=%g) sobre todo el dataset:\n%s",
        c_max,
        tabla.to_markdown(floatfmt=".3f"),
    )


def _cargar_basicos(dir_infos: Path, clfs: list[str]) -> pd.DataFrame:
    """R², exactitud y semilla de cada info de `clfs` en las variantes del oráculo."""
    filas = []
    for ruta in dir_infos.glob("*.pkl"):
        variante, semilla_dataset, clf, semilla_tarea, _ = ruta.stem.split("-")
        if variante not in variantes or clf not in clfs:
            continue
        with open(ruta, "rb") as fp:
            info = pickle.load(fp)[clf]
        sin_semilla_dataset = semilla_dataset == "None"
        filas.append(
            {
                "variante": variante,
                "clf": clf,
                "semilla": semilla_tarea if sin_semilla_dataset else semilla_dataset,
                "r2": info.get("r2", np.nan),
                "accuracy": info.get("accuracy", np.nan),
            }
        )
    return pd.DataFrame(filas)


@app.command()
def resumir():
    """Mediana e IQR de R² y exactitud por dataset: oráculos vs. clasificadores v5."""
    nombres_oraculo = [f"{clf}_oraculo" for clf in clfs_oraculo]
    basicos = pd.concat(
        [
            _cargar_basicos(dir_salida / "infos", nombres_oraculo),
            _cargar_basicos(dir_v5 / "infos", clfs_tesis),
        ]
    )

    def iqr(serie):
        return serie.quantile(0.75) - serie.quantile(0.25)

    resumen = basicos.groupby(["variante", "clf"])[["r2", "accuracy"]].agg(
        ["median", iqr]
    )
    resumen.columns = [
        f"{metrica}_{'mediana' if estadistico == 'median' else 'iqr'}"
        for metrica, estadistico in resumen.columns
    ]
    resumen["n"] = basicos.groupby(["variante", "clf"]).size()
    # Cada oráculo junto a su contraparte de la tesis (y fkdc, fkn junto a kdc, kn)
    clfs = []
    for clf in clfs_oraculo:
        clfs += [f"{clf}_oraculo", clf, *(["f" + clf] if clf in ("kdc", "kn") else [])]
    ancho = resumen.reset_index().pivot(index="variante", columns="clf")
    ancho = ancho.reindex(index=variantes)
    ancho.columns = [f"{clf}_{columna}" for columna, clf in ancho.columns]
    columnas = [
        f"{clf}_{metrica}_{estadistico}"
        for metrica in ("r2", "accuracy")
        for clf in clfs
        for estadistico in ("mediana", "iqr")
    ]
    columnas += [f"{clf}_n" for clf in clfs]
    ancho = ancho.reindex(columns=columnas)
    ruta = dir_salida / "resumen-oraculo.csv"
    ancho.to_csv(ruta, float_format="%.4f")
    logger.info("Escribió %s", ruta)
    for metrica in ("r2", "accuracy"):
        medianas = ancho[[f"{clf}_{metrica}_mediana" for clf in clfs]]
        medianas.columns = clfs
        logger.info("Mediana de %s:\n%s", metrica, medianas.to_markdown(floatfmt=".3f"))


if __name__ == "__main__":
    app()
