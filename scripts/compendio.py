"""
Barrido del compendio de clasificadores clásicos (`fkdc.compendio`) sobre las tareas
de la tesis.

Comandos:

- `verificar`: en tres tareas de v5 (una sintética, una `_std` y una de `mnist`)
  rehace la partición de `Tarea` y comprueba que coincidan exactamente con v5 la
  log-verosimilitud y las proporciones del clasificador trivial, y que `y_eval`
  reproduzca exactamente la exactitud y la log-verosimilitud guardadas de lr.
- `barrer`: para cada tarea de la tesis (las variantes de `fkdc.datasets.datasets`,
  con los datasets, semillas y particiones de `sandbox/v5`) y cada clasificador
  pedido, evalúa la `Tarea` y guarda `tarea.info` en `sandbox/fundacionales/infos`
  con el nombre de v5. Reparte las tareas entre `--procesos` procesos (cada
  búsqueda en grilla, secuencial y con BLAS en un hilo) y las encola repetición por
  repetición, de modo que un barrido interrumpido deja resultados parejos; saltea
  las que ya tienen info. Cada tarea se coteja con v5 como en `verificar`, y el
  tiempo de pared, el resultado y el cotejo se anotan en `registro-compendio.csv`.
  La búsqueda se guarda sin `best_estimator_`, que en los bosques y en Isomap pesa
  decenas de MB por tarea: `best_params_` y la partición, que depende solo de la
  semilla, alcanzan para reajustarlo.
- `resumir`: mediana y rango intercuartílico de R² y exactitud por variante, del
  compendio junto a los clasificadores de la tesis en v5, en
  `resumen-compendio.csv`.

Uso: `nice -n 10 .venv/bin/python scripts/compendio.py barrer --clf qda --procesos 4`.
"""

import logging
import pickle
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from scipy.sparse import SparseEfficiencyWarning
from sklearn.exceptions import ConvergenceWarning, InconsistentVersionWarning
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.utils import Bunch

from fkdc import compendio, config, dir_raiz
from fkdc.datasets import Dataset, datasets
from fkdc.tarea import Tarea

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("compendio")

dir_v5 = dir_raiz / "sandbox/v5"
dir_salida = dir_raiz / "sandbox/fundacionales"
dir_infos = dir_salida / "infos"
ruta_registro = dir_salida / "registro-compendio.csv"
clfs_tesis = list(config.clasificadores)
clfs_compendio = list(compendio.clasificadores)
opcion_clf = typer.Option(clfs_compendio, help="Clasificadores del compendio")
opcion_variante = typer.Option(None, help="Variantes a correr (todas, por omisión)")
app = typer.Typer(add_completion=False)


def _silenciar():
    """Silencia las advertencias esperables y el registro por tarea de `Tarea`."""
    # Isomap une las componentes conexas de un grafo de vecinos disconexo con
    # aristas euclídeas (UserWarning) y, al hacerlo, cambia la estructura de una
    # matriz rala (SparseEfficiencyWarning)
    for categoria in (
        ConvergenceWarning,
        InconsistentVersionWarning,
        UserWarning,
        RuntimeWarning,
        SparseEfficiencyWarning,
    ):
        warnings.filterwarnings("ignore", category=categoria)
    logging.getLogger("fkdc.tarea").setLevel(logging.WARNING)


def _semillas_tarea(semilla_dataset: int | None) -> list[int]:
    """Semillas de tarea de cada dataset, como en `config.hacer_configuraciones`."""
    semillas = config._obtener_semillas(config.semilla_principal, config.repeticiones)
    if semilla_dataset is not None and semilla_dataset in semillas:
        return [config.semilla_principal]
    return semillas


def _tareas(
    variantes: list[str] | None = None,
    repeticiones: int | None = None,
    dir_datasets: Path = dir_v5 / "datasets",
) -> list[Bunch]:
    """Tareas de v5, ordenadas por repetición y luego por variante.

    La repetición es la semilla del dataset o, si el dataset es único, la de la
    tarea; ambas salen de las mismas `config.repeticiones` semillas, y
    `repeticiones` se queda con las primeras.
    """
    semillas = config._obtener_semillas(config.semilla_principal, config.repeticiones)
    semillas = set(semillas[:repeticiones])
    tareas = []
    for orden, variante in enumerate(variantes or datasets):
        base = variante.removesuffix("_std")
        rutas = {None: dir_datasets / f"{base}.pkl"}
        rutas.update(
            {
                int(ruta.stem.split("-")[1]): ruta
                for ruta in dir_datasets.glob(f"{base}-*.pkl")
            }
        )
        for semilla_dataset, ruta in rutas.items():
            if not ruta.exists():
                continue
            for semilla_tarea in _semillas_tarea(semilla_dataset):
                repeticion = semilla_dataset or semilla_tarea
                if repeticion in semillas:
                    tareas.append(
                        Bunch(
                            variante=variante,
                            semilla_dataset=semilla_dataset,
                            semilla_tarea=semilla_tarea,
                            ruta=ruta,
                            repeticion=repeticion,
                            orden=orden,
                        )
                    )
    return sorted(tareas, key=lambda t: (t.repeticion, t.orden))


def _nombre_info(variante, semilla_dataset, clf, semilla_tarea, puntuacion):
    return f"{variante}-{semilla_dataset}-{clf}-{semilla_tarea}-{puntuacion}.pkl"


def _puntuacion(clasificador) -> str:
    """Como en `config.hacer_configuraciones`: sin probabilidades, exactitud."""
    return "neg_log_loss" if hasattr(clasificador, "predict_proba") else "accuracy"


def _busqueda(estimador, espacio, *, n_jobs=None, **kwargs):
    """`GridSearchCV` secuencial: el paralelismo es entre tareas, no dentro."""
    return GridSearchCV(estimador, espacio, n_jobs=1, **kwargs)


def _cotejar_con_v5(tarea: Tarea, t: Bunch) -> dict:
    """Coteja la partición de `tarea` con la del info de lr en v5 de la misma tarea.

    - `base_*`: proporciones de clase en entrenamiento y log-verosimilitud del
      clasificador trivial en evaluación, idénticas.
    - `y_eval`: con las probabilidades y predicciones guardadas de lr, `y_eval`
      reproduce exactamente su exactitud y su log-verosimilitud.
    - `X_eval_dif_max`: la lr ajustada en v5 reproduce, sobre `X_eval`, las
      probabilidades que guardó.
    """
    nombre = _nombre_info(
        t.variante, t.semilla_dataset, "lr", t.semilla_tarea, "neg_log_loss"
    )
    with open(dir_v5 / "infos" / nombre, "rb") as fp:
        info_v5 = pickle.load(fp)
    lr, base, base_v5 = info_v5.lr, tarea.info.base, info_v5.base
    logvero_lr = -log_loss(
        tarea.y_eval, lr.probas, normalize=False, labels=tarea.dataset.etiquetas
    )
    probas_lr = lr.busqueda.best_estimator_.predict_proba(tarea.X_eval)
    return {
        "base_logvero": base.logvero == base_v5.logvero,
        "base_probas": base.probas.sort_index().equals(base_v5.probas.sort_index()),
        "y_eval": bool(
            accuracy_score(lr.preds, tarea.y_eval) == lr.accuracy
            and logvero_lr == lr.logvero
        ),
        "X_eval_dif_max": float(np.abs(probas_lr - lr.probas).max()),
    }


def _inicializar_proceso():
    """BLAS en un hilo por proceso y advertencias silenciadas."""
    from threadpoolctl import threadpool_limits

    global _limites_hilos
    _limites_hilos = threadpool_limits(limits=1)
    _silenciar()


def _correr(t: Bunch, nombre_clf: str) -> dict:
    """Evalúa `nombre_clf` en la tarea `t`, guarda el info y devuelve su registro."""
    t_inicio = time.time()
    dataset = Dataset.cargar(t.ruta)
    clasificador = compendio.instanciar(nombre_clf)
    puntuacion = _puntuacion(clasificador)
    espacio = compendio.grilla(nombre_clf, dataset.n, dataset.p)
    tarea = Tarea(
        dataset,
        {nombre_clf: (clasificador, espacio)},
        busqueda_factory=_busqueda,
        scoring=puntuacion,
        cv=config.cv,
        split_evaluacion=config.split_evaluacion,
        semilla=t.semilla_tarea,
        estandarizar=t.variante.endswith("_std"),
    )
    tarea.evaluar()
    info = tarea.info[nombre_clf]
    if "accuracy" not in info or (puntuacion == "neg_log_loss" and "r2" not in info):
        raise RuntimeError(f"Evaluación incompleta de {nombre_clf} en {t.variante}")
    cotejo = _cotejar_con_v5(tarea, t)
    if info.busqueda is not None:
        mejores = info.busqueda.best_params_
        n_candidatos = len(info.busqueda.cv_results_["params"])
        del info.busqueda.best_estimator_
    else:
        mejores, n_candidatos = {}, 1
    nombre = _nombre_info(
        t.variante, t.semilla_dataset, nombre_clf, t.semilla_tarea, puntuacion
    )
    temporal = dir_infos / f".{nombre}.tmp"
    with open(temporal, "wb") as fp:
        pickle.dump(tarea.info, fp)
    temporal.rename(dir_infos / nombre)  # atómico: nunca queda un info a medias
    probas = info.get("probas")
    return {
        "tarea": Path(nombre).stem,
        "clf": nombre_clf,
        "variante": t.variante,
        "semilla_dataset": t.semilla_dataset,
        "semilla_tarea": t.semilla_tarea,
        "segundos": time.time() - t_inicio,
        "t_entrenar": info.t_entrenar,
        "r2": info.get("r2", np.nan),
        "accuracy": info.accuracy,
        "probas_validas": probas is not None
        and bool(np.isfinite(probas).all() and np.allclose(probas.sum(axis=1), 1)),
        "n_candidatos": n_candidatos,
        "mejores_params": repr(
            {k.removeprefix("clf__"): _escalar(v) for k, v in mejores.items()}
        ),
        **cotejo,
    }


def _escalar(valor):
    """Valor nativo y breve de un hiperparámetro, para el registro."""
    if isinstance(valor, np.generic):
        valor = valor.item()
    return float(f"{valor:.4g}") if isinstance(valor, float) else valor


@app.command()
def verificar():
    """Coteja con v5 la partición de tres tareas: sintética, `_std` y `mnist`."""
    _silenciar()
    primeras = {t.variante: t for t in _tareas(repeticiones=1)}
    for variante in ("lunas_lo", "pinguinos_std", "mnist"):
        t = primeras[variante]
        tarea = Tarea(
            Dataset.cargar(t.ruta),
            {},
            semilla=t.semilla_tarea,
            estandarizar=variante.endswith("_std"),
        )
        tarea.evaluar()  # sin clasificadores: solo la base
        cotejo = _cotejar_con_v5(tarea, t)
        coincide = all(v for k, v in cotejo.items() if k != "X_eval_dif_max")
        nombre = _nombre_info(variante, t.semilla_dataset, "*", t.semilla_tarea, "*")
        logger.info("%s %s %s", "OK" if coincide else "DIFIERE", nombre, cotejo)


@app.command()
def barrer(
    clf: list[str] = opcion_clf,
    variante: list[str] | None = opcion_variante,
    repeticiones: int | None = typer.Option(None, help="Primeras N repeticiones"),
    procesos: int = typer.Option(4, help="Procesos en paralelo"),
):
    """Evalúa los clasificadores del compendio en las tareas de v5."""
    _silenciar()
    dir_infos.mkdir(parents=True, exist_ok=True)
    tareas = _tareas(variante, repeticiones)
    pendientes = [
        (t, nombre_clf)
        for t in tareas
        for nombre_clf in clf
        if not (
            dir_infos
            / _nombre_info(
                t.variante,
                t.semilla_dataset,
                nombre_clf,
                t.semilla_tarea,
                _puntuacion(compendio.clasificadores[nombre_clf]),
            )
        ).exists()
    ]
    logger.info(
        "%d tareas × %d clfs; %d pendientes; %d procesos",
        len(tareas),
        len(clf),
        len(pendientes),
        procesos,
    )
    t_inicio, fallas, filas = time.time(), 0, []
    with ProcessPoolExecutor(procesos, initializer=_inicializar_proceso) as pool:
        futuros = {
            pool.submit(_correr, t, nombre): (t, nombre) for t, nombre in pendientes
        }
        for i, futuro in enumerate(as_completed(futuros), 1):
            t, nombre_clf = futuros[futuro]
            try:
                fila = futuro.result()
            except Exception:
                logger.exception(
                    "Falló %s en %s-%s", nombre_clf, t.variante, t.repeticion
                )
                fallas += 1
                continue
            pd.DataFrame([fila]).to_csv(
                ruta_registro, mode="a", header=not ruta_registro.exists(), index=False
            )
            filas.append(fila)
            cotejo_ok = fila["base_logvero"] and fila["base_probas"] and fila["y_eval"]
            logger.log(
                logging.INFO if cotejo_ok else logging.ERROR,
                "[%d/%d] %s: R²=%.3f, exactitud=%.3f, %s (%.1fs)%s",
                i,
                len(pendientes),
                fila["tarea"],
                fila["r2"],
                fila["accuracy"],
                fila["mejores_params"],
                fila["segundos"],
                "" if cotejo_ok else " ¡LA PARTICIÓN NO COINCIDE CON v5!",
            )
    if filas:
        filas = pd.DataFrame(filas)
        logger.info(
            "Tiempo de pared por tarea (s), por clasificador:\n%s",
            filas.groupby("clf").segundos.describe().to_markdown(floatfmt=".1f"),
        )
    logger.info("Barrido en %.1f min; %d fallas", (time.time() - t_inicio) / 60, fallas)


def _cargar_basicos(directorio: Path, clfs: list[str]) -> pd.DataFrame:
    """R², exactitud y tiempos de cada info de `clfs` en las variantes de la tesis."""
    filas = []
    for ruta in directorio.glob("*.pkl"):
        variante, semilla_dataset, clf, semilla_tarea, _ = ruta.stem.split("-")
        if variante not in datasets or clf not in clfs:
            continue
        with open(ruta, "rb") as fp:
            info = pickle.load(fp)[clf]
        filas.append(
            {
                "variante": variante,
                "clf": clf,
                "semilla_dataset": semilla_dataset,
                "semilla_tarea": semilla_tarea,
                "r2": info.get("r2", np.nan),
                "accuracy": info.get("accuracy", np.nan),
                "t_entrenar": info.get("t_entrenar", np.nan),
            }
        )
    return pd.DataFrame(filas)


@app.command()
def resumir():
    """Mediana e IQR de R² y exactitud por variante: compendio junto a la tesis."""
    _silenciar()
    basicos = pd.concat(
        [
            _cargar_basicos(dir_v5 / "infos", clfs_tesis),
            _cargar_basicos(dir_infos, clfs_compendio),
        ]
    )

    def iqr(serie):
        return serie.quantile(0.75) - serie.quantile(0.25)

    resumen = basicos.groupby(["variante", "clf"]).agg(
        n=("accuracy", "size"),
        r2_mediana=("r2", "median"),
        r2_iqr=("r2", iqr),
        accuracy_mediana=("accuracy", "median"),
        accuracy_iqr=("accuracy", iqr),
        t_entrenar_mediana=("t_entrenar", "median"),
    )
    orden = {
        "variante": {v: i for i, v in enumerate(datasets)},
        "clf": {c: i for i, c in enumerate(clfs_tesis + clfs_compendio)},
    }
    resumen = resumen.reset_index().sort_values(
        ["variante", "clf"], key=lambda col: col.map(orden[col.name])
    )
    ruta = dir_salida / "resumen-compendio.csv"
    resumen.to_csv(ruta, index=False, float_format="%.4g")
    logger.info("Escribió %s", ruta)
    for metrica in ("r2", "accuracy"):
        ancho = resumen.pivot(
            index="variante", columns="clf", values=f"{metrica}_mediana"
        )
        ancho = ancho.reindex(
            index=[v for v in datasets if v in ancho.index],
            columns=[c for c in clfs_tesis + clfs_compendio if c in ancho.columns],
        )
        logger.info("Mediana de %s:\n%s", metrica, ancho.to_markdown(floatfmt=".3f"))


if __name__ == "__main__":
    app()
