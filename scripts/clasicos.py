"""
Barrido de los clasificadores clásicos (`fkdc.clasicos`) sobre las tareas de la tesis.

Comandos:

- `verificar`: en tres tareas de v5 (una sintética, una `_std` y una de `mnist`)
  rehace la partición de `Tarea` y comprueba que coincidan exactamente con v5 la
  log-verosimilitud y las proporciones del clasificador trivial, y que `y_eval`
  reproduzca exactamente la exactitud y la log-verosimilitud guardadas de lr.
- `probar`: evalúa los clasificadores pedidos en la primera repetición de algunas
  variantes, sin guardar nada: tiempo, R², exactitud, validez de las probabilidades y
  lo que eligió cada uno.
- `barrer`: para cada tarea de la tesis (las variantes de `fkdc.datasets.datasets`,
  con los datasets, semillas y particiones de `sandbox/v5`) y cada clasificador
  pedido, evalúa la `Tarea` y guarda `tarea.info` en `sandbox/fundacionales/infos`
  con el nombre de v5. El estimador y su grilla salen de `fkdc.registro`, como en las
  campañas. Reparte las tareas entre `--procesos` procesos (un hilo de BLAS cada uno,
  con su propio caché de distancias de Fermat, que se vacía tras cada tarea) y las
  encola repetición por repetición; saltea las que ya tienen info. Cada tarea se
  coteja con v5 como en `verificar`, y el tiempo de pared, el resultado y el cotejo
  se anotan en `registro-clasicos.csv`. El info guarda lo que eligió el clasificador
  (`diagnostico`: anchos, escalas, pesos) y no el modelo reajustado.
- `resumir`: mediana y rango intercuartílico de R² y exactitud por variante, de los
  clásicos junto a TabPFN, fkdc, kdc, gpc y gbt, en `resumen-clasicos.csv`; prueba de
  rangos con signo de Wilcoxon (ceros por Pratt) de cada clásico contra TabPFN por
  variante, pareada por semilla, unilateral en cada sentido y con corrección de Holm
  entre variantes, en `wilcoxon-clasicos.csv`.
- `anchos`: h_j / s_j que eligió `kdc_ard` en las variantes con dimensiones de ruido,
  en `anchos-kdc_ard.csv`.

Uso: `nice -n 10 .venv/bin/python scripts/clasicos.py barrer --clf kdc_ard`.
"""

import logging
import os
import pickle
import shutil
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from scipy import stats
from sklearn.exceptions import ConvergenceWarning, InconsistentVersionWarning
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils import Bunch

from fkdc import clasicos, config, dir_raiz, registro
from fkdc.datasets import Dataset, datasets
from fkdc.tarea import Tarea

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("clasicos")

dir_v5 = dir_raiz / "sandbox/v5"
dir_salida = dir_raiz / "sandbox/fundacionales"
dir_infos = dir_salida / "infos"
dir_cache = dir_salida / "cache-clasicos"
ruta_registro = dir_salida / "registro-clasicos.csv"
clfs_clasicos = registro.nombres(["clasicos"])
# Referencias del resumen: TabPFN y gpc en `dir_infos`; los de la tesis, en v5
referencias = {"tabpfn": dir_infos, "gpc": dir_infos}
referencias_v5 = ["fkdc", "kdc", "gbt"]
variantes_ruido = ["eslabones_12", "helices_12", "hueveras_12", "pionono_12"]
nivel = 0.05
opcion_clf = typer.Option(clfs_clasicos, help="Clasificadores de fkdc.clasicos")
opcion_variante = typer.Option(None, help="Variantes a correr (todas, por omisión)")
opcion_variantes_prueba = typer.Option(
    ["helices_12", "hueveras_0", "pinguinos", "mnist", "espirales_hi"],
    help="Variantes",
)
app = typer.Typer(add_completion=False)


def _silenciar():
    """Silencia las advertencias esperables y el registro por tarea de `Tarea`."""
    for categoria in (
        ConvergenceWarning,
        InconsistentVersionWarning,
        UserWarning,
        RuntimeWarning,
        FutureWarning,
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


def _redirigir_cache(directorio: Path):
    """Caché en disco de `fkdc.fermat` propio de este proceso (como `fkdc.correr`)."""
    from joblib import Memory

    from fkdc import fermat

    memoria = Memory(directorio, verbose=0)
    fermat.euclidean = memoria.cache(fermat.euclidean.func)
    fermat.fermat_muestral = memoria.cache(fermat.fermat_muestral.func)


def _inicializar_proceso():
    """BLAS en un hilo, advertencias silenciadas y caché de Fermat propio."""
    from threadpoolctl import threadpool_limits

    import __main__

    global _limites_hilos
    __main__.Dataset = Dataset  # los datasets de v5 se guardaron desde __main__
    _limites_hilos = threadpool_limits(limits=1)
    _silenciar()
    _redirigir_cache(dir_cache / str(os.getpid()))


def _evaluar(t: Bunch, nombre_clf: str) -> tuple[Tarea, Bunch]:
    """`Tarea` de `nombre_clf` en `t`, evaluada con estimador y grilla del registro."""
    dataset = Dataset.cargar(t.ruta)
    estimador, espacio, puntuacion = registro.preparar(
        nombre_clf,
        dataset.n,
        dataset.p,
        variante=t.variante,
        semilla=t.semilla_tarea,
    )
    tarea = Tarea(
        dataset,
        {nombre_clf: (estimador, espacio)},
        scoring=puntuacion,
        cv=config.cv,
        split_evaluacion=config.split_evaluacion,
        semilla=t.semilla_tarea,
        estandarizar=t.variante.endswith("_std"),
        n_jobs=1,
    )
    tarea.evaluar()
    info = tarea.info[nombre_clf]
    if "accuracy" not in info or (puntuacion == "neg_log_loss" and "r2" not in info):
        raise RuntimeError(f"Evaluación incompleta de {nombre_clf} en {t.variante}")
    info.diagnostico = clasicos.diagnostico(tarea.clasificadores[nombre_clf])
    return tarea, info


def _probas_validas(probas) -> bool:
    return probas is not None and bool(
        np.isfinite(probas).all()
        and (probas >= 0).all()
        and np.allclose(probas.sum(axis=1), 1)
    )


def _escalar(valor):
    """Valor nativo y breve de un hiperparámetro, para el registro."""
    if isinstance(valor, np.generic):
        valor = valor.item()
    return float(f"{valor:.4g}") if isinstance(valor, float) else valor


def _resumen_diagnostico(diagnostico: dict) -> str:
    """Lo esencial de lo que eligió el clasificador, en una línea."""
    if "relativos" in diagnostico:
        return f"h/s={np.array2string(diagnostico['relativos'], precision=2)}"
    if "pesos" in diagnostico:
        return repr({k: round(float(v), 3) for k, v in diagnostico["pesos"].items()})
    if "escalas_longitud" in diagnostico:
        return f"ℓ={np.array2string(diagnostico['escalas_longitud'], precision=2)}"
    return ""


def _correr(t: Bunch, nombre_clf: str) -> dict:
    """Evalúa `nombre_clf` en la tarea `t`, guarda el info y devuelve su registro."""
    t_inicio = time.time()
    try:
        tarea, info = _evaluar(t, nombre_clf)
    finally:
        shutil.rmtree(dir_cache / str(os.getpid()), ignore_errors=True)
    cotejo = _cotejar_con_v5(tarea, t)
    puntuacion = tarea.scoring
    if info.busqueda is not None:
        mejores = info.busqueda.best_params_
        n_candidatos = len(info.busqueda.cv_results_["params"])
        info.mejores_params = dict(mejores)
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
        "probas_validas": _probas_validas(info.get("probas")),
        "n_candidatos": n_candidatos,
        "mejores_params": repr(
            {k.removeprefix("clf__"): _escalar(v) for k, v in mejores.items()}
        ),
        "diagnostico": _resumen_diagnostico(info.diagnostico),
        **cotejo,
    }


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
def probar(
    clf: list[str] = opcion_clf,
    variante: list[str] = opcion_variantes_prueba,
    procesos: int = typer.Option(5, help="Procesos en paralelo"),
):
    """Primera repetición de cada variante, sin guardar: tiempo, R² y validez."""
    _silenciar()
    tareas = {t.variante: t for t in _tareas(variante, repeticiones=1)}
    pares = [(tareas[v], nombre) for v in variante for nombre in clf]
    with ProcessPoolExecutor(procesos, initializer=_inicializar_proceso) as pool:
        futuros = {pool.submit(_probar_una, t, n): (t, n) for t, n in pares}
        for futuro in as_completed(futuros):
            t, nombre = futuros[futuro]
            try:
                fila = futuro.result()
            except Exception:
                logger.exception("Falló %s en %s", nombre, t.variante)
                continue
            logger.info(
                "%s en %s: %.1fs, R²=%.3f, exactitud=%.3f, probas %s; %s",
                nombre,
                t.variante,
                fila["segundos"],
                fila["r2"],
                fila["accuracy"],
                "válidas" if fila["probas_validas"] else "INVÁLIDAS",
                fila["diagnostico"],
            )


def _probar_una(t: Bunch, nombre_clf: str) -> dict:
    t_inicio = time.time()
    try:
        _, info = _evaluar(t, nombre_clf)
    finally:
        shutil.rmtree(dir_cache / str(os.getpid()), ignore_errors=True)
    return {
        "segundos": time.time() - t_inicio,
        "r2": info.get("r2", np.nan),
        "accuracy": info.accuracy,
        "probas_validas": _probas_validas(info.get("probas")),
        "diagnostico": _resumen_diagnostico(info.diagnostico)
        + (f" {info.busqueda.best_params_}" if info.busqueda is not None else ""),
    }


@app.command()
def barrer(
    clf: list[str] = opcion_clf,
    variante: list[str] | None = opcion_variante,
    repeticiones: int | None = typer.Option(None, help="Primeras N repeticiones"),
    procesos: int = typer.Option(8, help="Procesos en paralelo"),
):
    """Evalúa los clasificadores clásicos en las tareas de v5."""
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
                "neg_log_loss",
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
                "[%d/%d] %s: R²=%.3f, exactitud=%.3f, %s %s (%.1fs)%s",
                i,
                len(pendientes),
                fila["tarea"],
                fila["r2"],
                fila["accuracy"],
                fila["mejores_params"],
                fila["diagnostico"][:120],
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


def holm(p) -> np.ndarray:
    """p-valores ajustados por Holm (paso descendente, monótonos)."""
    p = np.asarray(p, float)
    ajustados = np.empty_like(p)
    acumulado = 0.0
    for rango, i in enumerate(np.argsort(p)):
        acumulado = max(acumulado, (len(p) - rango) * p[i])
        ajustados[i] = min(1.0, acumulado)
    return ajustados


def _wilcoxon(diferencias: np.ndarray, alternativa: str) -> float:
    """p-valor de rangos con signo (ceros por Pratt); 1 si todas son nulas."""
    if np.allclose(diferencias, 0):
        return 1.0
    return float(
        stats.wilcoxon(diferencias, zero_method="pratt", alternative=alternativa).pvalue
    )


def contrastes(basicos: pd.DataFrame, clfs: list[str], rival="tabpfn") -> pd.DataFrame:
    """Wilcoxon de cada clf contra `rival` por variante, pareado por semilla.

    `p_peor`: H₁ el clf tiene menor R² que el rival; `p_mejor`: mayor. Holm entre las
    variantes, por clf y sentido. "No significativamente peor": `holm_peor` ≥ 0,05.
    """
    claves = ["variante", "semilla_dataset", "semilla_tarea"]
    del_rival = basicos[basicos.clf == rival].set_index(claves).r2
    filas = []
    for clf in clfs:
        propios = basicos[basicos.clf == clf].set_index(claves).r2
        pares = pd.concat([propios, del_rival], axis=1, keys=["clf", "rival"]).dropna()
        for variante, grupo in pares.groupby(level="variante"):
            d = (grupo["clf"] - grupo["rival"]).to_numpy()
            filas.append(
                {
                    "clf": clf,
                    "variante": variante,
                    "n": len(d),
                    "r2_clf": grupo["clf"].median(),
                    "r2_rival": grupo["rival"].median(),
                    "dif_mediana": float(np.median(d)),
                    "gana": int((d > 0).sum()),
                    "pierde": int((d < 0).sum()),
                    "p_peor": _wilcoxon(d, "less"),
                    "p_mejor": _wilcoxon(d, "greater"),
                }
            )
    tabla = pd.DataFrame(filas)
    for sentido in ("peor", "mejor"):
        tabla[f"holm_{sentido}"] = tabla.groupby("clf")[f"p_{sentido}"].transform(holm)
    tabla["no_peor"] = tabla.holm_peor >= nivel
    tabla["mejor"] = tabla.holm_mejor < nivel
    orden = {v: i for i, v in enumerate(datasets)}
    return tabla.sort_values(
        ["clf", "variante"], key=lambda c: c.map(orden) if c.name == "variante" else c
    )


@app.command()
def resumir(
    clf: list[str] = opcion_clf,
):
    """Mediana e IQR de R² y exactitud por variante, y Wilcoxon contra TabPFN."""
    _silenciar()
    import __main__

    __main__.Dataset = Dataset
    basicos = pd.concat(
        [
            _cargar_basicos(dir_infos, [*clf, *referencias]),
            _cargar_basicos(dir_v5 / "infos", referencias_v5),
        ]
    )
    presentes = [c for c in clf if c in set(basicos.clf)]

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
    columnas = [*presentes, *referencias, *referencias_v5]
    orden = {
        "variante": {v: i for i, v in enumerate(datasets)},
        "clf": {c: i for i, c in enumerate(columnas)},
    }
    resumen = resumen.reset_index().sort_values(
        ["variante", "clf"], key=lambda col: col.map(orden[col.name])
    )
    ruta = dir_salida / "resumen-clasicos.csv"
    resumen.to_csv(ruta, index=False, float_format="%.4g")
    logger.info("Escribió %s", ruta)
    for metrica in ("r2", "accuracy"):
        ancho = resumen.pivot(
            index="variante", columns="clf", values=f"{metrica}_mediana"
        ).reindex(index=[v for v in datasets if v in set(resumen.variante)])
        ancho = ancho[[c for c in columnas if c in ancho.columns]]
        logger.info("Mediana de %s:\n%s", metrica, ancho.to_markdown(floatfmt=".3f"))
    tabla = contrastes(basicos, presentes)
    ruta = dir_salida / "wilcoxon-clasicos.csv"
    tabla.to_csv(ruta, index=False, float_format="%.4g")
    logger.info("Escribió %s", ruta)
    for nombre, grupo in tabla.groupby("clf", sort=False):
        logger.info(
            "%s contra TabPFN: no significativamente peor en %d/%d variantes (%s); "
            "significativamente mejor en %s",
            nombre,
            grupo.no_peor.sum(),
            len(grupo),
            ", ".join(grupo.variante[grupo.no_peor]),
            ", ".join(grupo.variante[grupo.mejor]) or "ninguna",
        )


@app.command()
def anchos():
    """h_j / s_j de `kdc_ard` en las variantes con 12 dimensiones de ruido."""
    _silenciar()
    filas = []
    for variante in variantes_ruido:
        for ruta in sorted(dir_infos.glob(f"{variante}-*-kdc_ard-*.pkl")):
            with open(ruta, "rb") as fp:
                info = pickle.load(fp)["kdc_ard"]
            relativos = info.diagnostico["relativos"]
            filas.append(
                {
                    "variante": variante,
                    "semilla_dataset": ruta.stem.split("-")[1],
                    "r2": info.r2,
                    **{f"h{j + 1}": r for j, r in enumerate(relativos)},
                }
            )
    tabla = pd.DataFrame(filas)
    ruta = dir_salida / "anchos-kdc_ard.csv"
    tabla.to_csv(ruta, index=False, float_format="%.4g")
    columnas = [c for c in tabla if c.startswith("h")]
    senal, ruido = columnas[:3], columnas[3:]  # las 3 primeras son la variedad
    resumen = pd.DataFrame(
        {
            "n": tabla.groupby("variante").size(),
            "r2_mediana": tabla.groupby("variante").r2.median(),
            "senal_mediana": tabla.groupby("variante")[senal].median().median(axis=1),
            "ruido_mediana": tabla.groupby("variante")[ruido].median().median(axis=1),
            **{
                f"{nombre}_desde_{umbral}": (tabla[cols] >= umbral)
                .mean(axis=1)
                .groupby(tabla.variante)
                .mean()
                for umbral in (10, 100, 999)
                for nombre, cols in (("senal", senal), ("ruido", ruido))
            },
        }
    )
    logger.info(
        "Escribió %s. Medianas de h/s y fracción de dimensiones con h/s ≥ umbral "
        "(999: en la cota):\n%s",
        ruta,
        resumen.to_markdown(floatfmt=".3g"),
    )


if __name__ == "__main__":
    app()
