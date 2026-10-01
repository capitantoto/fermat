"""
Tiempos de cómputo de las campañas: minería, medición y proyección.

Junta los tiempos que ya guardan las corridas (`t_entrenar`, `t_evaluar`, `t_probas`
y los tiempos por candidato de `GridSearchCV.cv_results_`) y los de los logs de las
corridas secuenciales, para estimar cuánto cuesta repetir las tareas de la tesis con
más semillas y con más clasificadores.

Definiciones (por tarea y clasificador, en segundos):

- `t_tarea` = `t_entrenar` + `t_evaluar` + `t_probas`: el tiempo de pared de la tarea
  sin el arranque del proceso. Las infos de v5 no cronometraban `predict_proba`; se
  estima igual a `t_evaluar` (`predict`) cuando el clasificador tiene probabilidades.
- `t_cpu`: el costo secuencial equivalente (como con `n_jobs=1`): la suma, sobre los
  candidatos de la grilla, de ajuste y puntaje por pliego, más el reajuste y la
  inferencia. `return_train_score=True` puntúa además el pliego de entrenamiento,
  (k - 1) veces más grande que el de validación y sin cronometrar: se estima como
  (k - 1) veces el puntaje de validación.
- `aceleracion` = `t_cpu` / `t_entrenar`: el paralelismo que logró `GridSearchCV`
  con `n_jobs=-1` dentro de la tarea.
- `sobrecarga` (solo logs): pared entre el comienzo de dos tareas consecutivas menos
  el "Tomó" de la primera; arranque de Python, imports, carga del dataset y volcado
  del pickle.

Equipos: las infos de v5 escritas con scikit-learn 1.6 son las de la corrida de 2025,
que según la tesis corrió en una MacBook Air M1 (`m1_air`); las escritas con la
versión actual (variantes `_std`, recómputo exacto de kdc, fundacionales) y los logs
de v6 son de esta máquina (`m4_pro`).

Comandos, desde la raíz del repositorio:

- `minar`: una fila por tarea y clasificador en `sandbox/heroica/tiempos-tareas.csv`
  y una por tarea de los logs en `tiempos-logs.csv`. Es incremental.
- `resumir`: mediana, p90 y total por origen, equipo, clasificador y variante de
  dataset en `sandbox/heroica/tiempos.csv`.
- `medir`: corre clasificadores registrados sobre tareas de la tesis, guarda tiempos
  y métricas en `sandbox/heroica/mediciones.csv` y, si el clasificador es de la
  tesis, coteja R² y exactitud con v5 (control de reproducibilidad).
- `proyectar`: horas por campaña para 25, 64 y 100 semillas, en secuencial y en
  paralelo, en `sandbox/heroica/proyecciones.csv` y `opciones.csv`.

Uso:
    .venv/bin/python scripts/tiempos.py minar --procesos 4
    .venv/bin/python scripts/tiempos.py resumir
    .venv/bin/python scripts/tiempos.py proyectar
    nice -n 10 .venv/bin/python scripts/tiempos.py medir kdc --patron 'lunas_lo-1075-*'
"""

import logging
import os
import pickle
import re
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import sklearn
import typer
from sklearn.exceptions import ConvergenceWarning, InconsistentVersionWarning

from fkdc import campanias, config, dir_raiz, registro

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("tiempos")
logging.getLogger("fkdc.tarea").setLevel(logging.WARNING)

dir_heroica = dir_raiz / "sandbox/heroica"
dir_v5 = dir_raiz / "sandbox/v5"
ruta_tareas = dir_heroica / "tiempos-tareas.csv"
ruta_logs = dir_heroica / "tiempos-logs.csv"
ruta_resumen = dir_heroica / "tiempos.csv"
ruta_mediciones = dir_heroica / "mediciones.csv"
ruta_proyecciones = dir_heroica / "proyecciones.csv"
ruta_opciones = dir_heroica / "opciones.csv"

fuentes_infos = {
    "v5": dir_v5 / "infos",
    "v5_kdc_exacto": dir_v5 / "infos-kdc-fix",
    "fundacionales": dir_raiz / "sandbox/fundacionales/infos",
}
fuentes_logs = {
    "v5": (dir_raiz / "sandbox/logs-v5.out", "m1_air"),
    "v6_prioritario": (dir_raiz / "sandbox/v6-prioritario.log", "m4_pro"),
    "v6_resto": (dir_raiz / "sandbox/v6-resto.log", "m4_pro"),
}
ruta_log_paralelas = dir_raiz / "docs/sandbox/run-paralelas.log"
campos_nombre = ["dataset", "semilla_ds", "clf", "semilla_tarea", "puntuacion"]
semillas_tesis = 25
variantes_tesis = [
    *(
        f"{familia}_{sufijo}"
        for familia, sufijos in {
            "lunas": ("lo", "hi"),
            "circulos": ("lo", "hi"),
            "espirales": ("lo", "hi"),
            "eslabones": ("0", "12"),
            "helices": ("0", "12"),
            "hueveras": ("0", "12"),
            "pionono": ("0", "12"),
        }.items()
        for sufijo in sufijos
    ),
    "mnist",
    "anteojos",
    "iris",
    "vino",
    "pinguinos",
    "digitos",
    *(f"{base}_std" for base in config.bases_estandarizar),
]
# Los oráculos solo existen para las familias sintéticas con variedad conocida
variantes_oraculo = [v for v in variantes_tesis if v.split("_")[0] in {
    "lunas", "circulos", "espirales", "eslabones", "helices", "hueveras", "pionono",
    "anteojos",
}]  # fmt: skip
# Datasets nuevos de `fkdc.variedades` y su nombre en la corrida v6 (los paralelos
# de v6 usaban otra semilla de partición y no se reutilizan)
equivalencias_v6 = {
    "paralelas_2": "paralelas_2_q0",
    "paralelas_4": "paralelas_4_q0",
    "paralelas_8": "paralelas_8_q0",
    "nudos_200": "nudos_ralo",
    "nudos_800": "nudos",
    "lissajous": "lissajous",
    "toro_3": "toro3",
    "esfera_2": "esfera_2",
    "esfera_4": "esfera_4",
}
variantes_nuevas = list(equivalencias_v6)
# Variantes cuyas infos de v6 (100 semillas) tienen la misma tarea y la misma grilla
# que la campaña para fkdc, fkn y kn: sus `cv_results_` sirven tal cual y solo
# cambia la regla de reajuste (cf. `plan.md`).
variantes_con_cv_v6 = [
    *(v for v in variantes_tesis if not v.endswith("_std")),
    *(v for v in variantes_nuevas if not v.startswith("paralelas")),
]
familia_k_v6 = ("fkdc", "fkn", "kn")
# Grillas de ancho de banda de la campaña: la de kdc tiene un doceavo de década
# (133 valores) y la de fkdc es uno de cada tres, idéntica bit a bit a la de la tesis
h_kdc_campania = 10.0 ** (-5 + np.arange(133) / 12)
h_fkdc_campania = h_kdc_campania[::3]
proporcion_grilla_kdc = len(h_kdc_campania) / len(config.grillas["kdc"]["bandwidth"])

app = typer.Typer(add_completion=False, help=__doc__)


# =====================================================================
# Minería
# =====================================================================


def _partes(nombre: str) -> dict:
    """Campos del nombre `{dataset}-{semilla_ds}-{clf}-{semilla_tarea}-{puntuacion}`."""
    return dict(zip(campos_nombre, Path(nombre).stem.split("-"), strict=True))


def _equipo(origen: str, version: str) -> str:
    """Equipo en que corrió una info, según su origen y versión de scikit-learn."""
    if origen == "v5" and version.startswith("1.6"):
        return "m1_air"
    return "m4_pro"


def _cargar_con_version(ruta: Path):
    """Carga un pickle y devuelve (objeto, versión de scikit-learn que lo escribió)."""
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always", InconsistentVersionWarning)
        objeto = pickle.load(open(ruta, "rb"))
    versiones = {
        aviso.message.original_sklearn_version
        for aviso in avisos
        if issubclass(aviso.category, InconsistentVersionWarning)
    }
    return objeto, (versiones.pop() if versiones else sklearn.__version__)


def _tiempos_busqueda(busqueda) -> dict:
    """Tiempos y tamaño de la grilla a partir de `cv_results_`."""
    if busqueda is None:
        return {"n_candidatos": 0, "n_pliegues": 0}
    resultados = busqueda.cv_results_
    n_pliegues = int(busqueda.n_splits_)
    return {
        "n_candidatos": len(resultados["params"]),
        "n_pliegues": n_pliegues,
        "t_cv_ajuste": float(np.sum(resultados["mean_fit_time"]) * n_pliegues),
        "t_cv_puntaje": float(np.sum(resultados["mean_score_time"]) * n_pliegues),
        "t_reajuste": float(getattr(busqueda, "refit_time_", np.nan)),
    }


def _fila_info(argumentos: tuple[str, str]) -> dict:
    """Una fila de tiempos para el pickle de info `ruta` del origen `origen`."""
    origen, ruta = argumentos
    ruta = Path(ruta)
    fila = {"origen": origen, "archivo": ruta.name, **_partes(ruta.name)}
    fila["mb"] = ruta.stat().st_size / 2**20
    try:
        info, version = _cargar_con_version(ruta)
    except Exception as exc:  # un pickle ilegible no detiene la minería
        return {**fila, "error": f"{type(exc).__name__}: {exc}"}
    fila["version_sklearn"] = version
    fila["equipo"] = _equipo(origen, version)
    if "t_corrida" in info:  # pickles "slim" del recómputo exacto de kdc
        return {**fila, "t_tarea": info["t_corrida"], "con_probas": True}
    datos = info[fila["clf"]]
    t_entrenar = float(datos.get("t_entrenar", np.nan))
    t_evaluar = float(datos.get("t_evaluar", np.nan))
    con_probas = "probas" in datos
    t_probas = datos.get("t_probas")
    estimado = t_probas is None and con_probas
    t_probas = float(t_evaluar if estimado else (t_probas or 0.0))
    fila.update(
        t_entrenar=t_entrenar,
        t_evaluar=t_evaluar,
        t_probas=t_probas,
        t_probas_estimado=estimado,
        con_probas=con_probas,
        t_tarea=t_entrenar + t_evaluar + t_probas,
        **_tiempos_busqueda(datos.get("busqueda")),
    )
    if fila["n_candidatos"]:
        fila["t_cpu"] = (
            fila["t_cv_ajuste"]
            + fila["t_cv_puntaje"] * fila["n_pliegues"]
            + np.nan_to_num(fila["t_reajuste"])
            + t_evaluar
            + t_probas
        )
    else:
        fila["t_cpu"] = fila["t_tarea"]
    fila["aceleracion"] = fila["t_cpu"] / t_entrenar if fila["n_candidatos"] else 1.0
    return fila


def _inicializar_trabajador():
    """Silencia avisos ruidosos y registra `Dataset` donde lo buscan los pickles."""
    import __main__
    from fkdc.datasets import Dataset

    __main__.Dataset = Dataset
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning)


_patron_inicio = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) .*Leyendo config (\S+\.yaml)"
)
_patron_fin = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) .*- Tomó ([\d.]+)s")


def _instante(texto: str) -> float:
    return datetime.strptime(texto, "%Y-%m-%d %H:%M:%S,%f").timestamp()


def minar_log(ruta: Path, origen: str, equipo: str) -> pd.DataFrame:
    """Una fila por tarea de un log de corrida secuencial (v5, v6, fundacionales)."""
    filas, actual = [], None
    with open(ruta, errors="replace") as archivo:
        for linea in archivo:
            if inicio := _patron_inicio.match(linea):
                comienzo = _instante(inicio.group(1))
                if actual is not None:
                    actual["t_pared"] = comienzo - actual.pop("_comienzo")
                    filas.append(actual)
                actual = {
                    "origen": origen,
                    "equipo": equipo,
                    "archivo": Path(inicio.group(2)).name,
                    **_partes(inicio.group(2)),
                    "_comienzo": comienzo,
                }
            elif (fin := _patron_fin.match(linea)) and actual is not None:
                actual["t_tomo"] = float(fin.group(2))
    if actual is not None:
        actual.pop("_comienzo")
        filas.append(actual)
    logs = pd.DataFrame(filas)
    if not logs.empty:
        logs["sobrecarga"] = logs.t_pared - logs.t_tomo
    return logs


_patron_paralelas = re.compile(r"^\[\s*\d+s\] (\S+): .*\((\d+\.\d+)s\)\s*$")


def minar_log_paralelas(ruta: Path) -> pd.DataFrame:
    """Una fila por tarea del log de la familia paralelas de v6 (un solo proceso)."""
    filas = []
    with open(ruta, errors="replace") as archivo:
        for linea in archivo:
            if coincidencia := _patron_paralelas.match(linea):
                nombre, segundos = coincidencia.groups()
                filas.append(
                    {
                        "origen": "v6_paralelas",
                        "equipo": "m4_pro",
                        "archivo": f"{nombre}.yaml",
                        **_partes(f"{nombre}.yaml"),
                        "t_tomo": float(segundos),
                    }
                )
    return pd.DataFrame(filas)


@app.command()
def minar(
    procesos: Annotated[int, typer.Option(help="Procesos lectores en paralelo")] = 4,
    rehacer: Annotated[bool, typer.Option(help="Ignorar lo ya minado")] = False,
    incluir_v6: Annotated[
        bool, typer.Option(help="Leer también los 12 GB de infos de v6")
    ] = False,
):
    """Extrae los tiempos de las infos y los logs existentes."""
    dir_heroica.mkdir(parents=True, exist_ok=True)
    fuentes = dict(fuentes_infos)
    if incluir_v6:
        fuentes["v6"] = dir_raiz / "sandbox/v6/infos"
    previas = (
        pd.read_csv(ruta_tareas)
        if ruta_tareas.exists() and not rehacer
        else pd.DataFrame(columns=["origen", "archivo"])
    )
    vistas = set(zip(previas.origen, previas.archivo, strict=True))
    pendientes = [
        (origen, str(ruta))
        for origen, directorio in fuentes.items()
        if directorio.exists()
        for ruta in sorted(directorio.glob("*.pkl"))
        if (origen, ruta.name) not in vistas
    ]
    logger.info("%d infos ya minadas, %d pendientes", len(vistas), len(pendientes))
    if pendientes:
        t0 = time.time()
        with ProcessPoolExecutor(procesos, initializer=_inicializar_trabajador) as pool:
            nuevas = pd.DataFrame(pool.map(_fila_info, pendientes, chunksize=16))
        logger.info("Leídas en %.0fs", time.time() - t0)
        if "error" in nuevas and nuevas.error.notna().any():
            logger.warning("%d infos ilegibles", nuevas.error.notna().sum())
        tareas = pd.concat([previas, nuevas], ignore_index=True)
        tareas.to_csv(ruta_tareas, index=False)
        logger.info("Escribió %s (%d filas)", ruta_tareas, len(tareas))

    logs = [
        minar_log(ruta, origen, equipo)
        for origen, (ruta, equipo) in fuentes_logs.items()
        if ruta.exists()
    ]
    for ruta in sorted((dir_raiz / "sandbox/fundacionales/logs").glob("*.log")):
        logs.append(minar_log(ruta, f"fundacionales/{ruta.stem}", "m4_pro"))
    if ruta_log_paralelas.exists():
        logs.append(minar_log_paralelas(ruta_log_paralelas))
    logs = pd.concat(logs, ignore_index=True)
    logs.to_csv(ruta_logs, index=False)
    logger.info("Escribió %s (%d tareas)", ruta_logs, len(logs))


# =====================================================================
# Resumen
# =====================================================================


def _p90(serie: pd.Series) -> float:
    return serie.quantile(0.9)


def _cargar_tareas() -> pd.DataFrame:
    """Tiempos por tarea minados, sin filas ilegibles."""
    if not ruta_tareas.exists():
        raise typer.BadParameter("Falta minar: corra `tiempos.py minar` primero")
    tareas = pd.read_csv(ruta_tareas)
    if "error" in tareas:
        tareas = tareas[tareas.error.isna()]
    return tareas


@app.command()
def resumir():
    """Mediana, p90 y total por origen, equipo, clasificador y variante."""
    tareas = _cargar_tareas()
    claves = ["origen", "equipo", "clf", "dataset"]
    infos = (
        tareas.groupby(claves)
        .agg(
            n=("t_tarea", "size"),
            t_mediana=("t_tarea", "median"),
            t_p90=("t_tarea", _p90),
            t_total=("t_tarea", "sum"),
            t_entrenar_mediana=("t_entrenar", "median"),
            t_inferencia_mediana=("t_probas", "median"),
            t_cpu_mediana=("t_cpu", "median"),
            aceleracion_mediana=("aceleracion", "median"),
            n_candidatos=("n_candidatos", "median"),
            mb_mediana=("mb", "median"),
        )
        .reset_index()
        .assign(fuente="infos")
    )
    partes = [infos]
    if ruta_logs.exists():
        logs = pd.read_csv(ruta_logs)
        partes.append(
            logs.groupby(claves)
            .agg(
                n=("t_tomo", "size"),
                t_mediana=("t_tomo", "median"),
                t_p90=("t_tomo", _p90),
                t_total=("t_tomo", "sum"),
                sobrecarga_mediana=("sobrecarga", "median"),
                t_pared_total=("t_pared", "sum"),
            )
            .reset_index()
            .assign(fuente="log")
        )
    resumen = pd.concat(partes, ignore_index=True)
    columnas = ["fuente", *claves]
    resumen = resumen[columnas + [c for c in resumen if c not in columnas]]
    resumen.sort_values(columnas).to_csv(ruta_resumen, index=False, float_format="%.4g")
    logger.info("Escribió %s (%d filas)", ruta_resumen, len(resumen))

    tesis = tareas[tareas.origen.eq("v5") & tareas.clf.isin(config.clasificadores)]
    por_clf = tesis.groupby(["equipo", "clf"]).agg(
        tareas=("t_tarea", "size"),
        mediana_s=("t_tarea", "median"),
        p90_s=("t_tarea", _p90),
        total_h=("t_tarea", lambda s: s.sum() / 3600),
        aceleracion=("aceleracion", "median"),
    )
    print(por_clf.round(2).to_markdown())
    if ruta_logs.exists():
        corridas = logs.groupby("origen").agg(
            tareas=("t_tomo", "size"),
            tomo_h=("t_tomo", lambda s: s.sum() / 3600),
            pared_h=("t_pared", lambda s: s.sum() / 3600),
            sobrecarga_mediana_s=("sobrecarga", "median"),
        )
        print(corridas.round(2).to_markdown())


# =====================================================================
# Medición
# =====================================================================


def _usar_cache_temporal(directorio: Path):
    """Redirige el caché en disco de `fkdc.fermat` a `directorio` (solo este proceso).

    Con `n_jobs != 1` los workers de loky reimportan `fkdc.fermat` y vuelven al caché
    global, cuyos aciertos de corridas previas abaratan la medición.
    """
    from joblib import Memory

    from fkdc import fermat

    memoria = Memory(directorio, verbose=0)
    fermat.euclidean = memoria.cache(fermat.euclidean.func)
    fermat.fermat_muestral = memoria.cache(fermat.fermat_muestral.func)


@app.command()
def medir(
    clasificadores: Annotated[list[str], typer.Argument(help="Nombres registrados")],
    patron: Annotated[
        list[str], typer.Option(help="Glob sobre infos de v5 que define las tareas")
    ] = ("*-1075-*",),
    nuevos: Annotated[
        list[str] | None,
        typer.Option(help="Glob sobre los datasets nuevos (`fkdc.variedades`)"),
    ] = None,
    n_jobs: Annotated[int, typer.Option(help="n_jobs de GridSearchCV")] = 1,
    cache_temporal: Annotated[
        bool, typer.Option(help="Medir con el caché de fkdc.fermat vacío")
    ] = True,
):
    """Mide clasificadores sobre tareas de la tesis y coteja con v5 si corresponde.

    Las tareas son las (variante, semilla del dataset, semilla de la tarea) de las
    infos de `sandbox/v5/infos` que coinciden con `patron`, con los datasets de
    `sandbox/v5/datasets`, más las de los datasets de `sandbox/fundacionales/datasets`
    que coinciden con `nuevos` (semilla de partición: la principal). Con `n_jobs=1`
    (por omisión) `t_entrenar` ya es el costo secuencial que usa la proyección en
    paralelo. Usa las grillas del registro (`fkdc.registro`), no las de la campaña.
    """
    from fkdc.datasets import Dataset
    from fkdc.tarea import Tarea

    _inicializar_trabajador()
    if cache_temporal:
        if n_jobs != 1:
            logger.warning("Con n_jobs != 1 los workers usan el caché global")
        _usar_cache_temporal(Path(tempfile.mkdtemp(prefix="fkdc-cache-")))
    rutas = {r for p in patron for r in (dir_v5 / "infos").glob(f"{p}.pkl")}
    tareas = set()
    for partes in map(_partes, (r.name for r in rutas)):
        base = partes["dataset"].removesuffix("_std")
        semilla_ds = partes["semilla_ds"]
        nombre_ds = base if semilla_ds == "None" else f"{base}-{semilla_ds}"
        ruta_ds = dir_v5 / "datasets" / f"{nombre_ds}.pkl"
        tareas.add((partes["dataset"], semilla_ds, partes["semilla_tarea"], ruta_ds))
    dir_nuevos = dir_raiz / "sandbox/fundacionales/datasets"
    for ruta_ds in {r for p in nuevos or [] for r in dir_nuevos.glob(f"{p}.pkl")}:
        variante, semilla_ds = ruta_ds.stem.split("-")
        tareas.add((variante, semilla_ds, str(config.semilla_principal), ruta_ds))
    tareas = sorted(tareas)
    logger.info("%d tareas × %d clasificadores", len(tareas), len(clasificadores))
    filas = []
    for variante, semilla_ds, semilla_tarea, ruta_ds in tareas:
        dataset = Dataset.cargar(ruta_ds)
        for nombre in clasificadores:
            estimador, grilla, puntuacion = registro.preparar(
                nombre,
                dataset.n,
                dataset.p,
                variante=variante,
                semilla=int(semilla_tarea),
            )
            tarea = Tarea(
                dataset,
                {nombre: (estimador, grilla)},
                scoring=puntuacion,
                cv=config.cv,
                split_evaluacion=config.split_evaluacion,
                semilla=int(semilla_tarea),
                estandarizar=variante.endswith("_std"),
                n_jobs=n_jobs,
            )
            t0 = time.time()
            tarea.evaluar()
            datos = tarea.info[nombre]
            fila = {
                "dataset": variante,
                "semilla_ds": semilla_ds,
                "clf": nombre,
                "semilla_tarea": semilla_tarea,
                "puntuacion": puntuacion,
                "n_jobs": n_jobs,
                "t_pared": time.time() - t0,
                "t_entrenar": datos.get("t_entrenar"),
                "t_evaluar": datos.get("t_evaluar"),
                "t_probas": datos.get("t_probas", 0.0),
                "r2": datos.get("r2"),
                "accuracy": datos.get("accuracy"),
                **_tiempos_busqueda(datos.get("busqueda")),
                "momento": datetime.now().isoformat(timespec="seconds"),
            }
            ruta_v5 = (
                dir_v5
                / "infos"
                / f"{variante}-{semilla_ds}-{nombre}-{semilla_tarea}-{puntuacion}.pkl"
            )
            if ruta_v5.exists():
                datos_v5 = _cargar_con_version(ruta_v5)[0][nombre]
                fila["r2_v5"] = datos_v5.get("r2")
                fila["accuracy_v5"] = datos_v5.get("accuracy")
                fila["coincide_v5"] = bool(
                    np.isclose(fila["accuracy"], fila["accuracy_v5"], atol=1e-12)
                    and (
                        fila["r2"] is None
                        or np.isclose(fila["r2"], fila["r2_v5"], atol=1e-9)
                    )
                )
            logger.info(
                "%s-%s %s: %.1fs, R²=%s%s",
                variante,
                semilla_ds if semilla_ds != "None" else semilla_tarea,
                nombre,
                fila["t_pared"],
                f"{fila['r2']:.4f}" if fila["r2"] is not None else "-",
                f" (v5 {'=' if fila.get('coincide_v5') else '≠'})"
                if "coincide_v5" in fila
                else "",
            )
            filas.append(fila)
    nuevas = pd.DataFrame(filas)
    if ruta_mediciones.exists():
        nuevas = pd.concat([pd.read_csv(ruta_mediciones), nuevas], ignore_index=True)
    dir_heroica.mkdir(parents=True, exist_ok=True)
    nuevas.to_csv(ruta_mediciones, index=False)
    logger.info("Escribió %s", ruta_mediciones)


# =====================================================================
# Proyección
# =====================================================================

# Costo secuencial supuesto (`t_cpu`, segundos por tarea de ~400 observaciones de
# entrenamiento en este equipo) de los clasificadores aún sin tiempos medidos:
# (bajo, central, alto, carril). Salen del tamaño de cada grilla de
# `fkdc.compendio.grillas` por 5 pliegos y de un costo por ajuste del orden conocido
# para cada estimador; `medir` los reemplaza en cuanto hay mediciones. El carril
# "torch" va en serie, en un proceso con 4 hilos (TabPFN, por MPS).
supuestos = {
    "qda": (0.2, 0.5, 2, "cpu"),  # 37 × 5 ajustes de milisegundos
    "lda": (0.2, 0.5, 2, "cpu"),
    "gmm": (3, 10, 40, "cpu"),  # 48 × 5 EM por clase
    "kdc_abramson": (2, 8, 30, "cpu"),  # 136 × 5 KDE exactas adaptativas
    "isomap_kn": (10, 40, 200, "cpu"),  # 20 Isomap memorizados + ≤ 800 × 5 kn
    "isomap_kdc": (20, 80, 400, "cpu"),  # 20 Isomap + ≤ 1800 × 5 KDE
    "nca_kn": (30, 150, 800, "cpu"),  # ≤ 7 NCA por pliego
    "label_spreading": (5, 15, 60, "cpu"),  # 133 × 5
    "gpc": (5, 30, 300, "cpu"),  # sin grilla; O(n³) y uno contra todos si K > 2
    "rf": (40, 100, 300, "cpu"),  # 16 × 5 bosques de 500 árboles
    "et": (20, 60, 200, "cpu"),
    "mlp": (50, 150, 600, "cpu"),  # 21 × 5 L-BFGS de hasta 5000 iteraciones
    "realmlp": (5, 20, 60, "torch"),
    "tabm": (10, 40, 120, "torch"),
    "tabicl": (1, 4, 15, "torch"),
    "tabdpt": (2, 8, 30, "torch"),
    "tabpfn": (2, 6, 20, "torch"),
    "tabpfn_v2": (0.2, 0.4, 1, "torch"),
}
# Grupos de la proyección: los ocho de la tesis y los conjuntos de `fkdc.campanias`
grupos = {
    "tesis": list(config.clasificadores),
    **{
        nombre: campanias.conjuntos[nombre]
        for nombre in ("compendio", "fundacionales", "compendio_caro", "profundos")
    },
}
escenarios_semillas = (25, 64, 100)
variantes_3d = ["eslabones_0", "helices_0", "hueveras_0", "pionono_0"]


def _factores_equipo(tareas: pd.DataFrame, logs: pd.DataFrame) -> pd.Series:
    """Cociente m4_pro / m1_air del tiempo por tarea, por clasificador.

    Para fkdc, fkn y kn (mismas grillas en v5 y v6) compara los "Tomó" de los logs
    de v5 (m1_air) y v6 (m4_pro): mediana por dataset del cociente de medianas.
    Para los demás compara las infos de v5 sobre los datos crudos (m1_air) con las
    variantes `_std` (m4_pro) de los cuatro datasets estandarizados. Si ninguna
    comparación es posible, usa la mediana de los factores obtenidos.
    """
    factores = {}
    v5 = logs[logs.origen.eq("v5")]
    v6 = logs[logs.origen.isin(["v6_prioritario", "v6_resto"])]
    for clf in familia_k_v6:
        cocientes = (
            v6[v6.clf.eq(clf)].groupby("dataset").t_tomo.median()
            / v5[v5.clf.eq(clf)].groupby("dataset").t_tomo.median()
        ).dropna()
        if len(cocientes):
            factores[clf] = cocientes.median()
    v5_infos = tareas[tareas.origen.eq("v5")]
    for clf in config.clasificadores:
        if clf in factores:
            continue
        de_clf = v5_infos[v5_infos.clf.eq(clf)]
        crudos = de_clf[de_clf.equipo.eq("m1_air")].groupby("dataset").t_tarea.median()
        estandar = de_clf[de_clf.dataset.str.endswith("_std")]
        estandar = estandar.groupby(estandar.dataset.str.removesuffix("_std"))
        cocientes = (estandar.t_tarea.median() / crudos).dropna()
        if len(cocientes):
            factores[clf] = cocientes.median()
    factores = pd.Series(factores, name="factor_m4_sobre_m1", dtype=float)
    for clf in config.clasificadores:
        if clf not in factores:
            factores[clf] = factores.median()
    return factores


def _costos(
    tareas: pd.DataFrame,
    logs: pd.DataFrame,
    factores: pd.Series,
    reutilizar: bool = False,
) -> pd.DataFrame:
    """Costo medio por tarea en este equipo, por clasificador y variante.

    Columnas: `t_secuencial` (pared con `n_jobs=-1` dentro de la tarea, como
    corrieron la tesis y v6), `t_cpu` (costo con `n_jobs=1`, para el paralelismo entre
    tareas), `t_reajuste` (reajuste y evaluación de otra selección sobre una grilla ya
    evaluada), `mb` (tamaño de la info), `hechas` (semillas de las 25 de la tesis cuyas
    infos sirven tal cual), `cv_v6` (si las infos de v6 tienen sus `cv_results_` para
    100 semillas), `fuente` y `carril`. Usa medias: se proyectan totales.

    kdc cambia de grilla en la campaña (133 anchos de banda en lugar de 136), así que
    no reutiliza ninguna semilla; `kdc_exacto` es el mismo kdc con `EuclideanKDE`.
    """
    v5 = tareas[tareas.origen.eq("v5") & tareas.clf.isin(config.clasificadores)].copy()
    en_m1 = v5.equipo.eq("m1_air")
    v5["t_m4"] = v5.t_tarea * np.where(en_m1, v5.clf.map(factores).astype(float), 1.0)
    v5["t_reajuste_m4"] = (
        v5.t_reajuste.fillna(0) + v5.t_evaluar + v5.t_probas
    ) * np.where(en_m1, v5.clf.map(factores).astype(float), 1.0)
    aceleracion = (
        v5[v5.equipo.eq("m4_pro")].groupby("clf").aceleracion.median().clip(lower=1)
    )
    mb_clf = v5.groupby("clf").mb.mean()
    reajuste_clf = v5.groupby("clf").t_reajuste_m4.mean()
    filas = []

    def agregar(clf, variante, t_seq, fuente, **extra):
        filas.append(
            {
                "clf": clf,
                "dataset": variante,
                "t_secuencial": t_seq,
                "t_cpu": extra.pop("t_cpu", t_seq * aceleracion.get(clf, 1.0)),
                "t_reajuste": reajuste_clf.get(clf.removesuffix("_exacto"), 0.0),
                "mb": extra.pop("mb", mb_clf.get(clf.removesuffix("_exacto"), 0.2)),
                "fuente": fuente,
                **extra,
            }
        )

    # Los ocho de la tesis sobre sus 24 variantes
    medias_v5 = v5.groupby(["clf", "dataset"]).t_m4.mean()
    for (clf, variante), t_seq in medias_v5.items():
        if clf == "kdc":
            t_seq *= proporcion_grilla_kdc
        hechas = semillas_tesis if reutilizar and clf != "kdc" else 0
        agregar(clf, variante, t_seq, "medido", hechas=hechas)
    exacto = tareas[tareas.origen.eq("v5_kdc_exacto")].groupby("dataset").t_tarea.mean()
    for variante in variantes_tesis:
        t_seq = exacto.get(variante.removesuffix("_std"), np.nan)
        agregar(
            "kdc_exacto",
            variante,
            t_seq * proporcion_grilla_kdc,
            "medido",
            t_cpu=t_seq * proporcion_grilla_kdc * aceleracion.get("kdc", 1.0),
            hechas=0,
        )

    # Los ocho de la tesis sobre los datasets nuevos: fkdc, fkn y kn medidos en v6
    # (mismas grillas); el resto, como en las variantes 3D de la tesis
    v6 = logs[logs.origen.str.startswith("v6")]
    medias_v6 = v6.groupby(["clf", "dataset"]).t_tomo.mean()
    en_3d = medias_v5.loc[(slice(None), variantes_3d)].groupby("clf").mean()
    for variante, nombre_v6 in equivalencias_v6.items():
        escala_n = 0.25 if variante == "nudos_200" else 1.0
        for clf in [*config.clasificadores, "kdc_exacto"]:
            if clf in familia_k_v6 and (clf, nombre_v6) in medias_v6:
                t_seq, fuente = medias_v6[(clf, nombre_v6)], "medido (v6)"
            elif clf == "kdc_exacto":
                t_seq = exacto.reindex(variantes_3d).mean() * proporcion_grilla_kdc
                t_seq, fuente = t_seq * escala_n, "análogo 3D"
            else:
                t_seq = en_3d[clf] * (proporcion_grilla_kdc if clf == "kdc" else 1)
                t_seq, fuente = t_seq * escala_n, "análogo 3D"
            agregar(clf, variante, t_seq, fuente, hechas=0)

    # Clasificadores nuevos medidos (infos de fundacionales y `medir` con n_jobs=1)
    medidos = tareas[tareas.origen.eq("fundacionales")][
        ["clf", "dataset", "t_tarea", "t_cpu", "mb"]
    ]
    if ruta_mediciones.exists():
        mediciones = pd.read_csv(ruta_mediciones)
        mediciones = mediciones[mediciones.n_jobs.eq(1)].assign(
            t_tarea=lambda m: m.t_pared, t_cpu=lambda m: m.t_pared, mb=np.nan
        )
        medidos = pd.concat(
            [medidos, mediciones[["clf", "dataset", "t_tarea", "t_cpu", "mb"]]]
        )
    conteos = (
        tareas[tareas.origen.eq("fundacionales")].groupby(["clf", "dataset"]).size()
    )
    nuevos = [clf for clfs in grupos.values() for clf in clfs]
    nuevos = [clf for clf in nuevos if clf not in config.clasificadores]
    for clf in nuevos:
        de_clf = medidos[medidos.clf.eq(clf)]
        carril = registro.entrada(clf).carril
        if de_clf.empty:
            bajo, central, alto, _ = supuestos[clf]
            for variante in [*variantes_tesis, *variantes_nuevas]:
                agregar(
                    clf,
                    variante,
                    central if carril == "torch" else central / 8,
                    "supuesto",
                    t_cpu=central,
                    t_cpu_bajo=bajo,
                    t_cpu_alto=alto,
                    carril=carril,
                    hechas=0,
                    mb=0.2,
                )
            continue
        por_variante = de_clf.groupby("dataset")[["t_tarea", "t_cpu"]].mean()
        tipico = por_variante.median()
        for variante in [*variantes_tesis, *variantes_nuevas]:
            medido = variante in por_variante.index
            valores = por_variante.loc[variante] if medido else tipico
            agregar(
                clf,
                variante,
                valores.t_tarea,
                "medido" if medido else "medido (extrapolado)",
                t_cpu=valores.t_cpu,
                carril=carril,
                # Solo cuentan las variantes con las 25 semillas de la tesis completas
                hechas=semillas_tesis
                if reutilizar and conteos.get((clf, variante), 0) >= semillas_tesis
                else 0,
                mb=de_clf.mb.mean() if de_clf.mb.notna().any() else 0.2,
            )
    costos = pd.DataFrame(filas)
    costos["carril"] = costos.get("carril", pd.Series(dtype=object)).fillna("cpu")
    for columna in ("t_cpu_bajo", "t_cpu_alto"):
        if columna not in costos:
            costos[columna] = np.nan
        costos[columna] = costos[columna].fillna(costos.t_cpu)
    costos["cv_v6"] = costos.clf.isin(familia_k_v6) & costos.dataset.isin(
        variantes_con_cv_v6
    )
    costos["nuevo_dataset"] = costos.dataset.isin(variantes_nuevas)
    return costos


def _nucleos_con_torch(capacidad: float, hilos_torch: int = 4) -> float:
    """Capacidad del carril de CPU mientras el de torch ocupa `hilos_torch` núcleos."""
    return min(capacidad, (os.cpu_count() or 14) - hilos_torch)


def _escenario(
    costos: pd.DataFrame,
    n_semillas: int,
    reutilizar_v6: bool,
    sobrecarga: float,
    capacidad: float,
) -> pd.DataFrame:
    """Horas por clasificador y variante para `n_semillas`, con o sin reutilizar v6.

    Las semillas ya hechas (`hechas`) no se calculan. Con `reutilizar_v6`, fkdc, fkn y
    kn no recorren la grilla en las variantes con `cv_v6`: solo se reajusta la
    selección de la tesis sobre los `cv_results_` de v6 en las semillas que faltan.

    En paralelo, cada tarea de CPU cuesta `min(t_cpu / capacidad, t_secuencial)`: las
    grillas grandes de fkdc y fkn ya saturan el equipo con `n_jobs=-1`, y su `t_cpu`
    está inflado (los workers de loky recalculan en simultáneo lo que el caché en
    disco de `fkdc.fermat` todavía no guardó), así que no ganan nada. Las del carril
    "torch" van en serie; `h_paralelo_con_torch` deja 4 núcleos para ese carril.
    """
    calcular = (n_semillas - costos.hechas).clip(lower=0)
    derivadas = reutilizar_v6 & costos.cv_v6
    reajustar = np.where(derivadas, calcular, 0)
    calcular = np.where(derivadas, 0, calcular)
    resultado = costos[["clf", "dataset", "carril", "fuente", "nuevo_dataset"]].copy()
    resultado["tareas"] = calcular
    resultado["reajustes"] = reajustar
    resultado["h_secuencial"] = (
        calcular * (costos.t_secuencial + sobrecarga) + reajustar * costos.t_reajuste
    ) / 3600
    en_torch = costos.carril.eq("torch")
    for sufijo in ("", "_bajo", "_alto"):
        t_cpu = costos[f"t_cpu{sufijo}"]
        resultado[f"h_cpu{sufijo}"] = (
            calcular * t_cpu + reajustar * costos.t_reajuste
        ) / 3600
        for nombre, nucleos in (
            ("h_paralelo", capacidad),
            ("h_paralelo_con_torch", _nucleos_con_torch(capacidad)),
        ):
            por_tarea = np.where(
                en_torch, t_cpu, np.minimum(t_cpu / nucleos, costos.t_secuencial)
            )
            resultado[f"{nombre}{sufijo}"] = (
                calcular * por_tarea + reajustar * costos.t_reajuste / nucleos
            ) / 3600
    resultado["gb_infos"] = calcular * costos.mb / 1024
    return resultado


@app.command()
def proyectar(
    capacidad: Annotated[
        float, typer.Option(help="Núcleos efectivos para tareas simultáneas")
    ] = 10.0,
    reutilizar: Annotated[
        bool, typer.Option(help="Reutilizar las 25 semillas ya corridas")
    ] = False,
):
    """Horas por campaña para 25/64/100 semillas, en secuencial y en paralelo.

    - Secuencial: una tarea por vez con `n_jobs=-1` adentro, más la sobrecarga de un
      proceso por tarea (como corrieron la tesis y v6, vía Make).
    - Paralelo: tareas simultáneas con `n_jobs=1` en procesos que corren muchas
      tareas cada uno (sin sobrecarga de arranque). La capacidad por omisión, 10, es
      la de `fkdc.correr` con 16 GB para la corrida y 1,5 GB por proceso (el M4 Pro
      tiene 10 núcleos de rendimiento y 4 de eficiencia); el p90 de la aceleración
      observada se informa como control.

    Por omisión no reutiliza nada (decisión del prerregistro: todo se recalcula).
    """
    tareas = _cargar_tareas()
    logs = pd.read_csv(ruta_logs)
    factores = _factores_equipo(tareas, logs)
    costos = _costos(tareas, logs, factores, reutilizar)
    m4 = tareas[tareas.equipo.eq("m4_pro") & tareas.clf.isin(config.clasificadores)]
    m4 = m4[(m4.n_candidatos * m4.n_pliegues).ge(1000)]
    sobrecarga = float(
        logs[logs.origen.isin(["v6_prioritario", "v6_resto"])].sobrecarga.median()
    )
    logger.info(
        "Factores m4/m1: %s", ", ".join(f"{c}={f:.2f}" for c, f in factores.items())
    )
    logger.info(
        "Capacidad: %.1f núcleos (aceleración observada en grillas grandes: p90 %.1f)"
        "; sobrecarga por proceso: %.2fs",
        capacidad,
        m4.aceleracion.quantile(0.9),
        sobrecarga,
    )
    costos.to_csv(
        dir_heroica / "costos-por-tarea.csv", index=False, float_format="%.4g"
    )

    escenarios = []
    for n_semillas in escenarios_semillas:
        for reutilizar_v6 in (False, True):
            escenario = _escenario(
                costos, n_semillas, reutilizar_v6, sobrecarga, capacidad
            )
            escenarios.append(
                escenario.assign(n_semillas=n_semillas, reutiliza_v6=reutilizar_v6)
            )
    escenarios = pd.concat(escenarios, ignore_index=True)
    escenarios["grupo"] = escenarios.clf.map(
        {clf: g for g, clfs in grupos.items() for clf in clfs} | {"kdc_exacto": "tesis"}
    )
    columnas = [
        "tareas",
        "reajustes",
        "h_secuencial",
        "h_cpu",
        "h_paralelo",
        "h_paralelo_bajo",
        "h_paralelo_alto",
        "gb_infos",
    ]
    proyecciones = (
        escenarios.groupby(
            ["grupo", "clf", "carril", "n_semillas", "reutiliza_v6", "nuevo_dataset"]
        )[columnas]
        .sum()
        .reset_index()
    )
    proyecciones.to_csv(ruta_proyecciones, index=False, float_format="%.4g")
    logger.info("Escribió %s", ruta_proyecciones)
    tesis = proyecciones[proyecciones.grupo.eq("tesis") & ~proyecciones.nuevo_dataset]
    print(
        tesis[tesis.n_semillas.eq(100) & ~tesis.reutiliza_v6]
        .drop(columns=["grupo", "carril", "nuevo_dataset"])
        .round(2)
        .to_markdown(index=False)
    )
    opciones = _tabla_opciones(escenarios, costos, capacidad)
    opciones.to_csv(ruta_opciones, index=False, float_format="%.3g")
    logger.info("Escribió %s", ruta_opciones)
    print(
        opciones[
            opciones.kdc.eq("ambos")
            & opciones.protocolo.eq("tesis+v6")
            & ~opciones.reutiliza_v6
        ]
        .round(1)
        .to_markdown(index=False)
    )


def _tabla_opciones(
    escenarios: pd.DataFrame, costos: pd.DataFrame, capacidad: float
) -> pd.DataFrame:
    """Horas de pared por opción.

    Ejes: semillas (64, 100) × clasificadores (tesis, + compendio, + profundos,
    + TabPFN) × datasets (las 24 variantes de la tesis, + los 9 nuevos) × kdc
    (sklearn como en la tesis, exacto) × protocolo (tesis; tesis+v6, que deriva a
    posteriori la selección ciega a α de v6 sobre los mismos `cv_results_` y paga un
    reajuste por tarea de fkdc, fkn y kdc) × reutilizar los `cv_results_` de v6.
    En paralelo, los carriles de CPU y de torch corren a la vez (pared = el mayor).
    """
    conjuntos = {
        "tesis": ["tesis"],
        "+ compendio": ["tesis", "compendio"],
        "+ compendio + fundacionales": ["tesis", "compendio", "fundacionales"],
        "+ todo lo local": [
            "tesis",
            "compendio",
            "fundacionales",
            "compendio_caro",
            "profundos",
        ],
    }
    reajuste = costos.set_index(["clf", "dataset"]).t_reajuste
    filas = []
    for (n_semillas, reutilizar_v6), escenario in escenarios.groupby(
        ["n_semillas", "reutiliza_v6"]
    ):
        if n_semillas == semillas_tesis:
            continue
        for nombre, nombres_grupos in conjuntos.items():
            for datasets in ("tesis (24)", "+ nuevos (9)"):
                base = escenario[escenario.grupo.isin(nombres_grupos)]
                if datasets == "tesis (24)":
                    base = base[~base.nuevo_dataset]
                for kdc in ("sklearn", "exacto", "ambos"):
                    excluido = {"exacto": "kdc", "sklearn": "kdc_exacto"}.get(kdc)
                    elegidos = base[base.clf.ne(excluido)]
                    cpu = elegidos[elegidos.carril.eq("cpu")]
                    serie = elegidos[elegidos.carril.eq("torch")]
                    con_torch = bool(serie.tareas.sum())
                    nucleos = _nucleos_con_torch(capacidad) if con_torch else capacidad
                    sufijo_cpu = "h_paralelo_con_torch" if con_torch else "h_paralelo"
                    familia = cpu[cpu.clf.isin(["fkdc", "fkn", "kdc", "kdc_exacto"])]
                    claves = list(zip(familia.clf, familia.dataset, strict=True))
                    s_reajuste = reajuste.reindex(claves).fillna(0).sum()
                    for protocolo in ("tesis", "tesis+v6"):
                        h_extra = (
                            n_semillas * s_reajuste / 3600
                            if protocolo == "tesis+v6"
                            else 0.0
                        )
                        fila = {
                            "semillas": n_semillas,
                            "clasificadores": nombre,
                            "datasets": datasets,
                            "kdc": kdc,
                            "protocolo": protocolo,
                            "reutiliza_v6": reutilizar_v6,
                            "tareas": int(elegidos.tareas.sum()),
                            "reajustes": int(elegidos.reajustes.sum()),
                            "h_secuencial": cpu.h_secuencial.sum()
                            + h_extra
                            + serie.h_secuencial.sum(),
                        }
                        for extremo in ("", "_bajo", "_alto"):
                            fila[f"h_paralelo{extremo}"] = max(
                                cpu[f"{sufijo_cpu}{extremo}"].sum() + h_extra / nucleos,
                                serie[f"h_paralelo{extremo}"].sum(),
                            )
                        fila["h_carril_cpu"] = cpu[sufijo_cpu].sum() + h_extra / nucleos
                        fila["h_carril_torch"] = serie.h_paralelo.sum()
                        fila["gb_infos"] = elegidos.gb_infos.sum()
                        fila["con_supuestos"] = bool(
                            elegidos.fuente.eq("supuesto").any()
                        )
                        filas.append(fila)
    return pd.DataFrame(filas)


if __name__ == "__main__":
    app()
