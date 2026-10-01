"""
Corredor de campañas: evalúa las configs de una campaña que todavía no tienen info.

`ejecutar` corre una config (la misma unidad que `fkdc/process.py`): instancia el
clasificador desde `fkdc.registro`, arma la `Tarea` con la regla de reajuste de la
config, la evalúa, deriva las selecciones adicionales sobre los mismos `cv_results_`
(`fkdc.seleccion`), calibra por temperatura las probabilidades (`fkdc.calibracion`),
descarta el modelo reajustado si la config lo pide y escribe el info de forma atómica
(nunca queda uno a medias).

`correr` reparte las configs pendientes de un carril en lotes, uno por tarea
(variante, semilla del dataset, semilla de la partición), de modo que fkdc y fkn de una
misma tarea compartan el caché de distancias de Fermat. Los lotes van en orden de
repetición: una corrida interrumpida deja todas las variantes con la misma cantidad de
semillas. Carriles (`fkdc.registro`):

- "cpu": muchos procesos con `n_jobs=1` y un hilo de BLAS/OpenMP cada uno; cuántos,
  según los núcleos y la memoria (`--memoria-gb` / `--gb-por-proceso`).
- "torch": un solo proceso con `--hilos` hilos (4 por omisión, como los barridos de los
  profundos); TabPFN usa MPS.

Los dos carriles pueden correr a la vez (dos comandos): cada uno tiene su caché, su
registro (`{dir}/registro-{carril}.csv`, con la memoria máxima de cada proceso) y sus
fallas (`{dir}/fallas-{carril}.csv`), que no detienen la corrida.

Make corre la campaña por lotes (`fkdc.correr lote`, una receta por lote; ver
`fkdc.campanias.escribir_lotes`): repetir el mismo `make` tras un corte saltea los
lotes terminados y rehace lo que falta de los demás. `fkdc.correr campania` corre lo
pendiente sin Make, con un grupo de procesos propio.

Uso, desde la raíz del repositorio:
    make -k -j10 correr-cpu          # y, a la par, make -k correr-torch
    .venv/bin/python -m fkdc.correr campania heroica --carril cpu --en-seco
"""

import logging
import os
import pickle
import platform
import resource
import shutil
import sys
import time
import traceback
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import scipy
import sklearn
import typer
import yaml
from sklearn.utils import Bunch

from fkdc import calibracion, campanias, dir_raiz, registro, seleccion
from fkdc.datasets import Dataset
from fkdc.tarea import Tarea

logger = logging.getLogger("fkdc.correr")
campos_nombre = ["variante", "semilla_ds", "clf", "semilla_tarea", "puntuacion"]


def _entorno() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "equipo": platform.machine(),
    }


def _memoria_maxima_mb() -> float:
    """Memoria residente máxima de este proceso (bytes en macOS, KiB en Linux)."""
    maxima = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return maxima / 2**20 if sys.platform == "darwin" else maxima / 2**10


def _templar(tarea: Tarea, probas, clases) -> dict:
    """R² templado de `probas` sobre la evaluación de `tarea` (`fkdc.calibracion`)."""
    frecuencias = pd.Series(tarea.y_entr).value_counts(normalize=True)
    return calibracion.templar(
        probas, tarea.y_eval, clases, frecuencias.reindex(clases).fillna(0).values
    )


def ejecutar(ruta_config: Path, dir_infos: Path, n_jobs: int = 1) -> dict:
    """Evalúa una config, escribe `{dir_infos}/{stem}.pkl` y devuelve su registro."""
    t_inicio = time.time()
    ruta_config = Path(ruta_config)
    dir_infos = Path(dir_infos)
    with open(ruta_config) as archivo:
        cfg = yaml.safe_load(archivo)
    nombre = cfg["clasificador"]
    variante = cfg.get("variante", ruta_config.stem.split("-")[0])
    ruta_dataset = Path(cfg["dataset"])
    dataset = Dataset.cargar(
        ruta_dataset if ruta_dataset.is_absolute() else dir_raiz / ruta_dataset
    )
    estimador, _, _ = registro.preparar(
        nombre, dataset.n, dataset.p, variante=variante, semilla=cfg["seed"], grilla={}
    )
    tarea = Tarea(
        dataset,
        {nombre: (estimador, cfg.get("grilla_hipers", {}))},
        scoring=cfg["scoring"],
        refit=seleccion.reglas[cfg.get("reajuste", "parsimoniosa")],
        cv=cfg["cv"],
        split_evaluacion=cfg["split_evaluacion"],
        semilla=cfg["seed"],
        estandarizar=cfg.get("estandarizar", False),
        n_jobs=n_jobs,
    )
    tarea.evaluar()
    info = tarea.info[nombre]
    if "accuracy" not in info or (
        cfg["scoring"] == "neg_log_loss" and "r2" not in info
    ):
        raise RuntimeError(f"Evaluación incompleta de {nombre} en {ruta_config.stem}")
    clases = np.asarray(tarea.clasificadores[nombre].classes_)
    if "probas" in info:
        info.temple = _templar(tarea, info.probas, clases)
    busqueda = info.get("busqueda")
    if busqueda is not None:
        info.mejores_params = dict(busqueda.best_params_)
        info.selecciones = Bunch()
        for pedido in cfg.get("selecciones", []):
            otra = seleccion.seleccionar_otra(
                tarea, nombre, pedido["regla"], pedido.get("restriccion")
            )
            if otra.igual and "temple" in info:
                otra.temple = info.temple
            elif "probas" in otra:
                otra.temple = _templar(tarea, otra.probas, clases)
            info.selecciones[pedido["nombre"]] = otra
        if not cfg.get("guardar_modelo", True):
            del busqueda.best_estimator_
    tarea.info.campania = cfg.get("campania")
    tarea.info.entorno = _entorno()
    info.t_pared = time.time() - t_inicio
    dir_infos.mkdir(parents=True, exist_ok=True)
    destino = dir_infos / f"{ruta_config.stem}.pkl"
    temporal = dir_infos / f".{ruta_config.stem}.pkl.tmp"
    with open(temporal, "wb") as archivo:
        pickle.dump(tarea.info, archivo)
    temporal.rename(destino)
    otras = info.get("selecciones", {})
    return {
        "tarea": ruta_config.stem,
        "clf": nombre,
        "segundos": info.t_pared,
        "t_entrenar": info.t_entrenar,
        "r2": info.get("r2", np.nan),
        "r2_temp": info.temple["r2_temp"] if "temple" in info else np.nan,
        "accuracy": info.accuracy,
        "n_candidatos": len(busqueda.cv_results_["params"]) if busqueda else 1,
        "memoria_max_mb": _memoria_maxima_mb(),
        **{f"igual_{k}": bool(v.igual) for k, v in otras.items()},
    }


# =====================================================================
# Procesos trabajadores
# =====================================================================

_dir_cache: Path | None = None


def _redirigir_cache(directorio: Path):
    """Caché en disco de `fkdc.fermat` propio de este proceso.

    Con `n_jobs=1` no hay workers de loky, así que alcanza con reemplazar las dos
    funciones memorizadas del módulo (que `SampleFermatDistance` busca por nombre).
    """
    from joblib import Memory

    from fkdc import fermat

    memoria = Memory(directorio, verbose=0)
    fermat.euclidean = memoria.cache(fermat.euclidean.func)
    fermat.fermat_muestral = memoria.cache(fermat.fermat_muestral.func)


def _inicializar(dir_cache_base: Path, carril: str, hilos: int):
    """Hilos por proceso, advertencias silenciadas y caché propio."""
    global _dir_cache, _limites_hilos
    from threadpoolctl import threadpool_limits

    import __main__

    __main__.Dataset = Dataset  # los datasets de v5 y v6 se guardaron desde __main__
    _limites_hilos = threadpool_limits(limits=hilos)
    if carril == "torch":
        import torch

        torch.set_num_threads(hilos)
    for categoria in (UserWarning, RuntimeWarning, FutureWarning):
        warnings.filterwarnings("ignore", category=categoria)
    logging.getLogger("fkdc.tarea").setLevel(logging.WARNING)
    _dir_cache = dir_cache_base / str(os.getpid())
    _redirigir_cache(_dir_cache)


def _correr_lote(rutas: list[str], dir_infos: str) -> list[dict]:
    """Corre las configs de una tarea y vacía el caché del proceso."""
    filas = []
    for ruta in rutas:
        try:
            filas.append(ejecutar(Path(ruta), Path(dir_infos), n_jobs=1))
        except Exception as exc:
            filas.append(
                {
                    "tarea": Path(ruta).stem,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traza": traceback.format_exc(limit=5),
                }
            )
    if _dir_cache is not None:
        shutil.rmtree(_dir_cache, ignore_errors=True)
    return filas


# =====================================================================
# Comando
# =====================================================================


def pendientes(
    c: campanias.Campania,
    carril: str,
    clasificadores=None,
    variantes=None,
    semillas: int | None = None,
    extras: bool = False,
) -> list[list[Path]]:
    """Lotes de configs sin info, por tarea, en orden de repetición y de variante."""
    elegidos = [
        nombre
        for nombre in (c.todos if extras else c.clasificadores)
        if registro.entrada(nombre).carril == carril
        and (clasificadores is None or nombre in clasificadores)
    ]
    orden_clf = {nombre: i for i, nombre in enumerate(c.todos)}
    orden_variante = {v: i for i, v in enumerate(c.variantes)}
    repeticion = {str(s): i for i, s in enumerate(c.semillas)}
    lotes = defaultdict(list)
    for ruta in c.dir_configs.glob("*.yaml"):
        partes = dict(zip(campos_nombre, ruta.stem.split("-"), strict=True))
        if partes["clf"] not in elegidos or partes["variante"] not in orden_variante:
            continue
        if variantes is not None and partes["variante"] not in variantes:
            continue
        semilla = (
            partes["semilla_tarea"]
            if partes["semilla_ds"] == "None"
            else partes["semilla_ds"]
        )
        if semillas is not None and repeticion[semilla] >= semillas:
            continue
        if (c.dir_infos / f"{ruta.stem}.pkl").exists():
            continue
        clave = (
            repeticion[semilla],
            orden_variante[partes["variante"]],
            partes["semilla_ds"],
            partes["semilla_tarea"],
        )
        lotes[clave].append(ruta)
    return [
        sorted(lotes[clave], key=lambda r: orden_clf[r.stem.split("-")[2]])
        for clave in sorted(lotes)
    ]


def correr_campania(
    c: campanias.Campania,
    carril: str = "cpu",
    procesos: int = 10,
    clasificadores=None,
    variantes=None,
    semillas: int | None = None,
    extras: bool = False,
    en_seco: bool = False,
    hilos: int | None = None,
    memoria_gb: float = 16.0,
    gb_por_proceso: float = 1.5,
):
    """Evalúa las configs pendientes de la campaña `c` en el carril elegido.

    En el carril "cpu", los procesos se limitan a `memoria_gb / gb_por_proceso` (el
    sistema operativo ya mató trabajos por falta de memoria con 24 GB compartidos);
    en el "torch" corre uno solo con `hilos` hilos.
    """
    lotes = pendientes(c, carril, clasificadores, variantes, semillas, extras)
    por_clf = pd.Series(
        [r.stem.split("-")[2] for lote in lotes for r in lote], dtype=object
    ).value_counts()
    if carril == "cpu":
        procesos = max(1, min(procesos, int(memoria_gb // gb_por_proceso)))
        hilos = hilos or 1
    else:
        procesos, hilos = 1, hilos or 4
    logger.info(
        "%s, carril %s: %d lotes, %d configs pendientes; %d procesos × %d hilos",
        c.nombre,
        carril,
        len(lotes),
        por_clf.sum(),
        procesos,
        hilos,
    )
    if en_seco or not lotes:
        print(por_clf.to_string())
        return
    ruta_registro = c.dir / f"registro-{carril}.csv"
    ruta_fallas = c.dir / f"fallas-{carril}.csv"
    dir_cache_base = c.dir / f"cache-{carril}"
    t0, hechas, total = time.time(), 0, int(por_clf.sum())
    with ProcessPoolExecutor(
        procesos,
        initializer=_inicializar,
        initargs=(dir_cache_base, carril, hilos),
    ) as pool:
        futuros = [
            pool.submit(_correr_lote, [str(r) for r in lote], str(c.dir_infos))
            for lote in lotes
        ]
        for futuro in as_completed(futuros):
            filas = pd.DataFrame(futuro.result())
            hechas += len(filas)
            con_error = (
                filas.error.notna()
                if "error" in filas
                else pd.Series(False, index=filas.index)
            )
            for tabla, ruta in (
                (filas[~con_error], ruta_registro),
                (filas[con_error], ruta_fallas),
            ):
                if not tabla.empty:
                    tabla.to_csv(ruta, mode="a", header=not ruta.exists(), index=False)
            transcurrido = time.time() - t0
            logger.info(
                "%d/%d configs (%d fallas en el lote); faltan ~%.1f h",
                hechas,
                total,
                int(con_error.sum()),
                transcurrido / hechas * (total - hechas) / 3600,
            )
    shutil.rmtree(dir_cache_base, ignore_errors=True)


app = typer.Typer(add_completion=False, help=__doc__)


def _escribir_atomico(tabla: pd.DataFrame, ruta: Path):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(f".{ruta.name}.tmp")
    tabla.to_csv(temporal, index=False)
    temporal.rename(ruta)


@app.command()
def lote(
    nombre: Annotated[str, typer.Argument(help="Campaña (fkdc.campanias)")],
    carril: Annotated[str, typer.Argument(help="cpu o torch")],
    marca: Annotated[Path, typer.Argument(help="Marca `.hecho` del lote")],
    rutas: Annotated[list[Path], typer.Argument(help="Configs del lote")],
    hilos: Annotated[
        int | None, typer.Option(help="Hilos del proceso (cpu: 1, torch: 4)")
    ] = None,
):
    """Corre un lote de configs en un proceso (la receta de Make de cada lote).

    Saltea las configs que ya tienen info, de modo que repetir el lote tras un corte
    rehace solo lo que falta. Si todas terminan bien escribe la marca (de forma
    atómica) y borra fallas viejas del lote; si alguna falla, las anota en
    `{dir}/fallas/{carril}/{lote}.csv`, no escribe la marca y sale con error, para que
    `make -k` siga con los demás lotes y la próxima corrida lo reintente.
    """
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    c = campanias.campania(nombre)
    hilos = hilos or (1 if carril == "cpu" else 4)
    dir_cache = c.dir / "cache" / carril / marca.stem
    shutil.rmtree(dir_cache, ignore_errors=True)  # restos de un corte anterior
    _inicializar(dir_cache.parent, carril, hilos)
    global _dir_cache
    _dir_cache = dir_cache
    _redirigir_cache(dir_cache)
    pendientes_lote = [r for r in rutas if not (c.dir_infos / f"{r.stem}.pkl").exists()]
    logger.info(
        "Lote %s: %d configs, %d pendientes",
        marca.stem,
        len(rutas),
        len(pendientes_lote),
    )
    filas = _correr_lote([str(r) for r in pendientes_lote], str(c.dir_infos))
    tabla = pd.DataFrame(filas)
    for fila in filas:
        logger.info(
            "%s: %s",
            fila["tarea"],
            fila.get("error") or f"{fila['segundos']:.1f}s, R²={fila['r2']:.4f}",
        )
    if not tabla.empty:
        _escribir_atomico(tabla, c.dir / "registros" / carril / f"{marca.stem}.csv")
    ruta_fallas = c.dir / "fallas" / carril / f"{marca.stem}.csv"
    if "error" in tabla and tabla.error.notna().any():
        _escribir_atomico(tabla[tabla.error.notna()], ruta_fallas)
        raise typer.Exit(code=1)
    ruta_fallas.unlink(missing_ok=True)
    marca.parent.mkdir(parents=True, exist_ok=True)
    temporal = marca.with_name(f".{marca.name}.tmp")
    temporal.touch()
    temporal.rename(marca)


@app.command("campania")
def correr(
    nombre: Annotated[str, typer.Argument(help="Campaña (fkdc.campanias)")],
    carril: Annotated[str, typer.Option(help="cpu o torch")] = "cpu",
    procesos: Annotated[int, typer.Option(help="Procesos simultáneos (cpu)")] = 10,
    hilos: Annotated[
        int | None, typer.Option(help="Hilos por proceso (cpu: 1, torch: 4)")
    ] = None,
    memoria_gb: Annotated[
        float, typer.Option(help="Memoria disponible para la corrida")
    ] = 16.0,
    gb_por_proceso: Annotated[
        float, typer.Option(help="Memoria máxima esperable por proceso (cpu)")
    ] = 1.5,
    conjuntos: Annotated[
        list[str] | None,
        typer.Option("--conjunto", help="Conjuntos de clasificadores (campanias)"),
    ] = None,
    clasificadores: Annotated[list[str] | None, typer.Option("--clf")] = None,
    variantes: Annotated[list[str] | None, typer.Option("--variante")] = None,
    semillas: Annotated[
        int | None, typer.Option(help="Solo las primeras N semillas")
    ] = None,
    extras: Annotated[
        bool, typer.Option(help="Incluir los extras no locales (API)")
    ] = False,
    en_seco: Annotated[bool, typer.Option(help="Solo contar lo pendiente")] = False,
):
    """Evalúa las configs pendientes de la campaña `nombre` en el carril elegido."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    correr_campania(
        campanias.campania(nombre, tuple(conjuntos) if conjuntos else None),
        carril,
        procesos,
        clasificadores,
        variantes,
        semillas,
        extras,
        en_seco,
        hilos,
        memoria_gb,
        gb_por_proceso,
    )


if __name__ == "__main__":
    app()
