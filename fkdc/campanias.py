"""
Campañas: qué clasificadores, datasets, semillas, grillas y reglas corre cada una.

Una `Campania` vive en un directorio propio con `datasets/`, `configs/` e `infos/`, y
fija todo lo que hace falta para regenerarla:

- `tesis`: la corrida entregada (`sandbox/v5`): 25 semillas, los ocho clasificadores y
  las grillas de `fkdc.config`, kdc con `sklearn.neighbors.KernelDensity`.
- `heroica`: 100 semillas sobre las 24 variantes de la tesis y los 9 datasets de
  `fkdc.variedades`; kdc exacto como primario y el de sklearn como análisis de
  sensibilidad; grillas emparejadas (la de anchos de banda de fkdc es uno de cada tres
  valores de la de kdc); selección R1SD de la tesis, más la ciega a α de v6 derivada
  sobre la misma búsqueda; infos sin el modelo reajustado. Sus clasificadores salen de
  `conjuntos`: por omisión, los de la tesis, el compendio barrido y los fundacionales
  locales.

La comparación crítica de una campaña usa solo clasificadores locales
(`fkdc.registro`); los que corren por una API van en `extras`.

Las semillas son `config._obtener_semillas(semilla_principal, repeticiones)`: los
conjuntos de 25, 64 y 100 están anidados. Datasets y configs de una campaña usan
siempre sus mismas `repeticiones`: si no, `hacer_configuraciones` trataría como real a
un dataset sintético cuya semilla no está en la lista.

Uso, desde la raíz del repositorio:
    .venv/bin/python -m fkdc.campanias describir heroica
    .venv/bin/python -m fkdc.campanias datasets heroica --escribir
    .venv/bin/python -m fkdc.campanias configs heroica
    .venv/bin/python -m fkdc.campanias tablas heroica
"""

import logging
import os
import pickle
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from sklearn.utils import Bunch

from fkdc import config, dir_raiz, registro
from fkdc.utils import yaml

logger = logging.getLogger(__name__)

# Grillas emparejadas de anchos de banda: un doceavo de década para kdc (133 valores)
# y uno de cada tres para fkdc, que coincide bit a bit con `np.logspace(-5, 6, 45)`,
# la grilla de fkdc de la tesis.
h_kdc = 10.0 ** (-5 + np.arange(133) / 12)
h_fkdc = h_kdc[::3]
alphas = np.asarray(config.grillas["fkdc"]["alpha"])

# Conjuntos de clasificadores para la campaña heroica. Los que no están en
# `conjuntos_por_omision` siguen registrados y se suman con `--conjunto`; los no
# locales (API) entran siempre como extras.
conjuntos = {
    # kdc_exacto cumple el papel de kdc; kdc (sklearn, como la tesis) es sensibilidad
    "tesis": ["fkdc", "kdc_exacto", "fkn", "kn", "gnb", "lr", "svc", "gbt", "kdc"],
    "compendio": [
        "qda",
        "lda",
        "gmm",
        "kdc_abramson",
        "isomap_kn",
        "nca_kn",
        "label_spreading",
        "gpc",
        "et",
    ],
    # Fundacionales y profundos locales elegidos por el autor el 2026-10-01
    "fundacionales": ["tabpfn", "tabpfn_v2", "tabicl", "tabm"],
    "compendio_caro": ["isomap_kdc", "rf", "mlp"],
    "profundos": ["realmlp", "tabdpt"],
    "oraculos": [
        f"{base}_oraculo" for base in ("lr", "gnb", "kdc", "kn", "svc", "gbt")
    ],
    # TabPFN Plus quedó fuera de toda comparación (decisión del autor)
    "api": ["tabpfn_api"],
}
conjuntos_por_omision = ("tesis", "compendio", "fundacionales")

dir_v5 = dir_raiz / "sandbox/v5"
dir_v6 = dir_raiz / "sandbox/v6"
dir_nuevos = dir_raiz / "sandbox/fundacionales/datasets"


@dataclass
class Campania:
    """Definición completa de una campaña.

    `grillas` reemplaza la grilla del registro de algunos clasificadores;
    `selecciones` lista, por clasificador, las reglas de selección que se derivan a
    posteriori sobre los mismos `cv_results_` (`fkdc.seleccion`); `roles` dice qué
    entrada del registro cumple cada papel en el análisis (p. ej. kdc → kdc_exacto).
    """

    nombre: str
    dir: Path
    repeticiones: int
    clasificadores: list[str]
    variantes: list[str]
    extras: list[str] = field(default_factory=list)
    grillas: dict[str, dict] = field(default_factory=dict)
    reajuste: str = "parsimoniosa"
    selecciones: dict[str, list[dict]] = field(default_factory=dict)
    roles: dict[str, str] = field(default_factory=dict)
    guardar_modelo: bool = True
    semilla_principal: int = config.semilla_principal
    cv: int = config.cv
    split_evaluacion: float = config.split_evaluacion

    def __post_init__(self):
        for nombre in [*self.clasificadores, *self.extras]:
            registro.entrada(nombre)  # falla si no está registrado
        no_locales = [c for c in self.clasificadores if not registro.entrada(c).local]
        if no_locales:
            raise ValueError(
                f"La comparación crítica admite solo locales; mover a `extras`: "
                f"{no_locales}"
            )

    @property
    def semillas(self) -> list[int]:
        return config._obtener_semillas(self.semilla_principal, self.repeticiones)

    @property
    def dir_datasets(self) -> Path:
        return self.dir / "datasets"

    @property
    def dir_configs(self) -> Path:
        return self.dir / "configs"

    @property
    def dir_infos(self) -> Path:
        return self.dir / "infos"

    @property
    def todos(self) -> list[str]:
        return [*self.clasificadores, *self.extras]


def _variantes_tesis() -> list[str]:
    from fkdc.datasets import datasets

    return list(datasets)


@cache
def _variantes_nuevas() -> tuple[str, ...]:
    from fkdc.variedades import variedades

    return tuple(variedades)


def _reales() -> set[str]:
    """Variantes con un único dataset, cuya semilla define la partición."""
    from fkdc.datasets import datasets_reales

    return set(datasets_reales)


@cache
def campania(nombre: str, elegidos: tuple[str, ...] | None = None) -> Campania:
    """La campaña `nombre` (`tesis` o `heroica`).

    `elegidos` son nombres de `conjuntos` (solo para `heroica`; por omisión,
    `conjuntos_por_omision`). Los clasificadores no locales van a `extras`.
    """
    if nombre == "tesis":
        return Campania(
            "tesis",
            config.dir_ejecucion.parent,
            config.repeticiones,
            list(config.clasificadores),
            _variantes_tesis(),
        )
    if nombre == "heroica":
        restriccion_h = {"bandwidth": h_fkdc.tolist()}
        ciega = [{"nombre": "ciega_a_alpha", "regla": "ciega_a_alpha"}]
        kdc_sobre_h_fkdc = [
            {
                "nombre": "parsimoniosa_h_fkdc",
                "regla": "parsimoniosa",
                "restriccion": restriccion_h,
            }
        ]
        desconocidos = set(elegidos or ()) - set(conjuntos)
        if desconocidos:
            raise KeyError(f"Conjuntos desconocidos: {sorted(desconocidos)}")
        nombres = [
            clf
            for conjunto in elegidos or conjuntos_por_omision
            for clf in conjuntos[conjunto]
        ]
        return Campania(
            "heroica",
            dir_raiz / "sandbox/heroica",
            100,
            [clf for clf in nombres if registro.entrada(clf).local],
            [*_variantes_tesis(), *_variantes_nuevas()],
            extras=[clf for clf in nombres if not registro.entrada(clf).local],
            grillas={
                "fkdc": {"alpha": alphas, "bandwidth": h_fkdc},
                "kdc_exacto": {"bandwidth": h_kdc},
                "kdc": {"bandwidth": h_kdc},
                "kdc_abramson": {"bandwidth": h_kdc},
            },
            selecciones={
                "fkdc": ciega,
                "fkn": ciega,
                "kn": ciega,
                "kdc_exacto": kdc_sobre_h_fkdc,
                "kdc": kdc_sobre_h_fkdc,
            },
            roles={"kdc": "kdc_exacto", "kdc_sklearn": "kdc"},
            guardar_modelo=False,
        )
    if nombre == "prueba":
        # Campaña mínima para probar el circuito de Make (corte y reanudación)
        return replace(
            campania("heroica"),
            nombre="prueba",
            dir=dir_raiz / "sandbox/prueba",
            repeticiones=2,
            clasificadores=["kn", "gnb", "qda", "kdc_exacto", "lr", "tabpfn_v2"],
            variantes=["lunas_lo", "iris", "nudos_200"],
            extras=[],
        )
    raise KeyError(f"Campaña desconocida: {nombre}")


# =====================================================================
# Tareas y datasets
# =====================================================================


def tareas(c: Campania) -> list[Bunch]:
    """Tareas de la campaña, ordenadas por repetición y luego por variante.

    Sintéticos: un dataset por semilla, partición con la semilla principal. Reales:
    un único dataset, partición con cada semilla. La repetición es el índice de la
    semilla en la lista, de modo que las primeras 25 son las de la tesis.
    """
    reales = _reales()
    indice = {s: i for i, s in enumerate(c.semillas)}
    lista = []
    for orden, variante in enumerate(c.variantes):
        base = variante.removesuffix("_std")
        for semilla in c.semillas:
            real = variante in reales
            lista.append(
                Bunch(
                    variante=variante,
                    base=base,
                    semilla_ds=None if real else semilla,
                    semilla_tarea=semilla if real else c.semilla_principal,
                    archivo=f"{base}.pkl" if real else f"{base}-{semilla}.pkl",
                    estandarizar=variante.endswith("_std"),
                    repeticion=indice[semilla],
                    orden=orden,
                )
            )
    return sorted(lista, key=lambda t: (t.repeticion, t.orden))


def _cargar(ruta: Path):
    import __main__
    from fkdc.datasets import Dataset

    __main__.Dataset = Dataset  # los datasets de v5 y v6 se guardaron desde __main__
    with open(ruta, "rb") as archivo:
        return pickle.load(archivo)


def _regenerar(ds):
    """El dataset que producen hoy la fábrica y los parámetros guardados en `ds`."""
    from sklearn import datasets as sk_datasets

    from fkdc import datasets as fk_datasets
    from fkdc.datasets import Dataset

    params = dict(ds.params)
    nombre_fabrica = params.pop("factory")
    ruido = params.pop("ruido", None)
    fabrica = getattr(fk_datasets, nombre_fabrica, None) or getattr(
        sk_datasets, nombre_fabrica
    )
    return Dataset.de_fabrica(fabrica, ruido=ruido, **params)


@cache
def _signos_mnist() -> np.ndarray:
    """Signos por componente que llevan el PCA de mnist de v6 al de v5.

    v5 ajustó el PCA sin `random_state` y v6 lo fijó; un PCA así puede diferir en el
    signo de cada componente (hoy no difiere en ninguna: solo en ruido de redondeo).
    Se exige el mismo vector en todas las semillas compartidas y una coincidencia
    relativa mejor que 1e-9 tras alinear.
    """
    signos = None
    for ruta in sorted(dir_v5.glob("datasets/mnist-*.pkl")):
        v5, v6 = _cargar(ruta), _cargar(dir_v6 / "datasets" / ruta.name)
        estos = np.sign((v5.X * v6.X).sum(axis=0))
        if signos is not None and not np.array_equal(signos, estos):
            raise ValueError(f"Signos de mnist inconsistentes en {ruta.name}")
        signos = estos
        escala = np.abs(v5.X).max(axis=0)
        if np.abs(v5.X - v6.X * signos).max(axis=0).max() > 1e-9 * escala.max():
            raise ValueError(f"mnist de v6 no coincide con v5 en {ruta.name}")
    return signos


def fuente_dataset(c: Campania, t: Bunch) -> tuple[Path | None, str]:
    """(ruta de origen, cómo se obtiene) del dataset de la tarea `t`.

    Las 25 semillas de la tesis y los reales salen de `sandbox/v5` tal cual; las demás
    de las 20 familias, de `sandbox/v6` (lo que hoy produce el código; mnist con los
    signos de v5); los datasets nuevos, de `sandbox/fundacionales/datasets` o, si
    falta la semilla, de `fkdc.variedades`.
    """
    if t.base in _variantes_nuevas():
        ruta = dir_nuevos / t.archivo
        return (ruta, "enlace") if ruta.exists() else (None, "generar")
    if t.semilla_ds is None or t.repeticion < config.repeticiones:
        return dir_v5 / "datasets" / t.archivo, "enlace"
    ruta = dir_v6 / "datasets" / t.archivo
    return ruta, ("signos_mnist" if t.base == "mnist" else "enlace_verificado")


def _guardar_atomico(dataset, destino: Path):
    """Guarda en un temporal y renombra: un corte nunca deja un pickle a medias."""
    temporal = destino.with_name(f".{destino.name}.tmp")
    dataset.guardar(temporal)
    temporal.rename(destino)


def preparar_datasets(c: Campania, escribir: bool = False) -> pd.DataFrame:
    """Lleva a `c.dir_datasets` un dataset por archivo de la campaña, sin tocar v5.

    Enlaces duros (mismo disco, sin espacio extra) para lo que se reutiliza; archivos
    nuevos para mnist con signos alineados y para los datasets que hay que generar.
    Verifica cada enlace desde v6 regenerándolo con su fábrica y sus parámetros
    guardados. Con `escribir=False` solo verifica e informa.
    """
    filas, vistos = [], set()
    if escribir:
        c.dir_datasets.mkdir(parents=True, exist_ok=True)
    for t in tareas(c):
        if t.archivo in vistos:
            continue
        vistos.add(t.archivo)
        origen, accion = fuente_dataset(c, t)
        destino = c.dir_datasets / t.archivo
        fila = {"archivo": t.archivo, "origen": origen, "accion": accion}
        if accion == "enlace_verificado":
            ds = _cargar(origen)
            nuevo = _regenerar(ds)
            fila["verificado"] = bool(
                np.array_equal(ds.X, nuevo.X) and np.array_equal(ds.y, nuevo.y)
            )
            if not fila["verificado"]:
                raise ValueError(f"{origen} no coincide con su regeneración")
        if escribir and not destino.exists():
            if accion in ("enlace", "enlace_verificado"):
                os.link(origen, destino)
            elif accion == "signos_mnist" and (_signos_mnist() > 0).all():
                os.link(origen, destino)
            elif accion == "signos_mnist":
                from fkdc.datasets import Dataset

                ds = _cargar(origen)
                _guardar_atomico(Dataset(ds.X * _signos_mnist(), ds.y), destino)
            else:  # generar
                from fkdc.datasets import Dataset
                from fkdc.variedades import variedades

                variedad = variedades[t.base]
                nuevo = Dataset.de_fabrica(
                    variedad.fabrica,
                    nombre=t.base,
                    random_state=t.semilla_ds,
                    **variedad.params,
                )
                _guardar_atomico(nuevo, destino)
        filas.append(fila)
    return pd.DataFrame(filas)


# =====================================================================
# Configuraciones
# =====================================================================


def _serializable(valores):
    """Lista de escalares de Python (YAML legible y exacto para floats)."""
    return [v.item() if hasattr(v, "item") else v for v in np.asarray(valores).tolist()]


def _ruta_relativa(ruta: Path) -> str:
    """Ruta relativa a la raíz del repositorio, o absoluta si está fuera."""
    return str(ruta.relative_to(dir_raiz) if ruta.is_relative_to(dir_raiz) else ruta)


def hacer_configuraciones(c: Campania, clasificadores=None, dir_configs=None) -> int:
    """Escribe un YAML por clasificador y tarea; devuelve cuántos escribió.

    Nombre: `{variante}-{semilla_ds}-{clf}-{semilla_tarea}-{puntuacion}.yaml`, como en
    la tesis. Además de las claves de siempre, cada config lleva `campania`,
    `variante`, `reajuste`, `selecciones` y `guardar_modelo`, que lee `fkdc.correr`.
    """
    dir_configs = dir_configs or c.dir_configs
    dir_configs.mkdir(parents=True, exist_ok=True)
    lista = tareas(c)
    formas = {}
    for t in lista:
        if t.variante not in formas:
            origen, _ = fuente_dataset(c, t)
            ruta = c.dir_datasets / t.archivo
            ds = _cargar(ruta if ruta.exists() else origen)
            formas[t.variante] = (ds.n, ds.p)
    escritas = 0
    for nombre in clasificadores or c.todos:
        for t in lista:
            if not registro.aplica(nombre, t.variante):
                continue
            n_muestras, n_atributos = formas[t.variante]
            estimador, grilla, puntuacion = registro.preparar(
                nombre,
                n_muestras,
                n_atributos,
                variante=t.variante,
                grilla=c.grillas.get(nombre),
                cv=c.cv,
                split_evaluacion=c.split_evaluacion,
            )
            configuracion = {
                "campania": c.nombre,
                "variante": t.variante,
                "dataset": _ruta_relativa(c.dir_datasets / t.archivo),
                "clasificador": nombre,
                "grilla_hipers": {k: _serializable(v) for k, v in grilla.items()},
                "cv": c.cv,
                "split_evaluacion": c.split_evaluacion,
                "seed": t.semilla_tarea,
                "scoring": puntuacion,
                "estandarizar": t.estandarizar,
                "reajuste": c.reajuste,
                "selecciones": c.selecciones.get(nombre, []),
                "guardar_modelo": c.guardar_modelo,
                "repeticiones": c.repeticiones,
            }
            clave = (t.variante, t.semilla_ds, nombre, t.semilla_tarea, puntuacion)
            ruta = dir_configs / ("-".join(map(str, clave)) + ".yaml")
            temporal = ruta.with_name(f".{ruta.name}.tmp")
            with open(temporal, "w") as archivo:
                yaml.dump(configuracion, archivo)
            temporal.rename(ruta)
            escritas += 1
    return escritas


# =====================================================================
# Lotes para Make
# =====================================================================

campos_config = ["variante", "semilla_ds", "clf", "semilla_tarea", "puntuacion"]
# Semillas por lote en cada carril: en "cpu", una tarea por lote (todos sus
# clasificadores comparten el caché de Fermat); en "torch", diez semillas, para cargar
# cada modelo una vez por lote y no una por tarea
semillas_por_lote = {"cpu": 1, "torch": 10}


def lotes(c: Campania, carril: str) -> dict[str, list[Path]]:
    """Configs de un carril agrupadas en lotes, en orden de repetición.

    Nombre del lote: `{variante}-{repetición inicial}` con tres cifras. Dentro de un
    lote, las configs van por clasificador (en el orden de la campaña) y por
    repetición.
    """
    tamano = semillas_por_lote[carril]
    elegidos = {
        clf: i
        for i, clf in enumerate(c.todos)
        if registro.entrada(clf).carril == carril
    }
    repeticion = {str(s): i for i, s in enumerate(c.semillas)}
    orden_variante = {v: i for i, v in enumerate(c.variantes)}
    grupos = {}
    for ruta in c.dir_configs.glob("*.yaml"):
        partes = dict(zip(campos_config, ruta.stem.split("-"), strict=True))
        if partes["clf"] not in elegidos or partes["variante"] not in orden_variante:
            continue
        semilla = (
            partes["semilla_tarea"]
            if partes["semilla_ds"] == "None"
            else partes["semilla_ds"]
        )
        rep = repeticion[semilla]
        clave = (rep // tamano, orden_variante[partes["variante"]], partes["variante"])
        grupos.setdefault(clave, []).append((elegidos[partes["clf"]], rep, ruta))
    return {
        f"{variante}-{bloque * tamano:03d}": [r for *_, r in sorted(grupo)]
        for (bloque, _, variante), grupo in sorted(grupos.items())
    }


def escribir_lotes(c: Campania, ruta_mk: Path | None = None) -> Path:
    """Escribe `{dir}/lotes.mk`: los lotes de cada carril y sus configs.

    Cada lote `{dir}/lotes/{carril}/{nombre}.hecho` depende de sus configs; la receta
    (en el `Makefile`) corre `fkdc.correr lote`, que crea la marca solo si todas las
    configs del lote terminaron bien.
    """
    ruta_mk = ruta_mk or c.dir / "lotes.mk"
    lineas = [f"# Generado por `python -m fkdc.campanias lotes {c.nombre}`; no editar."]
    reglas = []
    for carril in ("cpu", "torch"):
        por_lote = lotes(c, carril)
        marcas = [
            _ruta_relativa(c.dir / "lotes" / carril / f"{nombre}.hecho")
            for nombre in por_lote
        ]
        lineas.append(f"LOTES_{carril.upper()} := " + " ".join(marcas))
        for marca, rutas in zip(marcas, por_lote.values(), strict=True):
            reglas.append(f"{marca}: " + " ".join(map(_ruta_relativa, rutas)))
    temporal = ruta_mk.with_name(f".{ruta_mk.name}.tmp")
    temporal.write_text("\n".join([*lineas, *reglas]) + "\n")
    temporal.rename(ruta_mk)
    return ruta_mk


def progreso(c: Campania) -> pd.DataFrame:
    """Configs, infos y lotes fallidos por carril y clasificador."""
    hechas = {ruta.stem for ruta in c.dir_infos.glob("*.pkl")}
    filas = []
    for ruta in c.dir_configs.glob("*.yaml"):
        clf = ruta.stem.split("-")[2]
        filas.append(
            {
                "carril": registro.entrada(clf).carril,
                "clf": clf,
                "hecha": ruta.stem in hechas,
            }
        )
    tabla = pd.DataFrame(filas)
    if tabla.empty:
        return tabla
    resumen = tabla.groupby(["carril", "clf"]).hecha.agg(configs="size", infos="sum")
    resumen["faltan"] = resumen.configs - resumen.infos
    return resumen.reset_index()


# =====================================================================
# Tablas resumen
# =====================================================================


def _fila_corrida(ruta: Path) -> dict:
    """Métricas, hiperparámetros elegidos y selecciones derivadas de un info."""
    import warnings

    from sklearn.exceptions import InconsistentVersionWarning

    variante, semilla_ds, clf, semilla_tarea, _ = ruta.stem.split("-")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        with open(ruta, "rb") as archivo:
            info = pickle.load(archivo)
    datos = info[clf]
    busqueda = datos.get("busqueda")
    mejores = datos.get("mejores_params") or (
        dict(busqueda.best_params_) if busqueda is not None else {}
    )
    fila = {
        "variante": variante,
        "semilla": int(semilla_tarea if semilla_ds == "None" else semilla_ds),
        "clf": clf,
        "r2": datos.get("r2", np.nan),
        "accuracy": datos.get("accuracy", np.nan),
        "r2_temp": datos.temple["r2_temp"] if "temple" in datos else np.nan,
        "t_entrenar": datos.get("t_entrenar", np.nan),
        **{f"param_{k.split('__')[-1]}": v for k, v in mejores.items()},
    }
    for nombre, otra in (datos.get("selecciones") or {}).items():
        fila[f"igual_{nombre}"] = bool(otra.igual)
        fila[f"r2_{nombre}"] = otra.get("r2", np.nan)
        fila[f"accuracy_{nombre}"] = otra.get("accuracy", np.nan)
        if "temple" in otra:
            fila[f"r2_temp_{nombre}"] = otra.temple["r2_temp"]
    return fila


def corridas(c: Campania) -> pd.DataFrame:
    """Una fila por info de la campaña, de sus clasificadores (incluidos extras)."""
    import __main__
    from fkdc.datasets import Dataset

    __main__.Dataset = Dataset
    rutas = [
        ruta
        for ruta in sorted(c.dir_infos.glob("*.pkl"))
        if ruta.stem.split("-")[2] in c.todos
    ]
    return pd.DataFrame([_fila_corrida(ruta) for ruta in rutas])


def podio(tabla: pd.DataFrame, metrica: str) -> pd.DataFrame:
    """Clasificadores con la mayor mediana por variante, como el podio de la tesis.

    Mediana por (variante, clasificador) redondeada a 4 decimales; todo clasificador
    que alcanza el máximo de su variante cuenta (los empates, una vez por
    clasificador). Mismo cálculo que `fkdc/viz.py`.
    """
    tabla = tabla.rename(columns={"variante": "dataset"})
    medianas = (
        tabla.dropna(subset=metrica)
        .groupby(["dataset", "clf"])[metrica]
        .median()
        .round(4)
        .reset_index()
    )
    maximos = medianas.groupby("dataset")[metrica].transform("max")
    mejor = (
        medianas[medianas[metrica].eq(maximos)]
        .groupby("clf")["dataset"]
        .agg([len, ", ".join])
    )
    mejor.columns = ["cant", "datasets"]
    return mejor.sort_values("cant", ascending=False)


def tablas(c: Campania, dir_salida: Path | None = None) -> dict[str, pd.DataFrame]:
    """Tablas resumen de una campaña, en `{dir}/tablas` (o `dir_salida`).

    - `corridas.csv`: una fila por info.
    - `resumen.csv`: mediana e IQR de R², R² templado y exactitud por variante y
      clasificador.
    - `podio-{métrica}.csv`: el podio de la tesis, entre los clasificadores con papel en
      la comparación de la tesis (en la heroica, kdc es `kdc_exacto`) y sobre las 20
      variantes crudas de la tesis; `podio-nuevos-{métrica}.csv`, lo mismo sobre los
      datasets nuevos; `podio-ampliado-{métrica}.csv`, entre todos los clasificadores
      críticos (descriptivo).
    - `selecciones.csv`: por selección derivada, la proporción de tareas en que elige
      lo mismo que la principal y la mediana de la diferencia de R².
    """
    dir_salida = dir_salida or c.dir / "tablas"
    dir_salida.mkdir(parents=True, exist_ok=True)
    todas = corridas(c)

    def iqr(serie):
        return serie.quantile(0.75) - serie.quantile(0.25)

    metricas = [m for m in ("r2", "r2_temp", "accuracy") if todas[m].notna().any()]
    resumen = todas.groupby(["variante", "clf"]).agg(
        n=("r2", "size"),
        **{f"{m}_mediana": (m, "median") for m in metricas},
        **{f"{m}_iqr": (m, iqr) for m in metricas},
    )
    salidas = {"corridas": todas, "resumen": resumen.reset_index()}
    # Papeles de la tesis: los ocho, con el kdc que la campaña designa como primario
    papeles = {c.roles.get(clf, clf): clf for clf in config.clasificadores}
    de_tesis = todas[todas.clf.isin(papeles.values())].assign(
        clf=lambda t: t.clf.map({v: k for k, v in papeles.items()})
    )
    crudas = [v for v in _variantes_tesis() if not v.endswith("_std")]
    criticos = todas[todas.clf.isin(c.clasificadores)]
    for metrica in ("r2", "accuracy"):
        salidas[f"podio-{metrica}"] = podio(
            de_tesis[de_tesis.variante.isin(crudas)], metrica
        )
        nuevas = de_tesis[de_tesis.variante.isin(_variantes_nuevas())]
        if not nuevas.empty:
            salidas[f"podio-nuevos-{metrica}"] = podio(nuevas, metrica)
        salidas[f"podio-ampliado-{metrica}"] = podio(
            criticos[criticos.variante.isin(crudas)], metrica
        )
    filas = []
    for columna in (c for c in todas if c.startswith("igual_")):
        nombre = columna.removeprefix("igual_")
        con = todas[todas[columna].notna()]
        for clf, grupo in con.groupby("clf"):
            filas.append(
                {
                    "seleccion": nombre,
                    "clf": clf,
                    "n": len(grupo),
                    "igual": grupo[columna].astype(bool).mean(),
                    "delta_r2_mediana": (grupo[f"r2_{nombre}"] - grupo.r2).median(),
                }
            )
    if filas:
        salidas["selecciones"] = pd.DataFrame(filas)
    for nombre, tabla in salidas.items():
        indice = nombre.startswith("podio")
        tabla.to_csv(dir_salida / f"{nombre}.csv", index=indice)
    logger.info("Tablas de %s en %s", c.nombre, dir_salida)
    return salidas


# =====================================================================
# Comandos
# =====================================================================

app = typer.Typer(add_completion=False, help=__doc__)
opcion_conjuntos = typer.Option(
    "--conjunto", help="Conjuntos de clasificadores (por omisión, los de la campaña)"
)


def _campania_cli(nombre: str, elegidos: list[str] | None) -> Campania:
    return campania(nombre, tuple(elegidos) if elegidos else None)


@app.command()
def describir(
    nombre: str, elegidos: Annotated[list[str] | None, opcion_conjuntos] = None
):
    """Clasificadores, variantes, semillas y tareas de la campaña."""
    c = _campania_cli(nombre, elegidos)
    lista = tareas(c)
    print(f"Campaña {c.nombre} en {c.dir}: {c.repeticiones} semillas")
    print(f"- {len(c.variantes)} variantes, {len(lista)} tareas por clasificador")
    for nombre_clf in c.todos:
        e = registro.entrada(nombre_clf)
        n = sum(registro.aplica(nombre_clf, t.variante) for t in lista)
        rol = "extra" if nombre_clf in c.extras else "crítico"
        print(f"- {nombre_clf} ({e.grupo}, {e.carril}, {rol}): {n} tareas")


@app.command()
def datasets(
    nombre: str,
    escribir: Annotated[bool, typer.Option(help="Crear enlaces y archivos")] = False,
):
    """Verifica (y con `--escribir`, prepara) los datasets de la campaña."""
    logging.basicConfig(level=logging.INFO)
    informe = preparar_datasets(campania(nombre), escribir=escribir)
    print(informe.groupby("accion").size().to_string())
    if "verificado" in informe:
        print(f"Verificados contra su regeneración: {int(informe.verificado.sum())}")


@app.command()
def configs(
    nombre: str, elegidos: Annotated[list[str] | None, opcion_conjuntos] = None
):
    """Escribe las configs de la campaña en `{dir}/configs`."""
    logging.basicConfig(level=logging.INFO)
    c = _campania_cli(nombre, elegidos)
    logger.info("%d configs en %s", hacer_configuraciones(c), c.dir_configs)


@app.command("lotes")
def lotes_cli(
    nombre: str, elegidos: Annotated[list[str] | None, opcion_conjuntos] = None
):
    """Escribe `{dir}/lotes.mk` (lotes de cada carril para Make)."""
    logging.basicConfig(level=logging.INFO)
    ruta = escribir_lotes(_campania_cli(nombre, elegidos))
    logger.info("Escribió %s", ruta)


@app.command("progreso")
def progreso_cli(
    nombre: str, elegidos: Annotated[list[str] | None, opcion_conjuntos] = None
):
    """Avance de la campaña: configs, infos y lotes fallidos por carril."""
    c = _campania_cli(nombre, elegidos)
    tabla = progreso(c)
    if tabla.empty:
        print("Sin configs todavía")
        return
    print(tabla.to_string(index=False))
    for carril, grupo in tabla.groupby("carril"):
        hechos = len(list((c.dir / "lotes" / carril).glob("*.hecho")))
        fallidos = sorted((c.dir / "fallas" / carril).glob("*.csv"))
        print(
            f"{carril}: {int(grupo.infos.sum())}/{int(grupo.configs.sum())} infos, "
            f"{hechos} lotes completos, {len(fallidos)} lotes con fallas"
        )


@app.command("tablas")
def tablas_cli(
    nombre: str,
    elegidos: Annotated[list[str] | None, opcion_conjuntos] = None,
    dir_salida: Annotated[
        Path | None, typer.Option(help="Por omisión, {dir}/tablas")
    ] = None,
):
    """Tablas resumen de la campaña (`tablas`)."""
    logging.basicConfig(level=logging.INFO)
    salidas = tablas(_campania_cli(nombre, elegidos), dir_salida)
    for metrica in ("r2", "accuracy"):
        print(f"\nPodio por {metrica} mediano")
        print(salidas[f"podio-{metrica}"].to_markdown())


if __name__ == "__main__":
    app()
