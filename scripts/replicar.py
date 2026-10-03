"""
Réplica de las afirmaciones empíricas de la tesis con más semillas.

Cada afirmación de los capítulos de Resultados y Conclusiones de `docs/tesis.typ` se
codifica como una `Afirmacion`: qué compara, sobre qué datasets y clasificadores, con
qué métrica y qué dirección o valor sostiene la tesis. `replicar` la contrasta con una
tabla de corridas (una fila por tarea y clasificador) mediante una prueba por semilla
y corrige por comparaciones múltiples dentro de cada familia de afirmaciones: Holm
(FWER) como criterio y Benjamini–Hochberg (FDR) como referencia.

Tipos de afirmación y prueba:

- `par`: a vs. b apareados por semilla (mismo dataset y misma partición). Δ = a − b,
  con |Δ| < 1e-9 como empate (polvo de punto flotante, como en `fkdc/analisis.py` de
  `investigacion-v6`). Dirección ">" o "<": Wilcoxon de los rangos con signo, de dos
  colas; replica si p_Holm < nivel y el estimador de Hodges–Lehmann tiene el signo
  esperado, la contradice si p_Holm < nivel con el signo opuesto. Dirección "≈":
  equivalencia (TOST por bootstrap): replica si el IC 90 % de la media de Δ cae en
  ±margen.
- `nivel`: la mediana de un clasificador en un dataset vale lo que dice la tesis;
  IC 95 % por bootstrap de la mediana, replica si |mediana − valor| ≤ tolerancia.
- `seleccion`: proporción de semillas en que el hiperparámetro elegido cumple una
  condición (p. ej. α* = 1); IC de Clopper–Pearson, replica si contiene la proporción
  de la tesis.
- `podio`: clasificadores con la mayor mediana por dataset (redondeada a 4
  decimales, empates para todos, como `fkdc/viz.py`), contra el podio de la tesis
  (`docs/data/mejor-clf-por-dataset-segun-*-mediano.csv`); probabilidad bootstrap de
  que el ganador de la tesis siga ganando y Wilcoxon ganador vs. segundo.
- `dispersion`: cociente de rangos intercuartílicos de a sobre b, IC bootstrap.
- `caida`: diferencia de diferencias apareada, (a₁ − a₀) − (b₁ − b₀), p. ej. la caída
  de R² de `_0` a `_12` de un clasificador de la familia K frente a la de gbt.

Comandos, desde la raíz del repositorio:

    .venv/bin/python scripts/replicar.py cargar v5
    .venv/bin/python scripts/replicar.py cargar v5 --kdc-exacto
    .venv/bin/python scripts/replicar.py cargar v6
    .venv/bin/python scripts/replicar.py replicar corridas-v5 corridas-v6

`cargar heroica --dir-infos sandbox/heroica/infos` lee la campaña nueva.
"""

import logging
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from scipy.stats import beta, wilcoxon
from sklearn.exceptions import ConvergenceWarning, InconsistentVersionWarning

from fkdc import config, dir_raiz

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("replicar")

dir_heroica = dir_raiz / "sandbox/heroica"
dir_datos_tesis = dir_raiz / "docs/data"
tolerancia_empate = 1e-9
nivel_significacion = 0.05
n_bootstrap = 2000
semilla_bootstrap = 1312
clfs_k = ["fkdc", "kdc", "fkn", "kn"]
clfs_referencia = ["gnb", "lr", "svc", "gbt"]

app = typer.Typer(add_completion=False, help=__doc__)


# =====================================================================
# Carga de corridas
# =====================================================================


class _Desempaquetador(pickle.Unpickler):
    """Lee infos de otras ramas: reemplaza la regla de reajuste de v6 por un marcador.

    Las infos de `investigacion-v6` guardan `fkdc.v6.reajustar_ciego_a_alpha` como
    `refit` del `GridSearchCV`; para leer `cv_results_` y las métricas no hace falta.
    """

    def find_class(self, modulo, nombre):
        if modulo == "fkdc.v6":
            return _marcador
        return super().find_class(modulo, nombre)


def _marcador(*args, **kwargs):
    raise NotImplementedError("Regla de reajuste de v6 no disponible en esta rama")


def _fila_corrida(ruta: Path, origen: str) -> dict:
    """Métricas e hiperparámetros elegidos de una info `{ds}-{sds}-{clf}-{st}-{p}`."""
    dataset, semilla_ds, clf, semilla_tarea, _ = ruta.stem.split("-")
    info = _Desempaquetador(open(ruta, "rb")).load()
    fila = {
        "origen": origen,
        "dataset": dataset,
        "semilla": int(semilla_tarea if semilla_ds == "None" else semilla_ds),
        "clf": clf,
    }
    if "t_corrida" in info:  # pickles "slim" del recómputo exacto de kdc
        return {
            **fila,
            "r2": info["r2"],
            "accuracy": info["accuracy"],
            "bandwidth": info["best_bandwidth"],
            "alpha": 1.0,
        }
    datos = info[clf]
    parametros = {}
    if datos.get("busqueda") is not None:
        parametros = {
            nombre.split("__")[-1]: valor
            for nombre, valor in datos.busqueda.best_params_.items()
        }
    return {
        **fila,
        "r2": datos.get("r2", np.nan),
        "r2_temp": datos.get("r2_temp", np.nan),
        "accuracy": datos.get("accuracy", np.nan),
        **{
            parametro: parametros.get(parametro, np.nan)
            for parametro in ("alpha", "bandwidth", "n_neighbors", "weights")
        },
    }


def cargar_infos(dir_infos: Path, origen: str, clfs=None) -> pd.DataFrame:
    """Tabla de corridas de un directorio de infos (opcionalmente, solo `clfs`)."""
    import __main__
    from fkdc.datasets import Dataset

    __main__.Dataset = Dataset  # los pickles de datasets buscan `Dataset` ahí
    filas = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        for ruta in sorted(Path(dir_infos).glob("*.pkl")):
            if clfs is not None and ruta.stem.split("-")[2] not in clfs:
                continue
            try:
                filas.append(_fila_corrida(ruta, origen))
            except Exception:
                logger.warning("No se pudo leer %s", ruta.name, exc_info=True)
    return pd.DataFrame(filas)


def cargar_v6(ruta: Path = dir_raiz / "sandbox/v6/analisis-v61/corridas.csv"):
    """Corridas de v6.1 (100 semillas, kdc exacto, protocolo v6) en formato común."""
    corridas = pd.read_csv(ruta)
    corridas["semilla"] = corridas.semilla_dataset.fillna(corridas.semilla_tarea)
    corridas["semilla"] = corridas.semilla.astype(int)
    corridas["origen"] = "v6"
    columnas = ["origen", "dataset", "semilla", "clf", "r2", "r2_temp", "accuracy"]
    return corridas[columnas + ["alpha", "bandwidth", "n_neighbors", "weights"]]


# Selección secundaria (v6) de cada clasificador en la campaña heroica
selecciones_v6 = {
    "fkdc": "ciega_a_alpha",
    "fkn": "ciega_a_alpha",
    "kn": "ciega_a_alpha",
    "kdc_exacto": "parsimoniosa_h_fkdc",
    "kdc": "parsimoniosa_h_fkdc",
}


def cargar_campania(
    nombre: str = "heroica",
    kdc: str = "exacto",
    seleccion: str = "tesis",
    r2: str = "crudo",
    semillas: int | None = None,
) -> pd.DataFrame:
    """Corridas de una campaña de `fkdc.campanias` en el formato de `contrastar`.

    Los papeles del prerregistro: kdc es `kdc_exacto` (o, con `kdc="sklearn"`, el de
    `KernelDensity`); con `seleccion="v6"`, fkdc, fkn, kn y kdc toman las métricas de
    la selección secundaria; con `r2="templado"`, la columna `r2` es el R² templado;
    `semillas` se queda con las primeras de la campaña.
    """
    from fkdc import campanias

    c = campanias.campania(nombre)
    corridas = campanias.corridas(c).rename(columns={"variante": "dataset"})
    corridas.columns = [col.removeprefix("param_") for col in corridas.columns]
    if seleccion == "v6":
        for clf, regla in selecciones_v6.items():
            filas = corridas.clf.eq(clf)
            for metrica in ("r2", "accuracy", "r2_temp"):
                columna = f"{metrica}_{regla}"
                if columna in corridas:
                    corridas.loc[filas, metrica] = corridas.loc[filas, columna]
    primario = "kdc_exacto" if kdc == "exacto" else "kdc"
    corridas = corridas[corridas.clf.ne("kdc" if kdc == "exacto" else "kdc_exacto")]
    corridas["clf"] = corridas.clf.replace({primario: "kdc"})
    # Prerregistro, enmienda 1: las pruebas y el podio son entre los ocho de la tesis
    from fkdc import config

    corridas = corridas[corridas.clf.isin(config.clasificadores)]
    if r2 == "templado":
        corridas["r2"] = corridas.r2_temp
    if semillas is not None:
        corridas = corridas[corridas.semilla.isin(c.semillas[:semillas])]
    corridas["origen"] = nombre
    columnas = ["origen", "dataset", "semilla", "clf", "r2", "r2_temp", "accuracy"]
    extra = [
        p for p in ("alpha", "bandwidth", "n_neighbors", "weights") if p in corridas
    ]
    return corridas[columnas + extra]


@app.command()
def cargar(
    origen: Annotated[str, typer.Argument(help="v5, v6 o una campaña (heroica)")],
    dir_infos: Annotated[Path | None, typer.Option(help="Infos a leer")] = None,
    kdc_exacto: Annotated[
        bool, typer.Option(help="Reemplaza kdc de v5 por el recómputo exacto")
    ] = False,
    kdc: Annotated[str, typer.Option(help="Campaña: kdc exacto o sklearn")] = "exacto",
    seleccion: Annotated[str, typer.Option(help="Campaña: tesis o v6")] = "tesis",
    r2: Annotated[str, typer.Option(help="Campaña: crudo o templado")] = "crudo",
    semillas: Annotated[
        int | None, typer.Option(help="Campaña: solo las primeras N semillas")
    ] = None,
):
    """Escribe `sandbox/heroica/corridas-{origen}[-variante].csv` (caché de métricas).

    Para una campaña, la tabla primaria del prerregistro es la de las opciones por
    omisión; cada opción distinta da una sensibilidad con su propio nombre.
    """
    dir_heroica.mkdir(parents=True, exist_ok=True)
    nombre = f"corridas-{origen}"
    if origen == "v6":
        corridas = cargar_v6()
    elif origen not in ("v5",):
        corridas = cargar_campania(origen, kdc, seleccion, r2, semillas)
        for opcion, valor, omision in (
            ("kdc", kdc, "exacto"),
            ("seleccion", seleccion, "tesis"),
            ("r2", r2, "crudo"),
            ("semillas", semillas, None),
        ):
            if valor != omision:
                nombre += f"-{opcion}-{valor}"
    else:
        dir_infos = dir_infos or dir_raiz / "sandbox/v5/infos"
        corridas = cargar_infos(dir_infos, origen, clfs=list(config.clasificadores))
        if kdc_exacto:
            exacto = cargar_infos(dir_raiz / "sandbox/v5/infos-kdc-fix", origen)
            reemplazadas = corridas.clf.eq("kdc") & corridas.dataset.isin(
                exacto.dataset
            )
            corridas = pd.concat([corridas[~reemplazadas], exacto], ignore_index=True)
            nombre += "-kdc-exacto"
            logger.info(
                "kdc exacto en %d corridas (las `_std` siguen con el kdc de v5)",
                reemplazadas.sum(),
            )
    ruta = dir_heroica / f"{nombre}.csv"
    corridas.to_csv(ruta, index=False)
    logger.info("Escribió %s (%d corridas)", ruta, len(corridas))


# =====================================================================
# Afirmaciones de la tesis
# =====================================================================


@dataclass
class Afirmacion:
    """Una afirmación empírica de la tesis, lista para contrastar.

    `a` y `b` son (clasificador, dataset); `direccion` es ">", "<" o "≈" para `par`
    y `caida`; `valor` y `tolerancia` para `nivel`; `condicion` (consulta de pandas
    sobre las corridas de `a`) y `valor` (proporción) para `seleccion`.
    """

    id: str
    seccion: str
    texto: str
    tipo: str
    familia: str
    metrica: str = "r2"
    a: tuple | None = None
    b: tuple | None = None
    a0: tuple | None = None
    b0: tuple | None = None
    direccion: str | None = None
    valor: float | None = None
    tolerancia: float = 0.02
    margen: float = 0.01
    condicion: str | None = None
    extra: dict = field(default_factory=dict)


def _pares(prefijo, seccion, texto, familia, a_clf, b_clf, datasets, direccion):
    """Una afirmación `par` por dataset, con el mismo texto y la misma dirección."""
    return [
        Afirmacion(
            f"{prefijo}-{d}",
            seccion,
            f"{texto} ({d})",
            "par",
            familia,
            a=(a_clf, d),
            b=(b_clf, d),
            direccion=direccion,
        )
        for d in datasets
    ]


def afirmaciones_tesis() -> list[Afirmacion]:
    """Las afirmaciones de Resultados y Conclusiones, en el orden del texto."""
    lo = ["lunas_lo", "circulos_lo", "espirales_lo"]
    hi = ["lunas_hi", "circulos_hi", "espirales_hi"]
    af = [
        Afirmacion(
            "C01",
            "§ In toto",
            "Podio por R² mediano: fkdc 7, fkn 5, gbt 5, kdc 2, kn 2, lr 1, gnb 0",
            "podio",
            "podio",
            metrica="r2",
        ),
        Afirmacion(
            "C02",
            "§ In toto",
            "Podio por exactitud mediana: svc 7, gbt 6, fkn 5, kn 5, ...",
            "podio",
            "podio",
            metrica="accuracy",
        ),
    ]
    sec_lo = "§ Lunas, círculos y espirales"
    for d in ("lunas_lo", "espirales_lo"):
        af.append(
            Afirmacion(
                f"C03-{d}",
                sec_lo,
                f"fkdc saca ventaja en R² sobre kdc en {d}",
                "par",
                "fermat-vs-euclidea",
                a=("fkdc", d),
                b=("kdc", d),
                direccion=">",
            )
        )
    for d in lo:
        af.append(
            Afirmacion(
                f"C04-{d}",
                sec_lo,
                f"fkdc y kdc tienen exactitud comparable en {d}",
                "par",
                "equivalencias",
                metrica="accuracy",
                a=("fkdc", d),
                b=("kdc", d),
                direccion="≈",
            )
        )
        for ref in ("gnb", "lr"):
            af.append(
                Afirmacion(
                    f"C05-{ref}-{d}",
                    sec_lo,
                    f"{ref} rinde mal en {d} (fkdc lo supera en R²)",
                    "par",
                    "k-vs-referencia",
                    a=("fkdc", d),
                    b=(ref, d),
                    direccion=">",
                )
            )
    for d in ("lunas_lo", "circulos_lo"):
        af.append(
            Afirmacion(
                f"C06-{d}",
                "§ Ablación",
                f"fkn y kn casi no se distinguen en R² en {d}",
                "par",
                "equivalencias",
                a=("fkn", d),
                b=("kn", d),
                direccion="≈",
            )
        )
    af += [
        Afirmacion(
            "C07-alpha",
            "§ Hiperparámetros en lunas_lo",
            "fkdc elige α = 1 en todas las semillas de lunas_lo",
            "seleccion",
            "seleccion",
            a=("fkdc", "lunas_lo"),
            condicion="alpha == 1",
            valor=1.0,
        ),
        Afirmacion(
            "C07-h",
            "§ Hiperparámetros en lunas_lo",
            "h de fkdc ligera pero consistentemente menor que el de kdc en lunas_lo",
            "par",
            "seleccion-h",
            metrica="bandwidth",
            a=("fkdc", "lunas_lo"),
            b=("kdc", "lunas_lo"),
            direccion="<",
        ),
        Afirmacion(
            "C08-lunas_hi-r2",
            "§ Efectos de aumentar el ruido",
            "En lunas_hi gbt alcanza un R² marginalmente mejor que fkdc",
            "par",
            "k-vs-referencia",
            a=("gbt", "lunas_hi"),
            b=("fkdc", "lunas_hi"),
            direccion=">",
        ),
        Afirmacion(
            "C08-lunas_hi-acc",
            "§ Efectos de aumentar el ruido",
            "En lunas_hi la familia K tiene exactitud algo mejor que gbt",
            "par",
            "k-vs-referencia",
            metrica="accuracy",
            a=("fkdc", "lunas_hi"),
            b=("gbt", "lunas_hi"),
            direccion=">",
        ),
        *(
            Afirmacion(
                f"C08-circulos_hi-{m}",
                "§ Efectos de aumentar el ruido",
                f"En circulos_hi gbt es superior a fkdc en {m}",
                "par",
                "k-vs-referencia",
                metrica=m,
                a=("gbt", "circulos_hi"),
                b=("fkdc", "circulos_hi"),
                direccion=">",
            )
            for m in ("r2", "accuracy")
        ),
        Afirmacion(
            "C08-circulos_hi-nivel",
            "§ Efectos de aumentar el ruido",
            "R² de gbt en circulos_hi ≈ 0.09",
            "nivel",
            "niveles",
            a=("gbt", "circulos_hi"),
            valor=0.09,
            tolerancia=0.02,
        ),
        Afirmacion(
            "C08-espirales_hi-acc",
            "§ Efectos de aumentar el ruido",
            "En espirales_hi svc tiene la mejor exactitud, por encima de fkdc",
            "par",
            "k-vs-referencia",
            metrica="accuracy",
            a=("svc", "espirales_hi"),
            b=("fkdc", "espirales_hi"),
            direccion=">",
        ),
        *_pares(
            "C09",
            "§ Efectos de aumentar el ruido",
            "Ventajas de fkdc sobre kdc (casi) nulas con ruido alto",
            "equivalencias",
            "fkdc",
            "kdc",
            hi,
            "≈",
        ),
        Afirmacion(
            "C10-r2",
            "§ anteojos",
            "fkdc saca una ventaja pequeña pero consistente en R² sobre kdc",
            "par",
            "fermat-vs-euclidea",
            a=("fkdc", "anteojos"),
            b=("kdc", "anteojos"),
            direccion=">",
        ),
        Afirmacion(
            "C10-acc",
            "§ anteojos",
            "Exactitud de fkdc ≈ 97 %",
            "nivel",
            "niveles",
            metrica="accuracy",
            a=("fkdc", "anteojos"),
            valor=0.97,
            tolerancia=0.01,
        ),
        *(
            Afirmacion(
                f"C11-{clf}",
                "§ Eslabones",
                f"{clf} alcanza R² ≈ 1 en eslabones_0",
                "nivel",
                "niveles",
                a=(clf, "eslabones_0"),
                valor=0.99,
                tolerancia=0.02,
            )
            for clf in clfs_k
        ),
        Afirmacion(
            "C12-r2",
            "§ Hélices",
            "fkdc mejora en R² a kdc por casi 5 pp en casi todas las semillas",
            "par",
            "fermat-vs-euclidea",
            a=("fkdc", "helices_0"),
            b=("kdc", "helices_0"),
            direccion=">",
            extra={"magnitud_tesis": 0.05},
        ),
        *(
            Afirmacion(
                f"C12-nulo-{clf}",
                "§ Hélices",
                f"{clf} no se distingue del clasificador trivial en helices_0",
                "nivel",
                "niveles",
                a=(clf, "helices_0"),
                valor=0.0,
                tolerancia=0.05,
            )
            for clf in ("gnb", "lr")
        ),
        Afirmacion(
            "C13",
            "§ Efecto de D_Q,α en las vecindades de kn",
            "fkn supera a kn en helices_0 (vecindarios a lo largo de la hebra)",
            "par",
            "fermat-vs-euclidea",
            a=("fkn", "helices_0"),
            b=("kn", "helices_0"),
            direccion=">",
        ),
        Afirmacion(
            "C14-r2",
            "§ Pionono",
            "Sin diferencia significativa entre kdc y fkdc en pionono_0",
            "par",
            "equivalencias",
            a=("fkdc", "pionono_0"),
            b=("kdc", "pionono_0"),
            direccion="≈",
        ),
        Afirmacion(
            "C14-alpha",
            "§ Pionono",
            "fkdc eligió α = 1 en todas las semillas de pionono_0",
            "seleccion",
            "seleccion",
            a=("fkdc", "pionono_0"),
            condicion="alpha == 1",
            valor=1.0,
        ),
        Afirmacion(
            "C15-nivel",
            "§ Hueveras",
            "La familia K es la única con R² no nulo en hueveras_0 (0.25–0.30)",
            "nivel",
            "niveles",
            a=("fkdc", "hueveras_0"),
            valor=0.275,
            tolerancia=0.03,
        ),
        Afirmacion(
            "C15-acc",
            "§ Hueveras",
            "svc es ligeramente mejor en exactitud que la familia K en hueveras_0",
            "par",
            "k-vs-referencia",
            metrica="accuracy",
            a=("svc", "hueveras_0"),
            b=("fkdc", "hueveras_0"),
            direccion=">",
        ),
        Afirmacion(
            "C15-fkn",
            "§ Hueveras",
            "fkn ofrece una diferencia significativa en R² sobre kn en hueveras_0",
            "par",
            "fermat-vs-euclidea",
            a=("fkn", "hueveras_0"),
            b=("kn", "hueveras_0"),
            direccion=">",
        ),
    ]
    sec_15d = "§ Efecto de aumentar la dimensión ambiente"
    for familia_ds in ("pionono", "eslabones"):
        for clf in ("fkdc", "fkn"):
            for ref in ("gbt", "gnb", "lr"):
                af.append(
                    Afirmacion(
                        f"C16-{familia_ds}-{clf}-{ref}",
                        sec_15d,
                        f"Con 12 dims de ruido {clf} cae más que {ref} ({familia_ds})",
                        "caida",
                        "dimension",
                        a=(clf, f"{familia_ds}_12"),
                        a0=(clf, f"{familia_ds}_0"),
                        b=(ref, f"{familia_ds}_12"),
                        b0=(ref, f"{familia_ds}_0"),
                        direccion="<",
                    )
                )
    for d in ("helices_12", "hueveras_12"):
        for clf in [*clfs_k, *clfs_referencia]:
            if clf == "svc":
                continue
            af.append(
                Afirmacion(
                    f"C17-{d}-{clf}",
                    sec_15d,
                    f"En {d} ningún clasificador se distingue del azar ({clf})",
                    "nivel",
                    "niveles",
                    a=(clf, d),
                    valor=0.0,
                    tolerancia=0.05,
                )
            )
    af += [
        Afirmacion(
            "C17-pionono_12",
            sec_15d,
            "R² de la familia K se desploma a ≈ 0.1 en pionono_12",
            "nivel",
            "niveles",
            a=("fkdc", "pionono_12"),
            valor=0.10,
            tolerancia=0.03,
        ),
        Afirmacion(
            "C17-eslabones_12",
            sec_15d,
            "R² de la familia K se desploma a ≈ 0.25 en eslabones_12",
            "nivel",
            "niveles",
            a=("fkdc", "eslabones_12"),
            valor=0.25,
            tolerancia=0.03,
        ),
        Afirmacion(
            "C18-lr",
            "§ pinguinos",
            "lr domina en pinguinos con R² ≈ 0.96",
            "nivel",
            "niveles",
            a=("lr", "pinguinos"),
            valor=0.96,
            tolerancia=0.02,
        ),
        Afirmacion(
            "C18-k",
            "§ pinguinos",
            "fkdc queda en R² ≈ 0.42 sobre los atributos crudos",
            "nivel",
            "niveles",
            a=("fkdc", "pinguinos"),
            valor=0.42,
            tolerancia=0.03,
        ),
        Afirmacion(
            "C18-std-r2",
            "§ pinguinos",
            "Estandarizando, fkdc pasa de R² ≈ 0.42 a ≈ 0.96",
            "par",
            "escala",
            a=("fkdc", "pinguinos_std"),
            b=("fkdc", "pinguinos"),
            direccion=">",
        ),
        Afirmacion(
            "C18-std-nivel",
            "§ pinguinos",
            "fkdc estandarizado alcanza R² ≈ 0.96 y exactitud ≈ 99 %",
            "nivel",
            "niveles",
            a=("fkdc", "pinguinos_std"),
            valor=0.96,
            tolerancia=0.02,
        ),
        *(
            Afirmacion(
                f"C18-inmune-{clf}",
                "§ pinguinos",
                f"{clf} no se mueve al estandarizar pinguinos",
                "par",
                "equivalencias",
                a=(clf, "pinguinos_std"),
                b=(clf, "pinguinos"),
                direccion="≈",
            )
            for clf in ("lr", "gbt")
        ),
        Afirmacion(
            "C19-iris",
            "§ iris y vino",
            "En iris estandarizar empeora a fkn (R² 0.90 → 0.84)",
            "par",
            "escala",
            a=("fkn", "iris_std"),
            b=("fkn", "iris"),
            direccion="<",
        ),
        Afirmacion(
            "C19-vino-gbt",
            "§ iris y vino",
            "En vino gbt domina con R² ≈ 0.90",
            "nivel",
            "niveles",
            a=("gbt", "vino"),
            valor=0.90,
            tolerancia=0.02,
        ),
        *(
            Afirmacion(
                f"C19-vino-{clf}",
                "§ iris y vino",
                f"En vino {clf} mejora al estandarizar",
                "par",
                "escala",
                a=(clf, "vino_std"),
                b=(clf, "vino"),
                direccion=">",
            )
            for clf in ("lr", "fkdc", "kdc", "fkn", "kn")
        ),
        Afirmacion(
            "C20-r2",
            "§ digitos",
            "fkdc (R² ≈ 0.98) apenas por encima de kdc (≈ 0.97) en digitos",
            "par",
            "fermat-vs-euclidea",
            a=("fkdc", "digitos"),
            b=("kdc", "digitos"),
            direccion=">",
        ),
        Afirmacion(
            "C20-std",
            "§ digitos",
            "Estandarizar digitos perjudica a fkdc (R² 0.98 → 0.87)",
            "par",
            "escala",
            a=("fkdc", "digitos_std"),
            b=("fkdc", "digitos"),
            direccion="<",
        ),
        Afirmacion(
            "C20-alpha-fkdc",
            "§ digitos",
            "fkdc eligió α = 1 en las 25 semillas de digitos",
            "seleccion",
            "seleccion",
            a=("fkdc", "digitos"),
            condicion="alpha == 1",
            valor=1.0,
        ),
        Afirmacion(
            "C20-alpha-fkn",
            "§ digitos",
            "La parsimonia devolvió α = 1 para fkn en 13 de las 25 semillas",
            "seleccion",
            "seleccion",
            a=("fkn", "digitos"),
            condicion="alpha == 1",
            valor=13 / 25,
        ),
        Afirmacion(
            "C21-r2",
            "§ mnist",
            "kdc (R² ≈ 0.77) supera a fkdc (≈ 0.74) en mnist",
            "par",
            "fermat-vs-euclidea",
            a=("kdc", "mnist"),
            b=("fkdc", "mnist"),
            direccion=">",
        ),
        Afirmacion(
            "C21-alpha-fkn",
            "§ mnist",
            "Para fkn, α = 1 bajo parsimonia en 18 de las 25 semillas de mnist",
            "seleccion",
            "seleccion",
            a=("fkn", "mnist"),
            condicion="alpha == 1",
            valor=18 / 25,
        ),
        *(
            Afirmacion(
                f"C21-kdc-kn-{m}",
                "§ mnist",
                f"kdc supera a kn en {m} en mnist",
                "par",
                "k-vs-referencia",
                metrica=m,
                a=("kdc", "mnist"),
                b=("kn", "mnist"),
                direccion=">",
            )
            for m in ("r2", "accuracy")
        ),
    ]
    sec_conc = "§ Conclusiones"
    for d in ("eslabones_0", "helices_0"):
        for a_clf, b_clf in (("fkdc", "kdc"), ("fkn", "kn")):
            af.append(
                Afirmacion(
                    f"C22-{a_clf}-{d}",
                    sec_conc,
                    f"La distancia de Fermat aumenta la varianza entre semillas ({d})",
                    "dispersion",
                    "dispersion",
                    a=(a_clf, d),
                    b=(b_clf, d),
                    direccion=">",
                )
            )
    return af


# =====================================================================
# Pruebas
# =====================================================================


def corregir_holm(pvalores) -> np.ndarray:
    """p-valores ajustados por Holm; NaN cuenta como 1 (sigue en la familia)."""
    p = np.where(np.isnan(np.asarray(pvalores, float)), 1.0, pvalores)
    orden = np.argsort(p, kind="stable")
    ajustados = np.minimum(
        1.0, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[orden])
    )
    salida = np.empty(len(p))
    salida[orden] = ajustados
    return salida


def corregir_bh(pvalores) -> np.ndarray:
    """p-valores ajustados por Benjamini–Hochberg; NaN cuenta como 1."""
    p = np.where(np.isnan(np.asarray(pvalores, float)), 1.0, pvalores)
    orden = np.argsort(p, kind="stable")
    crudos = p[orden] * len(p) / np.arange(1, len(p) + 1)
    ajustados = np.minimum(1.0, np.minimum.accumulate(crudos[::-1])[::-1])
    salida = np.empty(len(p))
    salida[orden] = ajustados
    return salida


def _serie(corridas: pd.DataFrame, clf: str, dataset: str, metrica: str):
    """Valores de `metrica` por semilla de `clf` en `dataset`."""
    filas = corridas[corridas.clf.eq(clf) & corridas.dataset.eq(dataset)]
    return pd.to_numeric(filas.set_index("semilla")[metrica], errors="coerce")


def _hodges_lehmann(diferencias: np.ndarray) -> float:
    """Mediana de los promedios de Walsh (estimador asociado a Wilcoxon)."""
    i, j = np.triu_indices(len(diferencias))
    return float(np.median((diferencias[i] + diferencias[j]) / 2))


def _bootstrap(valores: np.ndarray, estadistico, n: int = n_bootstrap) -> np.ndarray:
    rng = np.random.default_rng(semilla_bootstrap)
    indices = rng.integers(0, len(valores), size=(n, len(valores)))
    return np.array([estadistico(valores[fila]) for fila in indices])


def _wilcoxon(diferencias: np.ndarray) -> float:
    if len(diferencias) == 0 or np.all(diferencias == 0):
        return np.nan
    try:
        return float(wilcoxon(diferencias).pvalue)
    except ValueError:
        return np.nan


def probar_par(corridas, af: Afirmacion) -> dict:
    a = _serie(corridas, *af.a, af.metrica)
    b = _serie(corridas, *af.b, af.metrica)
    comunes = a.dropna().index.intersection(b.dropna().index)
    d = (a[comunes] - b[comunes]).to_numpy(float)
    d = np.where(np.abs(d) < tolerancia_empate, 0.0, d)
    if len(d) == 0:
        return {"n": 0}
    resultado = {
        "n": len(d),
        "a_mediana": float(a[comunes].median()),
        "b_mediana": float(b[comunes].median()),
        "delta_mediana": float(np.median(d)),
        "delta_media": float(np.mean(d)),
        "hodges_lehmann": _hodges_lehmann(d),
        "ganadas": int((d > 0).sum()),
        "perdidas": int((d < 0).sum()),
        "empates": int((d == 0).sum()),
    }
    if af.direccion == "≈":
        medias = _bootstrap(d, np.mean)
        bajo, alto = np.quantile(medias, [0.05, 0.95])
        resultado.update(ic_bajo=bajo, ic_alto=alto, p=np.nan)
        resultado["equivalente"] = bool(-af.margen <= bajo and alto <= af.margen)
    else:
        resultado["p"] = _wilcoxon(d)
    return resultado


def probar_caida(corridas, af: Afirmacion) -> dict:
    caida_a = _serie(corridas, *af.a, af.metrica) - _serie(corridas, *af.a0, af.metrica)
    caida_b = _serie(corridas, *af.b, af.metrica) - _serie(corridas, *af.b0, af.metrica)
    d = (caida_a - caida_b).dropna().to_numpy(float)
    d = np.where(np.abs(d) < tolerancia_empate, 0.0, d)
    if len(d) == 0:
        return {"n": 0}
    return {
        "n": len(d),
        "a_mediana": float(caida_a.median()),
        "b_mediana": float(caida_b.median()),
        "delta_mediana": float(np.median(d)),
        "hodges_lehmann": _hodges_lehmann(d),
        "ganadas": int((d > 0).sum()),
        "perdidas": int((d < 0).sum()),
        "p": _wilcoxon(d),
    }


def probar_nivel(corridas, af: Afirmacion) -> dict:
    valores = _serie(corridas, *af.a, af.metrica).dropna().to_numpy(float)
    if len(valores) == 0:
        return {"n": 0}
    medianas = _bootstrap(valores, np.median)
    mediana = float(np.median(valores))
    return {
        "n": len(valores),
        "a_mediana": mediana,
        "ic_bajo": float(np.quantile(medianas, 0.025)),
        "ic_alto": float(np.quantile(medianas, 0.975)),
        "replica_nivel": bool(abs(mediana - af.valor) <= af.tolerancia),
    }


def probar_seleccion(corridas, af: Afirmacion) -> dict:
    filas = corridas[corridas.clf.eq(af.a[0]) & corridas.dataset.eq(af.a[1])]
    filas = filas.assign(alpha=pd.to_numeric(filas.alpha, errors="coerce"))
    if filas.empty or filas.alpha.isna().all():
        return {"n": 0}
    n, k = len(filas), int(len(filas.query(af.condicion)))
    bajo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    alto = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return {
        "n": n,
        "cumplen": k,
        "proporcion": k / n,
        "ic_bajo": float(bajo),
        "ic_alto": float(alto),
        "replica_nivel": bool(bajo <= af.valor <= alto),
    }


def probar_dispersion(corridas, af: Afirmacion) -> dict:
    a = _serie(corridas, *af.a, af.metrica)
    b = _serie(corridas, *af.b, af.metrica)
    comunes = a.dropna().index.intersection(b.dropna().index)
    if len(comunes) == 0:
        return {"n": 0}
    pares = np.column_stack([a[comunes], b[comunes]])

    def cociente_iqr(muestra):
        iqr_a, iqr_b = (
            np.subtract(*np.quantile(muestra[:, i], [0.75, 0.25])) for i in (0, 1)
        )
        return iqr_a / iqr_b if iqr_b > 0 else np.nan

    cocientes = _bootstrap(pares, cociente_iqr)
    return {
        "n": len(comunes),
        "cociente_iqr": cociente_iqr(pares),
        "ic_bajo": float(np.nanquantile(cocientes, 0.025)),
        "ic_alto": float(np.nanquantile(cocientes, 0.975)),
    }


def _podio_tesis(metrica: str) -> dict[str, set]:
    """Ganadores por dataset según el CSV de la tesis."""
    ruta = dir_datos_tesis / f"mejor-clf-por-dataset-segun-{metrica}-mediano.csv"
    ganadores = {}
    for fila in pd.read_csv(ruta).itertuples():
        for dataset in fila.datasets.split(", "):
            ganadores.setdefault(dataset, set()).add(fila.clf)
    return ganadores


def probar_podio(corridas, af: Afirmacion) -> list[dict]:
    """Una fila por dataset del podio de la tesis, más una con los totales."""
    esperados = _podio_tesis(af.metrica)
    filas, conteo = [], {}
    rng = np.random.default_rng(semilla_bootstrap)
    for dataset, ganadores_tesis in esperados.items():
        tabla = corridas[corridas.dataset.eq(dataset)].pivot_table(
            index="semilla", columns="clf", values=af.metrica
        )
        tabla = tabla.dropna(axis=1, how="all")
        if not ganadores_tesis <= set(tabla.columns) or tabla.shape[1] < 2:
            filas.append({"id": f"{af.id}-{dataset}", "dataset": dataset, "n": 0})
            continue
        medianas = tabla.median().round(4)
        ganadores = set(medianas[medianas.eq(medianas.max())].index)
        for clf in ganadores:
            conteo[clf] = conteo.get(clf, 0) + 1
        sigue = 0
        for _ in range(n_bootstrap // 4):
            muestra = tabla.iloc[rng.integers(0, len(tabla), len(tabla))]
            med = muestra.median().round(4)
            sigue += bool(ganadores_tesis & set(med[med.eq(med.max())].index))
        orden = medianas.sort_values(ascending=False)
        primero = orden.index[0]
        segundo = next((c for c in orden.index if orden[c] < orden[primero]), None)
        p = np.nan
        if segundo is not None:
            d = (tabla[primero] - tabla[segundo]).dropna().to_numpy(float)
            p = _wilcoxon(np.where(np.abs(d) < tolerancia_empate, 0.0, d))
        filas.append(
            {
                "id": f"{af.id}-{dataset}",
                "dataset": dataset,
                "n": len(tabla),
                "ganadores_tesis": ", ".join(sorted(ganadores_tesis)),
                "ganadores": ", ".join(sorted(ganadores)),
                "coincide": bool(ganadores & ganadores_tesis),
                "prob_bootstrap": sigue / (n_bootstrap // 4),
                "segundo": segundo,
                "p": p,
                # Sin los clasificadores de la tesis el podio es parcial (v6 solo
                # corrió la familia K); svc no tiene R²
                "completo": set(config.clasificadores)
                - ({"svc"} if af.metrica == "r2" else set())
                <= set(tabla.columns),
            }
        )
    filas.append(
        {
            "id": f"{af.id}-totales",
            "dataset": "(todos)",
            "ganadores": ", ".join(f"{c} {k}" for c, k in sorted(conteo.items())),
        }
    )
    return filas


def _veredicto(fila: pd.Series, af: Afirmacion) -> str:
    if not fila.get("n"):
        return "sin datos"
    if af.tipo in ("nivel", "seleccion"):
        return "replica" if fila.get("replica_nivel") else "no replica"
    if af.tipo == "dispersion":
        if fila.ic_bajo > 1:
            return "replica"
        return "contradice" if fila.ic_alto < 1 else "no concluyente"
    if af.direccion == "≈":
        return "replica" if fila.get("equivalente") else "no concluyente"
    if af.tipo == "podio":
        veredicto = "replica" if fila.get("coincide") else "no replica"
        return veredicto if fila.get("completo") else f"{veredicto} (parcial)"
    signo = np.sign(fila.hodges_lehmann)
    esperado = 1 if af.direccion == ">" else -1
    if fila.p_holm < nivel_significacion and signo == esperado:
        return "replica"
    if fila.p_holm < nivel_significacion and signo == -esperado:
        return "contradice"
    return "no concluyente"


pruebas = {
    "par": probar_par,
    "caida": probar_caida,
    "nivel": probar_nivel,
    "seleccion": probar_seleccion,
    "dispersion": probar_dispersion,
}


def contrastar(corridas: pd.DataFrame, afirmaciones=None) -> pd.DataFrame:
    """Contrasta todas las afirmaciones contra una tabla de corridas."""
    afirmaciones = afirmaciones or afirmaciones_tesis()
    por_id = {af.id: af for af in afirmaciones}
    filas = []
    for af in afirmaciones:
        base = {
            "id": af.id,
            "seccion": af.seccion,
            "texto": af.texto,
            "tipo": af.tipo,
            "familia": af.familia,
            "metrica": af.metrica,
        }
        if af.tipo == "podio":
            for fila in probar_podio(corridas, af):
                por_id[fila["id"]] = af
                filas.append({**base, **fila})
        else:
            filas.append({**base, **pruebas[af.tipo](corridas, af)})
    resultado = pd.DataFrame(filas)
    for columna in ("p", "n"):
        if columna not in resultado:
            resultado[columna] = np.nan
    resultado["p_holm"] = np.nan
    resultado["p_bh"] = np.nan
    # La familia de cada corrección son las pruebas con p-valor (direccionales y
    # ganador vs. segundo del podio) con datos; un p-valor NaN (todas empatadas)
    # cuenta como 1.
    direccionales = resultado.id.map(
        lambda i: por_id[i].direccion in (">", "<") or por_id[i].tipo == "podio"
    )
    con_p = resultado[direccionales & resultado.n.fillna(0).gt(0)]
    con_p = con_p[~con_p.id.str.endswith("totales")]
    for _, grupo in con_p.groupby("familia"):
        resultado.loc[grupo.index, "p_holm"] = corregir_holm(grupo.p)
        resultado.loc[grupo.index, "p_bh"] = corregir_bh(grupo.p)
    resultado["veredicto"] = [
        "" if fila.id.endswith("totales") else _veredicto(fila, por_id[fila.id])
        for _, fila in resultado.iterrows()
    ]
    return resultado


@app.command()
def replicar(
    nombres: Annotated[
        list[str], typer.Argument(help="Tablas `sandbox/heroica/<nombre>.csv`")
    ],
):
    """Contrasta las afirmaciones de la tesis con cada tabla de corridas."""
    resumenes = {}
    for nombre in nombres:
        corridas = pd.read_csv(dir_heroica / f"{nombre}.csv")
        resultado = contrastar(corridas)
        ruta = dir_heroica / f"replica-{nombre.removeprefix('corridas-')}.csv"
        resultado.to_csv(ruta, index=False, float_format="%.4g")
        logger.info("Escribió %s", ruta)
        resumenes[nombre] = resultado.set_index("id").veredicto
    tabla = pd.DataFrame(resumenes)
    tabla = tabla[tabla.ne("").any(axis=1)]
    print(tabla.to_markdown())
    print(tabla.apply(pd.Series.value_counts).fillna(0).astype(int).to_markdown())


if __name__ == "__main__":
    app()
