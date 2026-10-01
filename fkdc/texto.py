"""
Clasificadores de texto (Laya, GLiNER2.5-Decide, Jev) como líneas de base sobre las
tareas de la tesis.

Laya, GLiNER2.5-Decide y Jev son modelos de decisión sobre texto: reciben un texto y
un conjunto de opciones definido en cada llamada, y devuelven una probabilidad por
opción. Para aplicarlos a los
datasets tabulares de la tesis, cada observación se serializa como texto
(`x1=0.53; x2=-1.2`, o con los nombres de los atributos cuando el dataset los tiene)
y se pregunta «¿A qué clase pertenece la observación?», con una opción por clase.

El régimen decide qué acompaña a la observación en el estado:

- `zero-shot`: nada; solo los nombres de atributos y de clases pueden aportar.
- `nombre`: el nombre del dataset («dos espirales entrelazadas»).
- `descripcion`: el nombre y una descripción del proceso generador tomada de la
  tesis; en las opciones, qué componente de la variedad es cada clase.
- `codigo`: el nombre y el código fuente de la función generadora, con los
  parámetros usados (solo datasets sintéticos).
- `in-context`: observaciones etiquetadas del pliego de entrenamiento, tantas como
  quepan en el presupuesto de tokens (orden estratificado con semilla fija).
- `descripcion-ic`: la descripción y los ejemplos etiquetados.

Las tareas son las de la tesis (mismos datasets, semillas y `train_test_split`) y
las métricas, las de `Tarea.evaluar`. Los infos van a `sandbox/fundacionales/infos`
con clasificador `{familia}_{régimen}`: `laya_desc`, `gliner_zs`, `laya_en_desc`
(checkpoint y texto en inglés), `glinerft_zs` (GLiNER ajustado sobre el pliego de
entrenamiento, `GlinerAjustadoClassifier`).

Uso, desde la raíz del repositorio:
    python -m fkdc.texto correr descripcion --patron 'lunas_lo-1075-*'
    python -m fkdc.texto correr zero-shot --familia gliner_ajustado --epocas 10
    python -m fkdc.texto resumir --clf laya_desc --clf laya_zs
    python -m fkdc.texto estimar laya_desc
"""

import ast
import hashlib
import inspect
import logging
import math
import os
import pickle
import re
from fnmatch import fnmatch
from functools import cache
from importlib import import_module
from pathlib import Path
from time import time
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch

from fkdc import config, dir_raiz
from fkdc.datasets import Dataset

logger = logging.getLogger(__name__)

dir_fundacionales = dir_raiz / "sandbox/fundacionales"
dir_infos = dir_fundacionales / "infos"
dir_datasets = config.dir_ejecucion.parent / "datasets"

# Checkpoints de Laya: todos viven en el mismo repositorio, cada uno en su subcarpeta.
repo_laya = "convaiinnovations/laya"
# Revisión revisada que publica `laya.PINNED_REVISIONS` (laya 0.3.22)
revision_laya = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
subcarpetas_laya = {"multilingual": "multilingual", "english": None}
# Contexto máximo que admiten los codificadores (ModernBERT-large y mmBERT-base)
max_len_tope = 8192

# Régimen -> abreviatura en el nombre del clasificador (sin guiones: el nombre de
# los infos se parte por "-")
regimenes = {
    "zero-shot": "zs",
    "nombre": "nombre",
    "descripcion": "desc",
    "codigo": "codigo",
    "in-context": "ic",
    "descripcion-ic": "desc_ic",
}
regimenes_con_ejemplos = {"in-context", "descripcion-ic"}
regimenes_con_descripcion = {"descripcion", "descripcion-ic"}
regimenes_con_nombre = {"nombre", "descripcion", "codigo", "descripcion-ic"}

textos = {
    "es": Bunch(
        pregunta="¿A qué clase pertenece la observación?",
        dataset="Dataset: {}.",
        codigo="Código con el que se generaron los datos:",
        ejemplos="Ejemplos etiquetados (observación -> clase):",
        observacion="Observación: {}",
        clase="clase {}",
        clase_codigo="la clase y = {} del código",
        estandarizado=(
            "Las variables están estandarizadas (media 0 y desvío 1 en el pliego de"
            " entrenamiento)."
        ),
        ruido_12=(
            " Además, a cada observación se le agregaron 12 dimensiones de ruido"
            " normal independientes de la clase (x4 a x15), con una escala comparable"
            " a la de las tres primeras."
        ),
    ),
    "en": Bunch(
        pregunta="Which class does the observation belong to?",
        dataset="Dataset: {}.",
        codigo="Code that generated the data:",
        ejemplos="Labelled examples (observation -> class):",
        observacion="Observation: {}",
        clase="class {}",
        clase_codigo="class y = {} in the code",
        estandarizado=(
            "The variables are standardised (mean 0 and standard deviation 1 on the"
            " training split)."
        ),
        ruido_12=(
            " In addition, 12 dimensions of normal noise independent of the class"
            " (x4 to x15) were appended to each observation, on a scale comparable"
            " to that of the first three."
        ),
    ),
}

_ruido_es = (
    " Es una variedad de dimensión intrínseca {d_mm} en {espacio}, con K = {k} clases"
    " de {n} observaciones, y a cada punto se le suma ruido blanco normal de desvío"
    " σ = {sigma}."
)
_ruido_en = (
    " It is a manifold of intrinsic dimension {d_mm} in {espacio}, with K = {k}"
    " classes of {n} observations each, and normal white noise with standard"
    " deviation σ = {sigma} is added to each point."
)
_plano = {"es": "el plano (d = 2)", "en": "the plane (d = 2)"}
_espacio = {"es": "el espacio (d = 3)", "en": "space (d = 3)"}


def _sintetico(es, en, d_mm, d, k, n):
    """Descripción de un dataset sintético: forma, dimensiones, clases y ruido."""
    espacio = _plano if d == 2 else _espacio
    return {
        "es": es + _ruido_es.format(d_mm=d_mm, espacio=espacio["es"], k=k, n=n,
                                    sigma="{sigma}"),
        "en": en + _ruido_en.format(d_mm=d_mm, espacio=espacio["en"], k=k, n=n,
                                    sigma="{sigma}"),
    }  # fmt: skip


# Fichas por familia de datasets. Las descripciones parafrasean la tesis
# (`docs/tesis.typ`, sección de cada dataset en «Resultados»); las correspondencias
# entre clases y componentes (`clases`) y los desvíos de los datasets 3D, que la tesis
# no enuncia, salen de las funciones generadoras (`fkdc/datasets.py`, scikit-learn).
fichas = {
    "lunas": Bunch(
        nombre={"es": "dos lunas", "en": "two moons"},
        descripcion=_sintetico(
            "Dos «lunas»: semicírculos no superpuestos, cada uno con su centro en un"
            " extremo del semicírculo opuesto.",
            'Two "moons": non-overlapping half-circles, each centred at one end of'
            " the opposite half-circle.",
            d_mm=1,
            d=2,
            k=2,
            n=400,
        ),
        clases={
            "es": {
                "0": "la luna de arriba, un arco con forma de ∩",
                "1": "la luna de abajo, un arco con forma de ∪",
            },
            "en": {
                "0": "the upper moon, an arc shaped like ∩",
                "1": "the lower moon, an arc shaped like ∪",
            },
        },
        funciones=["sklearn.datasets:make_moons"],
        fuente="tesis, «Lunas, círculos y espirales» (σ en «Efectos de aumentar el"
        " ruido»); clases: sklearn.datasets.make_moons",
    ),
    "circulos": Bunch(
        nombre={"es": "dos círculos concéntricos", "en": "two concentric circles"},
        descripcion=_sintetico(
            "Dos círculos concéntricos.",
            "Two concentric circles.",
            d_mm=1,
            d=2,
            k=2,
            n=400,
        ),
        clases={
            "es": {"0": "el círculo exterior", "1": "el círculo interior"},
            "en": {"0": "the outer circle", "1": "the inner circle"},
        },
        funciones=["sklearn.datasets:make_circles"],
        fuente="tesis, «Lunas, círculos y espirales»; clases:"
        " sklearn.datasets.make_circles",
    ),
    "espirales": Bunch(
        nombre={"es": "dos espirales entrelazadas", "en": "two interleaved spirals"},
        descripcion=_sintetico(
            "Dos espirales con el mismo origen y sentido de rotación, pero desfasadas"
            " medio giro.",
            "Two spirals with the same origin and direction of rotation, but half a"
            " turn out of phase.",
            d_mm=1,
            d=2,
            k=2,
            n=400,
        ),
        clases={
            "es": {
                "0": "la espiral que sale del origen hacia x1 positivo",
                "1": "la espiral que sale del origen hacia x1 negativo (la otra,"
                " rotada media vuelta)",
            },
            "en": {
                "0": "the spiral that leaves the origin towards positive x1",
                "1": "the spiral that leaves the origin towards negative x1 (the"
                " other one, rotated half a turn)",
            },
        },
        funciones=["fkdc.datasets:_dos_muestras", "fkdc.datasets:hacer_espirales"],
        fuente="tesis, «Lunas, círculos y espirales»; clases: hacer_espirales",
    ),
    "anteojos": Bunch(
        nombre={"es": "anteojos", "en": "eyeglasses"},
        descripcion={
            "es": "Un dataset sintético de tres clases con forma de anteojos, en el"
            " plano (d = 2): la mitad de las observaciones está sobre el marco (dos"
            " aros unidos por un puente, con ruido blanco normal de desvío σ ="
            " {sigma}) y una cuarta parte en cada pupila, una nube gaussiana en el"
            " centro de cada aro.",
            "en": "A synthetic three-class dataset shaped like eyeglasses, in the"
            " plane (d = 2): half of the observations lie on the frame (two rings"
            " joined by a bridge, with normal white noise of standard deviation σ ="
            " {sigma}) and a quarter in each pupil, a Gaussian cloud at the centre"
            " of each ring.",
        },
        clases={
            "es": {
                "0": "el marco de los anteojos",
                "1": "la pupila izquierda",
                "2": "la pupila derecha",
            },
            "en": {
                "0": "the frame of the glasses",
                "1": "the left pupil",
                "2": "the right pupil",
            },
        },
        funciones=[
            "fkdc.utils:_longitud_elipse",
            "fkdc.utils:arco",
            "fkdc.utils:anteojos",
            "fkdc.datasets:hacer_anteojos",
        ],
        fuente="tesis, «anteojos»; marco y pupilas: hacer_anteojos",
    ),
    "eslabones": Bunch(
        nombre={"es": "dos eslabones enlazados", "en": "two linked rings"},
        descripcion=_sintetico(
            "Dos eslabones de cadena: dos anillos enlazados en el espacio.",
            "Two chain links: two rings linked in space.",
            d_mm=1,
            d=3,
            k=2,
            n=400,
        ),
        clases={
            "es": {
                "0": "el anillo horizontal, en el plano de x1 y x2",
                "1": "el anillo vertical, en el plano de x1 y x3",
            },
            "en": {
                "0": "the horizontal ring, in the plane of x1 and x2",
                "1": "the vertical ring, in the plane of x1 and x3",
            },
        },
        funciones=["fkdc.datasets:_dos_muestras", "fkdc.datasets:hacer_eslabones"],
        fuente="tesis, «Pionono, eslabones, hélices y hueveras»; σ y clases:"
        " hacer_eslabones",
    ),
    "helices": Bunch(
        nombre={"es": "dos hélices entrelazadas", "en": "two interleaved helices"},
        descripcion=_sintetico(
            "Dos hélices del mismo diámetro y «enroscadas» en la misma dirección"
            " alrededor del eje x3, una de ellas empezando a «media altura» entre dos"
            " brazos consecutivos de la otra.",
            'Two helices of the same diameter, "coiled" in the same direction around'
            ' the x3 axis, one of them starting "halfway up" between two consecutive'
            " turns of the other.",
            d_mm=1,
            d=3,
            k=2,
            n=400,
        ),
        clases={
            "es": {
                "0": "la hélice que a la altura x3 = 0 pasa por x1 positivo",
                "1": "la otra hélice, rotada media vuelta alrededor del eje x3",
            },
            "en": {
                "0": "the helix that passes through positive x1 at height x3 = 0",
                "1": "the other helix, rotated half a turn around the x3 axis",
            },
        },
        funciones=["fkdc.datasets:_dos_muestras", "fkdc.datasets:hacer_helices"],
        fuente="tesis, «Hélices»; eje, σ y clases: hacer_helices",
    ),
    "hueveras": Bunch(
        nombre={"es": "dos hueveras", "en": "two egg crates"},
        descripcion=_sintetico(
            "Dos clases con idénticas distribuciones pero signo opuesto en la"
            " coordenada vertical, x3 = ±sin(x1)·sin(x2), como los dos cartones de un"
            " maple de huevos intentando ocupar el mismo espacio.",
            "Two classes with identical distributions but opposite sign in the"
            " vertical coordinate, x3 = ±sin(x1)·sin(x2), like the two halves of an"
            " egg crate trying to occupy the same space.",
            d_mm=2,
            d=3,
            k=2,
            n=400,
        ),
        clases={
            "es": {
                "0": "la huevera x3 = sin(x1)·sin(x2)",
                "1": "la huevera x3 = −sin(x1)·sin(x2)",
            },
            "en": {
                "0": "the crate x3 = sin(x1)·sin(x2)",
                "1": "the crate x3 = −sin(x1)·sin(x2)",
            },
        },
        funciones=["fkdc.datasets:_dos_muestras", "fkdc.datasets:hacer_hueveras"],
        fuente="tesis, «Hueveras»; σ y signo por clase: hacer_hueveras",
    ),
    "pionono": Bunch(
        nombre={"es": "pionono (swiss roll)", "en": "swiss roll"},
        descripcion=_sintetico(
            "Un pionono (swiss roll), dataset clásico para evaluar algoritmos de"
            " clustering no lineales: una lámina rectangular enrollada en espiral"
            " alrededor del eje x2. Cada clase es una mancha gaussiana sobre la"
            " lámina.",
            "A swiss roll, a classic dataset for evaluating non-linear clustering"
            " algorithms: a rectangular sheet rolled up in a spiral around the x2"
            " axis. Each class is a Gaussian blob on the sheet.",
            d_mm=2,
            d=3,
            k=4,
            n=200,
        ),
        clases={
            "es": {
                "0": "la mancha en la vuelta interior del rollo, con x2 bajo",
                "1": "la mancha en la vuelta interior del rollo, con x2 alto",
                "2": "la mancha en la vuelta exterior del rollo, con x2 bajo",
                "3": "la mancha en la vuelta exterior del rollo, con x2 alto",
            },
            "en": {
                "0": "the blob on the inner turn of the roll, with low x2",
                "1": "the blob on the inner turn of the roll, with high x2",
                "2": "the blob on the outer turn of the roll, with low x2",
                "3": "the blob on the outer turn of the roll, with high x2",
            },
        },
        funciones=["fkdc.datasets:hacer_pionono"],
        fuente="tesis, «Pionono»; eje, σ y clases: hacer_pionono",
    ),
    "iris": Bunch(
        nombre={"es": "iris", "en": "iris"},
        descripcion={
            "es": "El dataset clásico de flores iris: K = 3 especies (setosa,"
            " versicolor y virginica) y d = 4 medidas en centímetros, el largo y el"
            " ancho del sépalo y del pétalo.",
            "en": "The classic iris flower dataset: K = 3 species (setosa, versicolor"
            " and virginica) and d = 4 measurements in centimetres, the length and"
            " width of the sepal and the petal.",
        },
        atributos={
            "es": (
                "largo_sepalo_cm ancho_sepalo_cm largo_petalo_cm ancho_petalo_cm"
            ).split(),
            "en": (
                "sepal_length_cm sepal_width_cm petal_length_cm petal_width_cm"
            ).split(),
        },
        opciones={"0": "setosa", "1": "versicolor", "2": "virginica"},
        descripcion_opcion={"es": "Iris {}", "en": "Iris {}"},
        fuente="tesis, «Datasets orgánicos» e «iris y vino»; especies y medidas:"
        " sklearn.datasets.load_iris",
    ),
    "pinguinos": Bunch(
        nombre={"es": "pingüinos de Palmer", "en": "Palmer penguins"},
        descripcion={
            "es": "El dataset de pingüinos de Palmer, casi linealmente separable:"
            " K = 3 especies (Adelie, Chinstrap y Gentoo) y d = 4 medidas, el largo"
            " y el alto del pico (mm), el largo de la aleta (mm) y la masa corporal"
            " (g).",
            "en": "The Palmer penguins dataset, almost linearly separable: K = 3"
            " species (Adelie, Chinstrap and Gentoo) and d = 4 measurements, bill"
            " length and depth (mm), flipper length (mm) and body mass (g).",
        },
        atributos={
            "es": ("largo_pico_mm alto_pico_mm largo_aleta_mm masa_corporal_g").split(),
            "en": (
                "bill_length_mm bill_depth_mm flipper_length_mm body_mass_g"
            ).split(),
        },
        etiquetas_reales=True,
        descripcion_opcion={"es": "pingüino {}", "en": "{} penguin"},
        fuente="tesis, «pinguinos: sensibilidad a la escala de los atributos»",
    ),
    "vino": Bunch(
        nombre={"es": "vino", "en": "wine"},
        descripcion={
            "es": "El dataset clásico de vinos: análisis químico de vinos de K = 3"
            " cultivares, con d = 13 variables que recorren cuatro órdenes de"
            " magnitud.",
            "en": "The classic wine dataset: chemical analysis of wines from K = 3"
            " cultivars, with d = 13 variables spanning four orders of magnitude.",
        },
        atributos={
            "es": (
                "alcohol acido_malico ceniza alcalinidad_ceniza magnesio"
                " fenoles_totales flavonoides fenoles_no_flavonoides"
                " proantocianinas intensidad_color matiz"
                " od280_od315_vinos_diluidos prolina"
            ).split(),
            "en": (
                "alcohol malic_acid ash alcalinity_of_ash magnesium"
                " total_phenols flavanoids nonflavanoid_phenols"
                " proanthocyanins color_intensity hue"
                " od280_od315_of_diluted_wines proline"
            ).split(),
        },
        fuente="tesis, «iris y vino»; cultivares: sklearn.datasets.load_wine",
    ),
    "digitos": Bunch(
        nombre={
            "es": "dígitos manuscritos de scikit-learn",
            "en": "scikit-learn handwritten digits",
        },
        descripcion={
            "es": "Imágenes de 8×8 píxeles de dígitos manuscritos, del 0 al 9"
            " (K = 10), con d = 64 píxeles de valores de 0 a 16; pixel_i_j es el de"
            " la fila i y la columna j.",
            "en": "8×8-pixel images of handwritten digits, 0 to 9 (K = 10), with"
            " d = 64 pixels taking values from 0 to 16; pixel_i_j is the one in row"
            " i and column j.",
        },
        atributos={
            idioma: [f"pixel_{i}_{j}" for i in range(8) for j in range(8)]
            for idioma in ("es", "en")
        },
        etiquetas_reales=True,
        descripcion_opcion={"es": "el dígito {}", "en": "the digit {}"},
        fuente="tesis, «digitos»",
    ),
    "mnist": Bunch(
        nombre={"es": "MNIST", "en": "MNIST"},
        descripcion={
            "es": "Imágenes de dígitos manuscritos MNIST (28×28 = 784 píxeles, K = 10"
            " dígitos), reducidas por PCA a d = 96 componentes principales que"
            " conservan al menos el 90 % de la variación; pc1 es la de mayor"
            " varianza.",
            "en": "MNIST handwritten digit images (28×28 = 784 pixels, K = 10"
            " digits), reduced by PCA to d = 96 principal components that keep at"
            " least 90% of the variation; pc1 has the largest variance.",
        },
        atributos={idioma: [f"pc{i}" for i in range(1, 97)] for idioma in ("es", "en")},
        etiquetas_reales=True,
        descripcion_opcion={"es": "el dígito {}", "en": "the digit {}"},
        fuente="tesis, «mnist»",
    ),
}


def partes_variante(variante: str) -> tuple[str, str | None, bool]:
    """(familia, sufijo de ruido, estandarizada) de una variante de dataset."""
    estandarizada = variante.endswith("_std")
    base = variante.removesuffix("_std")
    familia, _, sufijo = base.rpartition("_")
    if familia in fichas and sufijo in {"lo", "hi", "0", "12"}:
        return familia, sufijo, estandarizada
    return base, None, estandarizada


def _numero(valor: float, decimales: int) -> str:
    """Número con `decimales` fijos, sin ceros a la derecha (5.10 -> 5.1)."""
    texto = f"{valor:.{decimales}f}"
    if "." in texto:
        texto = texto.rstrip("0").rstrip(".")
    return "0" if texto == "-0" else texto


def _sin_docstring(fuente: str) -> str:
    """Fuente de una función sin decoradores ni docstring (conserva comentarios)."""
    funcion = ast.parse(fuente).body[0]
    quitar = set()
    if funcion.decorator_list:
        quitar |= set(range(funcion.decorator_list[0].lineno - 1, funcion.lineno - 1))
    primera = funcion.body[0]
    if isinstance(primera, ast.Expr) and isinstance(primera.value, ast.Constant):
        quitar |= set(range(primera.lineno - 1, primera.end_lineno))
    lineas = fuente.splitlines()
    return "\n".join(linea for i, linea in enumerate(lineas) if i not in quitar)


def _argumentos(parametros: dict) -> str:
    return ", ".join(f"{clave}={valor!r}" for clave, valor in parametros.items())


def codigo_generador(familia: str, parametros: dict) -> str:
    """Fuente de las funciones generadoras de `familia` y la llamada que se usó."""
    funciones = fichas[familia].get("funciones")
    if not funciones or not parametros:
        raise ValueError(f"No hay código generador para {familia}")
    partes = []
    for ruta in [*funciones, "fkdc.datasets:agregar_dims_ruido"]:
        if ruta.endswith("agregar_dims_ruido") and not parametros.get("ruido"):
            continue
        nombre_modulo, nombre_funcion = ruta.split(":")
        funcion = getattr(import_module(nombre_modulo), nombre_funcion)
        partes.append(_sin_docstring(inspect.getsource(funcion)))
    argumentos = {k: v for k, v in parametros.items() if k not in {"factory", "ruido"}}
    llamada = f"X, y = {parametros['factory']}({_argumentos(argumentos)})"
    if ruido := parametros.get("ruido"):
        llamada += f"\nX = agregar_dims_ruido(X, {_argumentos(dict(ruido))})"
    return "\n\n".join([*partes, llamada])


def orden_estratificado(y, semilla: int) -> np.ndarray:
    """Permutación de `y` cuyos prefijos respetan (aprox.) las proporciones de clase."""
    rng = np.random.default_rng(semilla)
    y = np.asarray(y)
    posiciones = np.empty(len(y))
    for etiqueta in np.unique(y):
        indices = rng.permutation(np.flatnonzero(y == etiqueta))
        posiciones[indices] = (np.arange(len(indices)) + rng.uniform()) / len(indices)
    return np.argsort(posiciones, kind="stable")


class Redactor:
    """Redacta estado y pregunta de una tarea según el régimen y el idioma.

    Es lo único que comparten Laya y la exportación para Jev, de modo que ambos
    modelos leen exactamente el mismo texto.
    """

    def __init__(
        self,
        variante: str,
        regimen: str,
        idioma: str = "es",
        parametros: dict | None = None,
        decimales: int = 2,
    ):
        if regimen not in regimenes:
            raise ValueError(f"Régimen desconocido: {regimen}")
        self.variante, self.regimen, self.idioma = variante, regimen, idioma
        self.parametros = parametros or {}
        self.decimales = decimales
        self.familia, self.sufijo_ruido, self.estandarizada = partes_variante(variante)
        self.ficha = fichas[self.familia]
        self.textos = textos[idioma]
        self.contexto = self._contexto()

    def atributos(self, p: int) -> list[str]:
        """Nombres de los `p` atributos: los del dataset o x1..xp."""
        nombres = self.ficha.get("atributos", {}).get(self.idioma)
        return (
            nombres
            if nombres and len(nombres) == p
            else [f"x{i + 1}" for i in range(p)]
        )

    def clave(self, etiqueta: str) -> str:
        """Nombre de la opción que representa a la clase `etiqueta`.

        El nombre real de la clase si el dataset lo tiene (especies, dígitos); si no,
        una letra (0 -> A, 1 -> B...). Las etiquetas numéricas no son neutras: se
        confunden con los valores de los atributos (en `lunas`, con opciones "0" y
        "1", Laya acierta por coincidencia léxica bastante más que al azar).
        """
        etiqueta = str(etiqueta)
        if etiqueta in self.ficha.get("opciones", {}):
            return self.ficha.opciones[etiqueta]
        if self.ficha.get("etiquetas_reales"):
            return etiqueta
        return "ABCDEFGHIJ"[int(etiqueta)]

    def opciones(self, etiquetas) -> dict[str, str]:
        """Opción -> descripción, en el orden de `etiquetas`."""
        por_clase = {}
        if self.regimen in regimenes_con_descripcion:
            por_clase = self.ficha.get("clases", {}).get(self.idioma, {})
        elif self.regimen == "codigo":  # la etiqueta tal como sale del código
            por_clase = {str(e): self.textos.clase_codigo.format(e) for e in etiquetas}
        formato = self.ficha.get("descripcion_opcion", {}).get(
            self.idioma, self.textos.clase
        )
        return {
            self.clave(e): por_clase.get(str(e), formato.format(self.clave(e)))
            for e in etiquetas
        }

    def pregunta(self, etiquetas) -> dict:
        """Pregunta de opción múltiple con una opción por clase."""
        return {
            "type": "choice",
            "instructions": self.textos.pregunta,
            "criteria": self.opciones(etiquetas),
        }

    def fila(self, x) -> str:
        """Una observación como `nombre=valor; ...`."""
        nombres = self.atributos(len(x))
        valores = (_numero(v, self.decimales) for v in x)
        return "; ".join(f"{n}={v}" for n, v in zip(nombres, valores, strict=True))

    def _contexto(self) -> str:
        """Lo que precede a ejemplos y observación: nombre, descripción, código."""
        partes = []
        if self.regimen in regimenes_con_nombre:
            partes.append(self.textos.dataset.format(self.ficha.nombre[self.idioma]))
        if self.regimen in regimenes_con_descripcion:
            descripcion = self.ficha.descripcion[self.idioma]
            sigma = self.parametros.get("noise")
            if "{sigma}" in descripcion and sigma is None:
                raise ValueError(f"{self.variante}: la descripción requiere σ")
            descripcion = descripcion.replace("{sigma}", str(sigma))
            if self.sufijo_ruido == "12":
                descripcion += self.textos.ruido_12
            partes.append(descripcion)
        if self.regimen == "codigo":
            codigo = codigo_generador(self.familia, self.parametros)
            partes.append(f"{self.textos.codigo}\n{codigo}")
        if self.estandarizada and len(partes):
            partes.append(self.textos.estandarizado)
        return "\n".join(partes)

    def ejemplos(self, X, y, semilla: int, contar_tokens, presupuesto: int):
        """Ejemplos etiquetados que caben en `presupuesto` tokens de estado.

        Toma un prefijo del orden estratificado de (`X`, `y`); el costo fijo (contexto
        y la observación más larga del pliego) se descuenta antes de sumar ejemplos.
        Devuelve el bloque de texto y la cantidad de ejemplos `k`.
        """
        filas = [self.fila(x) for x in X]
        fila_larga = max(filas, key=lambda f: contar_tokens(f))
        encabezado = self.textos.ejemplos
        usados = contar_tokens(self.estado_texto(fila_larga, encabezado)) + 8
        lineas = []
        for i in orden_estratificado(y, semilla):
            linea = f"{filas[i]} -> {self.clave(y[i])}"
            costo = contar_tokens(linea) + 1
            if usados + costo > presupuesto:
                break
            lineas.append(linea)
            usados += costo
        return "\n".join([encabezado, *lineas]), len(lineas)

    def estado_texto(self, fila: str, ejemplos: str = "") -> str:
        """Estado: contexto, ejemplos y, al final, la observación a clasificar."""
        partes = [self.contexto, ejemplos, self.textos.observacion.format(fila)]
        return "\n".join(p for p in partes if p)

    def estado(self, x, ejemplos: str = "") -> str:
        """Estado para la observación `x`."""
        return self.estado_texto(self.fila(x), ejemplos)


familias = {
    "laya": "laya",
    "gliner": "gliner",
    "gliner_ajustado": "glinerft",
}


def nombre_clf(
    regimen: str,
    modelo: str = "multilingual",
    idioma: str = "es",
    familia: str = "laya",
) -> str:
    """Nombre del clasificador en los infos, p. ej. `laya_desc` o `gliner_en_zs`."""
    variantes = {("multilingual", "es"): "", ("english", "en"): "_en",
                 ("multilingual", "en"): "_mlen"}  # fmt: skip
    if (modelo, idioma) not in variantes:
        raise ValueError(f"Combinación no prevista: {modelo}, {idioma}")
    return f"{familias[familia]}{variantes[modelo, idioma]}_{regimenes[regimen]}"


@cache
def _clase_agente():
    """Subclase de `laya.agent.Agent` que conserva las probabilidades sin redondear.

    Laya redondea las probabilidades a 4 decimales, y una probabilidad redondeada a
    0 vuelve infinita (recortada) la log-verosimilitud. Esta subclase repite el
    cálculo de `Agent._decode_answers` (logits / temperatura, softmax) en doble
    precisión y lo guarda en `probabilidades_exactas`.
    """
    os.environ.setdefault("USE_TF", "0")
    from laya.agent import Agent, _option_logits
    from laya.common import QTYPES, temp_bucket, unpermute_probs

    class AgenteExacto(Agent):
        def _decode_answers(self, logits, act, items, ids, internal, offset, lang=None):
            respuestas = super()._decode_answers(
                logits, act, items, ids, internal, offset, lang=lang
            )
            filas = _option_logits(logits, items, offset)
            for j, qid in enumerate(ids):
                pregunta, k = internal[qid], len(items[j]["markers"])
                tipo = QTYPES[pregunta["t"]]
                t = self.temperature_by_options.get(
                    temp_bucket(tipo, k), self.temperature[tipo]
                )
                z = np.asarray(filas[j], dtype=np.float64) / t
                p = np.exp(z - z.max())
                p = unpermute_probs(p / p.sum(), pregunta.get("option_order"))
                respuestas[qid]["probabilidades_exactas"] = p
            return respuestas

    return AgenteExacto


_agentes = {}


def dispositivo_por_omision() -> str:
    """`FKDC_DISPOSITIVO` si está definida; si no, MPS si está disponible, o CPU."""
    import torch

    if dispositivo := os.environ.get("FKDC_DISPOSITIVO"):
        return dispositivo
    return "mps" if torch.backends.mps.is_available() else "cpu"


def cargar_agente(modelo: str = "multilingual", dispositivo: str | None = None):
    """Agente de Laya (uno por proceso y combinación), con revisión fija."""
    dispositivo = dispositivo or dispositivo_por_omision()
    if (modelo, dispositivo) not in _agentes:
        logger.info("Cargando Laya %s en %s", modelo, dispositivo)
        _agentes[modelo, dispositivo] = _clase_agente()(
            repo_laya,
            subfolder=subcarpetas_laya[modelo],
            revision=revision_laya,
            device=dispositivo,
        )
    return _agentes[modelo, dispositivo]


@cache
def cargar_tokenizador(modelo: str = "multilingual"):
    """Tokenizador de un checkpoint de Laya, sin cargar el modelo."""
    for (nombre, _), agente in _agentes.items():
        if nombre == modelo:
            return agente.tok
    os.environ.setdefault("USE_TF", "0")
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config
    from transformers import AutoTokenizer

    subcarpeta = subcarpetas_laya[modelo]
    prefijo = f"{subcarpeta}/" if subcarpeta else ""
    raiz = snapshot_download(
        repo_laya,
        revision=revision_laya,
        allow_patterns=[f"{prefijo}tokenizer/*", f"{prefijo}rl_agent_config.json"],
    )
    dir_modelo = os.path.join(raiz, subcarpeta) if subcarpeta else raiz
    _fix_tokenizer_config(dir_modelo)
    return AutoTokenizer.from_pretrained(os.path.join(dir_modelo, "tokenizer"))


@cache
def configuracion_laya(modelo: str = "multilingual") -> dict:
    """`rl_agent_config.json` del checkpoint (max_len, head_max_len, temperaturas)."""
    import json

    from huggingface_hub import hf_hub_download

    subcarpeta = subcarpetas_laya[modelo]
    archivo = (
        f"{subcarpeta}/rl_agent_config.json" if subcarpeta else "rl_agent_config.json"
    )
    ruta = hf_hub_download(repo_laya, archivo, revision=revision_laya)
    return json.load(open(ruta))


def contador_tokens(modelo: str = "multilingual"):
    """Función que cuenta los tokens de un texto con el tokenizador de `modelo`."""
    tokenizador = cargar_tokenizador(modelo)

    def contar(texto: str) -> int:
        return len(tokenizador(texto, add_special_tokens=False)["input_ids"])

    return contar


def presupuesto_estado(max_len: int, modelo: str = "multilingual") -> int:
    """Tokens de estado que deja `max_len` tras la cabeza (pregunta y opciones)."""
    return max_len - configuracion_laya(modelo)["head_max_len"] - 8


def elegir_max_len(n_tokens_estado: int, modelo: str = "multilingual") -> int:
    """Menor potencia de 2 (no menor que la del checkpoint) que no trunca el estado."""
    cfg = configuracion_laya(modelo)
    necesarios = n_tokens_estado + cfg["head_max_len"] + 8
    potencia = 2 ** math.ceil(math.log2(necesarios))
    return min(max(cfg["max_len"], potencia), max_len_tope)


class _ClasificadorTexto(ClassifierMixin, BaseEstimator):
    """Base de los clasificadores de texto: clases, ejemplos y memoria.

    `fit` guarda las clases y, en los regímenes con ejemplos, arma el bloque de
    ejemplos etiquetados (orden estratificado con `semilla`, tantos como quepan en
    `presupuesto_estado(max_len_ic)` tokens de Laya multilingüe, para que todos los
    modelos lean los mismos ejemplos). `predict_proba` devuelve columnas en el orden
    de `classes_`, memorizadas para la última `X` de modo que `predict` (el argmax)
    no repita la inferencia. Tras `predict_proba` quedan `t_inferencia_`,
    `max_len_`, `tokens_estado_` (mediana) y `truncadas_` (filas con estado
    truncado, que deberían ser 0).
    """

    def fit(self, X, y):
        """Guarda las clases y, si el régimen los usa, los ejemplos etiquetados."""
        X, y = np.asarray(X), np.asarray(y).astype(str)
        self.classes_ = np.unique(y)
        self.redactor_ = Redactor(
            self.variante, self.regimen, self.idioma, self.parametros, self.decimales
        )
        self.ejemplos_, self.k_ejemplos_ = "", 0
        if self.regimen in regimenes_con_ejemplos:
            self.ejemplos_, self.k_ejemplos_ = self.redactor_.ejemplos(
                X,
                y,
                self.semilla,
                contador_tokens("multilingual"),
                presupuesto_estado(self.max_len_ic, "multilingual"),
            )
        self._memoria = None
        return self

    def estados(self, X) -> list[str]:
        """Estados (textos) de las filas de `X`."""
        return [self.redactor_.estado(x, self.ejemplos_) for x in np.asarray(X)]

    def _inferir(self, estados: list[str]) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X):
        """Probabilidades por clase, en el orden de `classes_`."""
        huella = hashlib.sha1(np.ascontiguousarray(X).tobytes()).hexdigest()
        if self._memoria is not None and self._memoria[0] == huella:
            return self._memoria[1]
        estados = self.estados(X)
        contar = contador_tokens("multilingual")
        self.tokens_estado_ = float(np.median([contar(e) for e in estados]))
        t0 = time()
        probas = self._inferir(estados)
        self.t_inferencia_ = time() - t0
        if self.truncadas_:
            logger.warning("%d estados truncados", self.truncadas_)
        self._memoria = (huella, probas)
        return probas

    def predict(self, X):
        """Clase de máxima probabilidad."""
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class LayaClassifier(_ClasificadorTexto):
    """Laya como clasificador de scikit-learn, sobre filas serializadas como texto.

    `max_len` es el contexto de cada llamada; por omisión, la menor potencia de 2 que
    no trunca ningún estado.
    """

    def __init__(
        self,
        regimen: str = "zero-shot",
        variante: str = "lunas_lo",
        parametros: dict | None = None,
        idioma: str = "es",
        modelo: str = "multilingual",
        max_len_ic: int = 2048,
        max_len: int | None = None,
        decimales: int = 2,
        semilla: int = 0,
        dispositivo: str | None = None,
    ):
        self.regimen = regimen
        self.variante = variante
        self.parametros = parametros
        self.idioma = idioma
        self.modelo = modelo
        self.max_len_ic = max_len_ic
        self.max_len = max_len
        self.decimales = decimales
        self.semilla = semilla
        self.dispositivo = dispositivo

    def _inferir(self, estados: list[str]) -> np.ndarray:
        contar = contador_tokens(self.modelo)
        n_tokens = max(contar(e) for e in estados)
        self.max_len_ = self.max_len or elegir_max_len(n_tokens, self.modelo)
        agente = cargar_agente(self.modelo, self.dispositivo)
        resultados = agente.predict_batch(
            estados,
            {"clase": self.redactor_.pregunta(self.classes_)},
            batch_size=max(1, 32768 // self.max_len_),
            max_len=self.max_len_,
        )
        self.truncadas_ = int(sum(r["usage"]["truncated"] for r in resultados))
        return np.array(
            [r["answers"]["clase"]["probabilidades_exactas"] for r in resultados]
        )


# GLiNER2.5-Decide (Fastino): checkpoints y revisiones fijas
repos_gliner = {
    "multilingual": (
        "fastino/GLiNER2.5-multi-Decide",
        "a35a0cd3b7a0f00f2effc576f454cd48fa98aa5f",
    ),
    "english": ("fastino/GLiNER2.5-Decide", "5a7adf72a23b4d311abae6ce050d7f0012bb3416"),
}


@cache
def ruta_gliner(modelo: str = "multilingual") -> str:
    """Directorio local del checkpoint de GLiNER2.5-Decide (lo descarga si falta)."""
    from huggingface_hub import snapshot_download

    repo, revision = repos_gliner[modelo]
    return snapshot_download(
        repo, revision=revision, allow_patterns=["*.json", "*.safetensors"]
    )


_glineres = {}


def cargar_gliner(modelo: str = "multilingual", dispositivo: str | None = None):
    """Clasificador de GLiNER2 (uno por proceso y combinación)."""
    dispositivo = dispositivo or dispositivo_por_omision()
    if (modelo, dispositivo) not in _glineres:
        from gliner2.classification.engine import Classifier

        logger.info("Cargando GLiNER2.5-Decide %s en %s", modelo, dispositivo)
        clasificador = Classifier.from_pretrained(ruta_gliner(modelo))
        _glineres[modelo, dispositivo] = clasificador.to(dispositivo).eval()
    return _glineres[modelo, dispositivo]


def _criterios_gliner(pregunta: dict) -> dict[str, str]:
    """Opciones de la pregunta como etiquetas de GLiNER2.

    GLiNER2 no admite paréntesis en etiquetas ni descripciones (son marcadores de su
    prompt): "x (y)" pasa a "x, y".
    """
    return {
        opcion: re.sub(r"\s*\(([^)]*)\)", r", \1", descripcion)
        for opcion, descripcion in pregunta["criteria"].items()
    }


def _probas_gliner(clasificador, estados, pregunta: dict, lote: int = 32):
    """Probabilidad de cada opción (softmax de los logits por etiqueta).

    La pregunta se traduce a una tarea de etiqueta única de GLiNER2: las opciones
    con sus descripciones son las etiquetas y la consigna es la instrucción.
    """
    from gliner2.classification.engine import ClassificationConfig
    from gliner2.classification.schema import ClassificationSchema

    criterios = _criterios_gliner(pregunta)
    esquema = ClassificationSchema().single(
        "clase", criterios, instruction=pregunta["instructions"]
    )
    puntajes = clasificador.batch_score(
        estados, esquema, config=ClassificationConfig(batch_size=lote)
    )
    return np.array([[p.probability("clase", c) for c in criterios] for p in puntajes])


def _palabras_de_mas(clasificador, estados) -> int:
    """Estados con más palabras que el `max_len` de GLiNER2 (las trunca)."""
    tope = getattr(getattr(clasificador.model, "config", None), "max_len", None)
    if not tope:
        return 0
    return sum(len(re.findall(r"\w+|[^\w\s]", e)) > tope for e in estados)


class GlinerClassifier(_ClasificadorTexto):
    """GLiNER2.5-Decide sin ajustar, con el mismo texto que lee Laya."""

    def __init__(
        self,
        regimen: str = "zero-shot",
        variante: str = "lunas_lo",
        parametros: dict | None = None,
        idioma: str = "es",
        modelo: str = "multilingual",
        max_len_ic: int = 2048,
        decimales: int = 2,
        semilla: int = 0,
        dispositivo: str | None = None,
        lote: int = 32,
    ):
        self.regimen = regimen
        self.variante = variante
        self.parametros = parametros
        self.idioma = idioma
        self.modelo = modelo
        self.max_len_ic = max_len_ic
        self.decimales = decimales
        self.semilla = semilla
        self.dispositivo = dispositivo
        self.lote = lote

    def _clasificador(self):
        return cargar_gliner(self.modelo, self.dispositivo)

    def _inferir(self, estados: list[str]) -> np.ndarray:
        clasificador = self._clasificador()
        self.max_len_ = None
        self.truncadas_ = _palabras_de_mas(clasificador, estados)
        pregunta = self.redactor_.pregunta(self.classes_)
        return _probas_gliner(clasificador, estados, pregunta, self.lote)


@cache
def _clase_entrenador():
    """`ExtractorTrainer` de GLiNER2 que entrena en `dispositivo` (también MPS) y no
    escribe checkpoints: el modelo ajustado se usa en memoria."""
    from gliner2.training.trainer import ExtractorTrainer

    class EntrenadorGliner(ExtractorTrainer):
        dispositivo = "cpu"

        def _setup_device(self):
            import torch

            self.device = torch.device(self.dispositivo)
            self.is_distributed = False
            self.model.to(self.device)
            self.model.float()

        def _save_checkpoint(self, name):
            pass

    return EntrenadorGliner


class GlinerAjustadoClassifier(GlinerClassifier):
    """GLiNER2.5-Decide ajustado (todos los pesos) sobre el pliego de entrenamiento.

    Cada observación de entrenamiento, serializada según `regimen` (por omisión
    solo la fila), es un ejemplo de clasificación con su clase verdadera. El
    ajuste parte siempre del checkpoint publicado; `t_ajuste_` mide su duración.
    """

    def __init__(
        self,
        regimen: str = "zero-shot",
        variante: str = "lunas_lo",
        parametros: dict | None = None,
        idioma: str = "es",
        modelo: str = "multilingual",
        max_len_ic: int = 2048,
        decimales: int = 2,
        semilla: int = 0,
        dispositivo: str | None = None,
        lote: int = 32,
        epocas: int = 10,
        lote_ajuste: int = 16,
        lr_codificador: float = 1e-5,
        lr_tarea: float = 5e-4,
    ):
        super().__init__(
            regimen, variante, parametros, idioma, modelo, max_len_ic, decimales,
            semilla, dispositivo, lote,
        )  # fmt: skip
        self.epocas = epocas
        self.lote_ajuste = lote_ajuste
        self.lr_codificador = lr_codificador
        self.lr_tarea = lr_tarea

    def fit(self, X, y):
        """Ajusta una copia del checkpoint sobre (`X`, `y`) serializados."""
        import tempfile

        os.environ.setdefault("TQDM_DISABLE", "1")
        from gliner2 import AutoExtractor
        from gliner2.classification.engine import Classifier
        from gliner2.training.data import Classification, InputExample
        from gliner2.training.trainer import TrainingConfig

        super().fit(X, y)
        X, y = np.asarray(X), np.asarray(y).astype(str)
        pregunta = self.redactor_.pregunta(self.classes_)
        criterios = _criterios_gliner(pregunta)
        ejemplos = [
            InputExample(
                text=self.redactor_.estado(x, self.ejemplos_),
                classifications=[
                    Classification(
                        task="clase",
                        labels=list(criterios),
                        true_label=self.redactor_.clave(etiqueta),
                        label_descriptions=dict(criterios),
                        prompt=pregunta["instructions"],
                    )
                ],
            )
            for x, etiqueta in zip(X, y, strict=True)
        ]
        dispositivo = self.dispositivo or dispositivo_por_omision()
        entrenador = type(
            "EntrenadorGliner", (_clase_entrenador(),), {"dispositivo": dispositivo}
        )
        t0 = time()
        modelo = AutoExtractor.from_pretrained(ruta_gliner(self.modelo))
        with tempfile.TemporaryDirectory() as directorio:
            configuracion = TrainingConfig(
                output_dir=directorio,
                num_epochs=self.epocas,
                batch_size=self.lote_ajuste,
                encoder_lr=self.lr_codificador,
                task_lr=self.lr_tarea,
                eval_strategy="no",
                save_best=False,
                num_workers=0,
                pin_memory=False,
                fp16=False,
                bf16=False,
                seed=self.semilla,
                logging_steps=10**9,
                logging_first_step=False,
            )
            entrenador(modelo, configuracion, train_data=ejemplos).train()
        self.t_ajuste_ = time() - t0
        self.clasificador_ = Classifier(modelo).to(dispositivo).eval()
        return self

    def _clasificador(self):
        return self.clasificador_


class _Desempaquetador(pickle.Unpickler):
    """Los datasets se guardaron corriendo `fkdc/datasets.py` como script."""

    def find_class(self, module, name):
        if module == "__main__" and name == "Dataset":
            return Dataset
        return super().find_class(module, name)


def cargar_dataset(ruta: Path) -> Dataset:
    """Dataset de `sandbox/v5/datasets`, sin depender de `__main__.Dataset`."""
    with open(ruta, "rb") as archivo:
        return _Desempaquetador(archivo).load()


def parametros_dataset(dataset: Dataset) -> dict | None:
    """Parámetros de la fábrica del dataset, como diccionarios simples."""
    parametros = getattr(dataset, "params", None)
    if parametros is None:
        return None
    return {
        clave: dict(valor) if isinstance(valor, dict) else valor
        for clave, valor in parametros.items()
    }


def tareas_tesis(patrones=("*",)) -> list[Bunch]:
    """Las 600 tareas de la tesis (las de `lr` en `sandbox/v5/infos`).

    El identificador de una tarea es `{variante}-{semilla_dataset}-{semilla_tarea}`,
    p. ej. `lunas_lo-1075-1312` o `iris_std-None-1182`; `patrones` son globs sobre él.
    """
    tareas = []
    for ruta in sorted(config.dir_ejecucion.glob("*-lr-*-neg_log_loss.pkl")):
        variante, semilla_ds, _, semilla_tarea, _ = ruta.stem.split("-")
        identificador = f"{variante}-{semilla_ds}-{semilla_tarea}"
        if not any(fnmatch(identificador, patron) for patron in patrones):
            continue
        base = variante.removesuffix("_std")
        nombre_ds = base if semilla_ds == "None" else f"{base}-{semilla_ds}"
        tareas.append(
            Bunch(
                id=identificador,
                variante=variante,
                semilla_ds=semilla_ds,
                semilla_tarea=int(semilla_tarea),
                ruta_dataset=dir_datasets / f"{nombre_ds}.pkl",
                estandarizar=variante.endswith("_std"),
            )
        )
    return tareas


def ruta_info(tarea: Bunch, clf: str, directorio: Path = dir_infos) -> Path:
    """Ruta del info, con el mismo esquema de nombres que los de la tesis."""
    return directorio / (
        f"{tarea.variante}-{tarea.semilla_ds}-{clf}-{tarea.semilla_tarea}"
        "-neg_log_loss.pkl"
    )


def aplica(regimen: str, variante: str) -> bool:
    """Si el régimen tiene sentido para la variante (`codigo`: solo sintéticos)."""
    familia, _, _ = partes_variante(variante)
    return regimen != "codigo" or bool(fichas[familia].get("funciones"))


class _AjusteDirecto:
    """Reemplazo de `GridSearchCV` en `Tarea`: ajusta sin validación cruzada."""

    def __init__(self, estimador, espacio=None, **_):
        self.estimador = estimador

    def fit(self, X, y):
        self.best_estimator_ = clone(self.estimador).fit(X, y)
        return self


def evaluar_tarea(tarea: Bunch, clf: str, estimador) -> Bunch:
    """Corre `Tarea` (partición, base y métricas de la tesis) con un estimador fijo."""
    from fkdc.tarea import Tarea

    t = Tarea(
        cargar_dataset(tarea.ruta_dataset),
        {clf: (estimador, {})},
        busqueda_factory=_AjusteDirecto,
        semilla=tarea.semilla_tarea,
        estandarizar=tarea.estandarizar,
    )
    t.evaluar()
    info = t.info[clf]
    if "r2" not in info:  # `Tarea.evaluar` traga las excepciones
        raise RuntimeError(f"Evaluación incompleta en {tarea.id}")
    ajustado = t.clasificadores[clf]
    if isinstance(ajustado, Pipeline):
        ajustado = ajustado[-1]
    info.busqueda = None
    info.n_eval = len(t.y_eval)
    info.t_probas = info.get("t_probas", ajustado.t_inferencia_)
    info.t_inferencia = ajustado.t_inferencia_
    info.s_por_fila = ajustado.t_inferencia_ / info.n_eval
    for campo in ["regimen", "modelo", "idioma"]:
        info[campo] = getattr(ajustado, campo)
    for campo in ["k_ejemplos_", "max_len_", "tokens_estado_", "truncadas_"]:
        info[campo.rstrip("_")] = getattr(ajustado, campo)
    info.familia = type(ajustado).__name__
    if hasattr(ajustado, "t_ajuste_"):
        info.t_ajuste = ajustado.t_ajuste_
        info.epocas = ajustado.epocas
    return t.info


app = typer.Typer(add_completion=False)


@app.command()
def correr(
    regimen: str,
    patron: Annotated[list[str] | None, typer.Option(help="Glob de tareas")] = None,
    familia: Annotated[
        str, typer.Option(help="laya, gliner o gliner_ajustado")
    ] = "laya",
    idioma: str = "es",
    modelo: str = "multilingual",
    max_len_ic: int = 2048,
    epocas: Annotated[int, typer.Option(help="Solo gliner_ajustado")] = 10,
    dispositivo: str | None = None,
    sufijo: Annotated[str, typer.Option(help="Agregado al nombre del clf")] = "",
    forzar: bool = False,
):
    """Corre un clasificador de texto en las tareas de `patron` que no tienen info."""
    clf = nombre_clf(regimen, modelo, idioma, familia) + sufijo
    clases = {
        "laya": LayaClassifier,
        "gliner": GlinerClassifier,
        "gliner_ajustado": GlinerAjustadoClassifier,
    }
    dir_infos.mkdir(parents=True, exist_ok=True)
    tareas = [t for t in tareas_tesis(patron or ["*"]) if aplica(regimen, t.variante)]
    pendientes = [t for t in tareas if forzar or not ruta_info(t, clf).exists()]
    logger.info("%s: %d tareas, %d pendientes", clf, len(tareas), len(pendientes))
    for i, tarea in enumerate(pendientes, start=1):
        dataset = cargar_dataset(tarea.ruta_dataset)
        argumentos = {
            "regimen": regimen,
            "variante": tarea.variante,
            "parametros": parametros_dataset(dataset),
            "idioma": idioma,
            "modelo": modelo,
            "max_len_ic": max_len_ic,
            "semilla": tarea.semilla_tarea,
            "dispositivo": dispositivo,
        }
        if familia == "gliner_ajustado":
            argumentos["epocas"] = epocas
        info = evaluar_tarea(tarea, clf, clases[familia](**argumentos))
        logger.info(
            "[%d/%d] %s %s: acc=%.3f r2=%.3f %.4f s/fila k=%d (ajuste %.0f s)",
            i,
            len(pendientes),
            clf,
            tarea.id,
            info[clf].accuracy,
            info[clf].r2,
            info[clf].s_por_fila,
            info[clf].k_ejemplos,
            info[clf].t_entrenar,
        )
        with open(ruta_info(tarea, clf), "wb") as archivo:
            pickle.dump(info, archivo)


def cargar_resultados(clfs, patrones=("*",), directorio: Path = dir_infos):
    """Una fila por info de `clfs` en `directorio`."""
    campos = ["accuracy", "r2", "s_por_fila", "k_ejemplos", "max_len", "truncadas",
              "t_entrenar"]  # fmt: skip
    filas = []
    for tarea in tareas_tesis(patrones):
        for clf in clfs:
            ruta = ruta_info(tarea, clf, directorio)
            if not ruta.exists():
                continue
            info = pickle.load(open(ruta, "rb"))
            fila = {campo: info[clf].get(campo, np.nan) for campo in campos}
            filas.append({"tarea": tarea.id, "variante": tarea.variante, "clf": clf,
                          **fila})  # fmt: skip
    return pd.DataFrame(filas)


@app.command()
def resumir(
    clf: Annotated[list[str], typer.Option(help="Clasificadores")],
    patron: Annotated[list[str] | None, typer.Option(help="Glob de tareas")] = None,
    referencia: Annotated[
        list[str] | None, typer.Option(help="Clasificadores de la tesis")
    ] = None,
    ruta_salida: Path | None = None,
):
    """Media por variante y clasificador de accuracy, R² y segundos por fila."""
    import warnings

    from sklearn.exceptions import InconsistentVersionWarning

    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    resultados = cargar_resultados(clf, patron or ["*"])
    if referencia:
        tareas = set(resultados.tarea)
        tesis = cargar_resultados(referencia, patron or ["*"], config.dir_ejecucion)
        resultados = pd.concat([resultados, tesis[tesis.tarea.isin(tareas)]])
    resumen = resultados.groupby(["variante", "clf"], sort=False).agg(
        n=("r2", "size"),
        accuracy=("accuracy", "mean"),
        r2=("r2", "mean"),
        s_por_fila=("s_por_fila", "mean"),
        k=("k_ejemplos", "mean"),
        truncadas=("truncadas", "sum"),
    )
    if ruta_salida:
        resumen.to_csv(ruta_salida, float_format="%.4g")
    tabla = resumen.reset_index().pivot(
        index="variante", columns="clf", values=["accuracy", "r2", "s_por_fila"]
    )
    print(tabla.to_string(float_format=lambda v: f"{v:.3f}"))


@app.command()
def estimar(
    clf: str,
    patron: Annotated[list[str] | None, typer.Option(help="Glob de tareas")] = None,
):
    """Horas que llevaría el barrido completo de `clf`, extrapolando el piloto."""
    abreviaturas = sorted(regimenes.items(), key=lambda par: -len(par[1]))
    regimen = next(r for r, a in abreviaturas if clf.endswith(f"_{a}"))
    piloto = cargar_resultados([clf])
    piloto["familia"] = [partes_variante(v)[0] for v in piloto.variante]
    s_por_fila = piloto.groupby("familia").s_por_fila.median()
    filas = []
    for tarea in tareas_tesis(patron or ["*"]):
        if not aplica(regimen, tarea.variante):
            continue
        familia = partes_variante(tarea.variante)[0]
        n_eval = math.ceil(cargar_dataset(tarea.ruta_dataset).n / 2)
        segundos = s_por_fila.get(familia, s_por_fila.max()) * n_eval
        filas.append({"familia": familia, "n_eval": n_eval, "s": segundos})
    costo = (
        pd.DataFrame(filas)
        .groupby("familia")
        .agg(tareas=("n_eval", "size"), filas=("n_eval", "sum"), horas=("s", "sum"))
    )
    costo["horas"] /= 3600
    print(costo.to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"Total: {costo.horas.sum():.2f} h ({costo.filas.sum()} filas)")


if __name__ == "__main__":
    import warnings

    from sklearn.exceptions import InconsistentVersionWarning

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    app()
