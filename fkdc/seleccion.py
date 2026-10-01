"""
Reglas de selección sobre una grilla ya evaluada, y su aplicación a posteriori.

`GridSearchCV` acepta como `refit` una función de `cv_results_` que devuelve el índice
del candidato elegido. Dos reglas:

- `parsimoniosa`: la de la tesis, `fkdc.utils.reajustar_parsimoniosamente` (R1SD).
  Entre los candidatos a menos de un desvío (el del mejor) del máximo puntaje, elige el
  más simple ordenando primero por α ascendente y después por los demás
  hiperparámetros. Las filas con α > 1 mueven el umbral y con él el candidato con
  α = 1 que resulta elegido.
- `ciega_a_alpha`: la de la corrida v6. Aplica la misma regla dentro de cada rebanada
  de α (filas con igual α) y, entre las campeonas de cada rebanada, acepta un α mayor
  solo si su campeona supera por más de un desvío a las de α menor. Su elección dentro
  de α = 1 depende solo de las filas con α = 1, y en una grilla sin α coincide con la
  regla de la tesis.

Una regla puede restringirse a un subconjunto de la grilla (`restriccion`): por
ejemplo, la de kdc sobre los anchos de banda de fkdc, que en la campaña heroica son uno
de cada tres de los de kdc.

`seleccionar_otra` aplica una regla a los `cv_results_` de una tarea ya evaluada: si
elige el mismo candidato que la búsqueda copia sus métricas y, si no, reajusta ese
candidato sobre el pliego de entrenamiento y lo evalúa como `Tarea.evaluar`.
"""

from time import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils import Bunch

from fkdc.utils import reajustar_parsimoniosamente

# Convenio de parsimonia de la tesis sin α (α se resuelve entre rebanadas):
# (columna, ascendente); con `Pipeline` (variantes `_std`) las columnas llevan `clf__`
_regularizar_ascendente = [
    ("param_bandwidth", False),
    ("param_n_neighbors", False),
    ("param_C", True),
    ("param_logreg__C", True),
    ("param_var_smoothing", False),
    ("param_max_depth", True),
]


def _columna(resultados: pd.DataFrame, nombre: str) -> str | None:
    """`param_{nombre}` o, dentro de un `Pipeline`, `param_clf__{nombre}`."""
    for columna in (f"param_{nombre}", f"param_clf__{nombre}"):
        if columna in resultados.columns:
            return columna
    return None


def _campeon_parsimonioso(resultados: pd.DataFrame, ratio_std: float):
    """Regla de parsimonia de la tesis restringida a `resultados`.

    umbral = máximo de `mean_test_score` − ratio_std · desvío de la mejor fila (el
    menor entre empatadas; NaN cuenta como 0); entre las filas que alcanzan el umbral
    se ordena por los regularizadores (sin α), en orden estable, y gana la primera.
    """
    puntaje_max = resultados.mean_test_score.max()
    std_max = resultados[
        resultados.mean_test_score.eq(puntaje_max)
    ].std_test_score.min()
    std_max = 0.0 if pd.isna(std_max) else std_max
    umbral = puntaje_max - ratio_std * std_max
    candidatas = resultados[resultados.mean_test_score.ge(umbral)]
    regularizadores = [
        (columna, ascendente)
        for nombre, ascendente in _regularizar_ascendente
        if (columna := _columna(candidatas, nombre.removeprefix("param_")))
    ]
    if regularizadores:
        por, ascendentes = map(list, zip(*regularizadores, strict=True))
        candidatas = candidatas.sort_values(
            by=por, ascending=ascendentes, kind="stable"
        )
    return candidatas.index[0]


def reajustar_ciego_a_alpha(cv_results_: dict, ratio_std: float = 1.0) -> int:
    """Selección ciega a α (corrida v6), apta como `refit` de `GridSearchCV`.

    1. Dentro de cada rebanada de α (todas las filas si la grilla no tiene α), la
       regla de parsimonia de la tesis restringida a la rebanada elige una campeona.
    2. Entre rebanadas gana la campeona de mayor puntaje medio; si otras quedan a
       menos de `ratio_std` desvíos (el de la mejor campeona), se prefiere la de menor
       α. La carga de la prueba recae sobre α > 1.
    """
    resultados = pd.DataFrame(cv_results_)
    columna_alpha = _columna(resultados, "alpha")
    if columna_alpha:
        alphas = pd.to_numeric(resultados[columna_alpha], errors="coerce").fillna(1.0)
    else:
        alphas = pd.Series(1.0, index=resultados.index)
    campeonas = []
    for alpha, rebanada in resultados.groupby(alphas):
        etiqueta = _campeon_parsimonioso(rebanada, ratio_std)
        campeonas.append(
            {
                "alpha": float(alpha),
                "indice": etiqueta,
                "puntaje": resultados.loc[etiqueta, "mean_test_score"],
                "std": resultados.loc[etiqueta, "std_test_score"],
            }
        )
    campeonas = pd.DataFrame(campeonas)
    puntaje_max = campeonas.puntaje.max()
    std_ref = campeonas.loc[campeonas.puntaje.eq(puntaje_max), "std"].min()
    std_ref = 0.0 if pd.isna(std_ref) else std_ref
    elegibles = campeonas[campeonas.puntaje.ge(puntaje_max - ratio_std * std_ref)]
    return int(elegibles.sort_values("alpha", kind="stable").iloc[0].indice)


reglas = {
    "parsimoniosa": reajustar_parsimoniosamente,
    "ciega_a_alpha": reajustar_ciego_a_alpha,
}


def aplicar(cv_results_: dict, regla: str, restriccion: dict | None = None) -> int:
    """Índice que elige `regla`, opcionalmente solo entre las filas de `restriccion`.

    `restriccion` mapea un hiperparámetro (sin prefijo) a los valores admitidos; el
    índice devuelto es el de la grilla completa.
    """
    if not restriccion:
        return int(reglas[regla](cv_results_))
    resultados = pd.DataFrame({"params": cv_results_["params"]})
    mascara = np.ones(len(resultados), dtype=bool)
    for nombre, valores in restriccion.items():
        columna = next(
            (c for c in (nombre, f"clf__{nombre}") if c in cv_results_["params"][0]),
            None,
        )
        if columna is None:
            raise KeyError(f"La grilla no tiene el hiperparámetro {nombre}")
        observados = np.array([p[columna] for p in cv_results_["params"]], float)
        admitidos = np.asarray(valores, float)
        mascara &= np.isclose(
            observados[:, None], admitidos[None, :], rtol=1e-12, atol=0
        ).any(axis=1)
    if not mascara.any():
        raise ValueError("La restricción no deja ningún candidato")
    posiciones = np.flatnonzero(mascara)
    sub = {
        clave: (
            [valor[i] for i in posiciones]
            if isinstance(valor, list)
            else np.asarray(valor)[posiciones]
        )
        for clave, valor in cv_results_.items()
    }
    return int(posiciones[reglas[regla](sub)])


def seleccionar_otra(
    tarea, nombre: str, regla: str, restriccion: dict | None = None
) -> Bunch:
    """Métricas de `tarea` para el clasificador `nombre` con otra regla de selección.

    Si la regla elige el candidato de la búsqueda (`igual=True`), copia sus métricas;
    si no, reajusta ese candidato sobre el pliego de entrenamiento y lo evalúa sobre el
    de evaluación como `Tarea.evaluar` (R² de McFadden contra el clasificador nulo).
    """
    info = tarea.info[nombre]
    busqueda = info.busqueda
    indice = aplicar(busqueda.cv_results_, regla, restriccion)
    params = busqueda.cv_results_["params"][indice]
    resultado = Bunch(
        regla=regla,
        restriccion=restriccion,
        indice=indice,
        params=params,
        igual=indice == busqueda.best_index_,
    )
    if resultado.igual:
        for campo in ("logvero", "r2", "accuracy"):
            if campo in info:
                resultado[campo] = info[campo]
        return resultado
    t0 = time()
    estimador = clone(busqueda.estimator).set_params(**params)
    estimador.fit(tarea.X_entr, tarea.y_entr)
    resultado.t_reajuste = time() - t0
    if hasattr(estimador, "predict_proba"):
        resultado.probas = estimador.predict_proba(tarea.X_eval)
        resultado.logvero = -log_loss(
            tarea.y_eval,
            resultado.probas,
            normalize=False,
            labels=tarea.dataset.etiquetas,
        )
        resultado.r2 = 1 - resultado.logvero / tarea.info.base.logvero
    resultado.accuracy = accuracy_score(estimador.predict(tarea.X_eval), tarea.y_eval)
    return resultado
