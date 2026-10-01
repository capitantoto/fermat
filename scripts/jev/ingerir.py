"""
Ingiere las respuestas de Jev y escribe infos con la estructura de `Tarea.info`.

Lee `manifiesto.jsonl` (de `exportar.py`) y uno o más `respuestas*.jsonl` (de
`correr.py`). Para cada tarea y régimen con todas sus filas respondidas calcula, con
las fórmulas de `Tarea.evaluar`, la base, `logvero`, `r2` y `accuracy`, y escribe
`{variante}-{semilla_ds}-jev_{régimen}-{semilla_tarea}-neg_log_loss.pkl` en
`sandbox/fundacionales/infos`.

Las probabilidades llegan redondeadas, y una probabilidad 0 en la clase verdadera
haría que `log_loss` la recorte a ~2e-16 (-36 nats por fila). Por eso se acotan por
debajo con `--piso` (por omisión 5e-5, medio paso de un redondeo a 4 decimales) y se
renormalizan; `logvero_sin_piso` y `ceros` (filas con probabilidad 0 en la clase
verdadera) quedan en el info para ver cuánto pesa esa decisión.

Uso, desde la raíz del repositorio:
    uv run --group fundacionales python scripts/jev/ingerir.py respuestas.jsonl
"""

import json
import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils import Bunch

dir_fundacionales = Path(__file__).resolve().parents[2] / "sandbox/fundacionales"
logger = logging.getLogger("ingerir")


def leer_jsonl(ruta: Path):
    with open(ruta, encoding="utf-8") as archivo:
        for linea in archivo:
            try:
                yield json.loads(linea)
            except json.JSONDecodeError:
                continue


def respuestas_exitosas(rutas: list[Path]) -> dict[str, dict]:
    """Última respuesta exitosa de cada id."""
    exitosas = {}
    for ruta in rutas:
        for registro in leer_jsonl(ruta):
            if registro.get("estado") == 200:
                exitosas[registro["id"]] = registro
    return exitosas


def info_tarea(entrada: dict, registros: list[dict], clf: str, piso: float) -> Bunch:
    """Info con la estructura de `Tarea.info` (base y clasificador)."""
    etiquetas = np.array(entrada["etiquetas"])
    y_eval = np.array(entrada["y_eval"])
    base = Bunch(
        probas=pd.Series(dict(entrada["base"]), name="proportion"),
    )
    base.accuracy = (y_eval == base.probas.idxmax()).mean()
    conteos = pd.Series(y_eval).value_counts()
    base.logvero = (np.log(base.probas) * conteos).sum()
    base.r2 = 0

    crudas = np.array(
        [
            [
                r["respuesta"]["answers"]["clase"]["probabilities"].get(opcion, 0.0)
                for opcion in entrada["opciones"]
            ]
            for r in registros
        ],
        dtype=np.float64,
    )
    renormalizadas = crudas / crudas.sum(axis=1, keepdims=True)
    acotadas = np.maximum(renormalizadas, piso)
    probas = acotadas / acotadas.sum(axis=1, keepdims=True)
    verdadera = np.searchsorted(etiquetas, y_eval)  # `etiquetas` está ordenado
    info = Bunch(busqueda=None, t_entrenar=0.0)
    info.probas = probas
    info.preds_proba = etiquetas[probas.argmax(axis=1)]
    info.logvero = -log_loss(y_eval, probas, normalize=False, labels=etiquetas)
    info.r2 = 1 - info.logvero / base.logvero
    info.preds = info.preds_proba
    info.accuracy = accuracy_score(info.preds, y_eval)
    info.logvero_sin_piso = -log_loss(
        y_eval, renormalizadas, normalize=False, labels=etiquetas
    )
    info.piso = piso
    info.ceros = int((crudas[np.arange(len(y_eval)), verdadera] == 0).sum())
    segundos = np.array([r.get("segundos", np.nan) for r in registros])
    info.t_probas = info.t_evaluar = info.t_inferencia = float(np.nansum(segundos))
    info.s_por_fila = float(np.nanmean(segundos))
    info.n_eval = len(y_eval)
    info.regimen = entrada["regimen"]
    info.idioma = entrada["idioma"]
    info.k_ejemplos = entrada["k_ejemplos"]
    modelos = Counter(r["respuesta"].get("model") for r in registros)
    info.modelo = modelos.most_common(1)[0][0]
    info.tokens_cobrados = sum(
        (r["respuesta"].get("usage") or {}).get("input_tokens", 0) for r in registros
    )
    info.truncadas = sum(
        bool((r["respuesta"].get("usage") or {}).get("truncated")) for r in registros
    )
    return Bunch(base=base, **{clf: info})


def main(
    respuestas: Annotated[list[Path], typer.Argument(help="respuestas*.jsonl")],
    manifiesto: Path = dir_fundacionales / "jev/manifiesto.jsonl",
    salida: Path = dir_fundacionales / "infos",
    piso: float = 5e-5,
    prefijo: Annotated[
        str, typer.Option(help="Prefijo del clasificador en los infos")
    ] = "jev",
):
    """Escribe un info por tarea y régimen completos."""
    exitosas = respuestas_exitosas(respuestas)
    salida.mkdir(parents=True, exist_ok=True)
    escritos, incompletos = 0, Counter()
    for entrada in leer_jsonl(manifiesto):
        ids = [
            f"{entrada['clf']}:{entrada['tarea']}:{i}"
            for i in range(len(entrada["y_eval"]))
        ]
        registros = [exitosas.get(i) for i in ids]
        if not all(registros):
            if any(registros):
                incompletos[entrada["regimen"]] += 1
            continue
        clf = entrada["clf"].replace("jev", prefijo, 1)
        info = info_tarea(entrada, registros, clf, piso)
        nombre = (
            f"{entrada['variante']}-{entrada['semilla_ds']}-{clf}"
            f"-{entrada['semilla_tarea']}-neg_log_loss.pkl"
        )
        with open(salida / nombre, "wb") as archivo:
            pickle.dump(info, archivo)
        escritos += 1
        logger.info(
            "%s %s: acc=%.3f r2=%.3f (sin piso %.3f, %d ceros)",
            clf,
            entrada["tarea"],
            info[clf].accuracy,
            info[clf].r2,
            1 - info[clf].logvero_sin_piso / info.base.logvero,
            info[clf].ceros,
        )
    logger.info("Escribió %d infos en %s", escritos, salida)
    if incompletos:
        logger.warning("Tareas a medio responder, por régimen: %s", dict(incompletos))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )
    typer.run(main)
