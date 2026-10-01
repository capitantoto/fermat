"""
Exporta las tareas de la tesis como solicitudes para Jev (TypeSafe, `/v1/systemone`).

Cada solicitud clasifica una observación del pliego de evaluación de una tarea, con
el mismo texto que lee Laya (`fkdc.texto.Redactor`): mismo estado, misma pregunta,
mismas opciones y, en los regímenes con ejemplos, el mismo `k` (se cuenta con el
tokenizador de Laya multilingüe y el mismo presupuesto de `max_len_ic`).

Escribe en `--salida` (por omisión `sandbox/fundacionales/jev`):

- `solicitudes.jsonl.gz`: una línea por solicitud, `{"id", "regimen",
  "tokens_estimados", "cuerpo"}`; `cuerpo` es el JSON que se envía a Jev.
- `manifiesto.jsonl`: una línea por tarea y régimen, con lo que `ingerir.py`
  necesita para calcular las métricas de `Tarea.evaluar` (`y_eval`, orden de clases,
  opciones y probabilidades de la base).
- `jev.zip`: el paquete para quien tiene la clave (`correr.py`, `LEEME.md` y las
  solicitudes); el manifiesto queda acá. `--solo-paquete` lo rearma sin exportar.

Uso, desde la raíz del repositorio:
    uv run --group fundacionales python scripts/jev/exportar.py \
        --regimen zero-shot --regimen nombre --regimen descripcion --regimen codigo
"""

import gzip
import json
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from sklearn.preprocessing import StandardScaler

from fkdc import texto
from fkdc.tarea import Tarea

logger = logging.getLogger("exportar")

dir_jev = Path(__file__).parent
regimenes_por_omision = ["zero-shot", "nombre", "descripcion", "codigo"]


def particion(tarea) -> tuple:
    """Partición de `Tarea` (misma semilla y `train_test_split`) y parámetros."""
    dataset = texto.cargar_dataset(tarea.ruta_dataset)
    t = Tarea(dataset, {}, semilla=tarea.semilla_tarea)
    X_entr, X_eval = t.X_entr, t.X_eval
    if tarea.estandarizar:  # como el `StandardScaler` que `Tarea` antepone
        escalador = StandardScaler().fit(X_entr)
        X_entr, X_eval = escalador.transform(X_entr), escalador.transform(X_eval)
    base = pd.Series(t.y_entr).value_counts(normalize=True)  # como `Tarea.evaluar`
    return dataset, X_entr, X_eval, t.y_entr, t.y_eval, base


def main(
    regimen: Annotated[
        list[str] | None, typer.Option(help="Regímenes (repetible)")
    ] = None,
    patron: Annotated[list[str] | None, typer.Option(help="Glob de tareas")] = None,
    idioma: str = "es",
    modelo_jev: str = "jev-latest",
    max_len_ic: int = 2048,
    salida: Path = texto.dir_fundacionales / "jev",
    solo_paquete: Annotated[
        bool, typer.Option(help="Solo rearma jev.zip (p. ej., si cambió correr.py)")
    ] = False,
):
    """Escribe solicitudes, manifiesto y paquete para Jev."""
    if solo_paquete:
        return empaquetar(salida)
    regimenes = regimen or regimenes_por_omision
    salida.mkdir(parents=True, exist_ok=True)
    contar = texto.contador_tokens("multilingual")
    presupuesto = texto.presupuesto_estado(max_len_ic, "multilingual")
    tareas = texto.tareas_tesis(patron or ["*"])
    totales = {r: [0, 0] for r in regimenes}  # solicitudes, tokens
    ruta_solicitudes = salida / "solicitudes.jsonl.gz"
    with (
        gzip.open(ruta_solicitudes, "wt", encoding="utf-8") as solicitudes,
        open(salida / "manifiesto.jsonl", "w", encoding="utf-8") as manifiesto,
    ):
        for n_tarea, tarea in enumerate(tareas, start=1):
            dataset, X_entr, X_eval, y_entr, y_eval, base = particion(tarea)
            parametros = texto.parametros_dataset(dataset)
            for nombre_regimen in regimenes:
                if not texto.aplica(nombre_regimen, tarea.variante):
                    continue
                redactor = texto.Redactor(
                    tarea.variante, nombre_regimen, idioma, parametros
                )
                ejemplos, k = "", 0
                if nombre_regimen in texto.regimenes_con_ejemplos:
                    ejemplos, k = redactor.ejemplos(
                        X_entr, y_entr, tarea.semilla_tarea, contar, presupuesto
                    )
                pregunta = redactor.pregunta(dataset.etiquetas)
                tokens_pregunta = contar(json.dumps(pregunta, ensure_ascii=False))
                clf = f"jev_{texto.regimenes[nombre_regimen]}"
                for i, x in enumerate(X_eval):
                    estado = redactor.estado(x, ejemplos)
                    tokens = contar(estado) + tokens_pregunta
                    cuerpo = {
                        "model": modelo_jev,
                        "state": estado,
                        "questions": {"clase": pregunta},
                    }
                    linea = {
                        "id": f"{clf}:{tarea.id}:{i}",
                        "regimen": nombre_regimen,
                        "tokens_estimados": tokens,
                        "cuerpo": cuerpo,
                    }
                    solicitudes.write(json.dumps(linea, ensure_ascii=False) + "\n")
                    totales[nombre_regimen][0] += 1
                    totales[nombre_regimen][1] += tokens
                registro = {
                    "tarea": tarea.id,
                    "variante": tarea.variante,
                    "semilla_ds": tarea.semilla_ds,
                    "semilla_tarea": tarea.semilla_tarea,
                    "regimen": nombre_regimen,
                    "clf": clf,
                    "idioma": idioma,
                    "etiquetas": [str(e) for e in dataset.etiquetas],
                    "opciones": [redactor.clave(e) for e in dataset.etiquetas],
                    "y_eval": [str(y) for y in y_eval],
                    "base": [[str(e), float(p)] for e, p in base.items()],
                    "k_ejemplos": k,
                    "tokens_pregunta": tokens_pregunta,
                }
                manifiesto.write(json.dumps(registro, ensure_ascii=False) + "\n")
            if n_tarea % 50 == 0:
                logger.info("%d/%d tareas", n_tarea, len(tareas))

    for nombre_regimen, (n, tokens) in totales.items():
        logger.info(
            "%s: %d solicitudes, %.1f M tokens estimados",
            nombre_regimen,
            n,
            tokens / 1e6,
        )
    empaquetar(salida)


def empaquetar(salida: Path):
    """Arma `jev.zip` con `correr.py`, `LEEME.md` y las solicitudes ya exportadas."""
    with zipfile.ZipFile(salida / "jev.zip", "w") as paquete:
        for nombre in ["correr.py", "LEEME.md"]:
            paquete.write(dir_jev / nombre, f"jev/{nombre}", zipfile.ZIP_DEFLATED)
        # Ya comprimido: se guarda tal cual
        paquete.write(salida / "solicitudes.jsonl.gz", "jev/solicitudes.jsonl.gz")
    shutil.copy(dir_jev / "correr.py", salida / "correr.py")
    logger.info("Escribió %s", salida)


if __name__ == "__main__":
    import warnings

    from sklearn.exceptions import InconsistentVersionWarning

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    typer.run(main)
