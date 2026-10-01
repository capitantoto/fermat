"""
Clasificadores fundacionales como líneas de base adicionales.

No forman parte de los resultados de la tesis: tienen registro propio
(`clasificadores`, `grillas`), configuraciones propias (`dir_configs`) y resultados
propios (`dir_infos`), de modo que `config.clasificadores` y las figuras de `viz.py`
no cambian. Las tareas son las de la tesis (datasets de `sandbox/v5/datasets`, mismas
semillas y mismo `train_test_split`), pero sin búsqueda de hiperparámetros: con la
grilla vacía, `Tarea.entrenar` ajusta directamente sobre el pliego de entrenamiento.

Cada clasificador es un `Fundacional`, que importa su fábrica (y con ella `torch`,
`tabpfn` o `tabpfn_client`) recién al ajustarse. Para sumar otro basta una línea en
`clasificadores` con la ruta `"modulo:fabrica"` de un invocable que devuelva un
estimador de scikit-learn.

Uso, desde la raíz del repositorio:
    python -m fkdc.fundacionales configurar
    python -m fkdc.fundacionales correr tabpfn --patron 'lunas_lo-1075-*'
    python -m fkdc.fundacionales verificar --patron 'pinguinos-None-*-1075-*'
    python -m fkdc.fundacionales comparar tabpfn tabpfn_api
    python -m fkdc.fundacionales resumir
"""

import hashlib
import inspect
import logging
import os
import pickle
from importlib import import_module
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils import Bunch

from fkdc import config, dir_raiz

logger = logging.getLogger(__name__)

dir_fundacionales = dir_raiz / "sandbox/fundacionales"
dir_configs = dir_fundacionales / "configs"
dir_infos = dir_fundacionales / "infos"
dir_datasets = config.dir_ejecucion.parent / "datasets"


class Fundacional(ClassifierMixin, BaseEstimator):
    """Clasificador sin hiperparámetros que se construye recién al ajustarlo.

    `fabrica` es la ruta `"modulo:atributo"` de un invocable que devuelve un
    estimador de scikit-learn; recibe `opciones` como argumentos nombrados y, si lo
    acepta, `random_state`. Construirlo en `fit` evita importar `torch` al cargar el
    registro. `local` es solo un metadato para separar resultados en las tablas: si
    el modelo corre sin red en esta máquina (pesos descargados una vez) o si es un
    modelo solo-API, cuya inferencia consume la cuota de Prior Labs.

    Las probabilidades de la última `X` quedan memorizadas, de modo que `predict`
    (el argmax de `predict_proba`) no repite la inferencia, que en la API se cobra.
    Por eso, en los infos de estos clasificadores el tiempo de inferencia es
    `t_probas`, y `t_evaluar` apenas mide el argmax. Las probabilidades se
    renormalizan en doble precisión: TabPFN las calcula en simple precisión y sus
    filas suman 1 solo hasta ~1e-7.
    """

    def __init__(
        self,
        fabrica: str,
        opciones: dict | None = None,
        random_state: int | None = None,
        local: bool = True,
    ):
        self.fabrica = fabrica
        self.opciones = opciones
        self.random_state = random_state
        self.local = local

    def _construir(self):
        nombre_modulo, nombre_fabrica = self.fabrica.split(":")
        fabrica = getattr(import_module(nombre_modulo), nombre_fabrica)
        argumentos = dict(self.opciones or {})
        if "random_state" in inspect.signature(fabrica).parameters:
            argumentos["random_state"] = self.random_state
        return fabrica(**argumentos)

    def fit(self, X, y):
        """Construye el estimador y lo ajusta."""
        self.estimador_ = self._construir().fit(X, y)
        self.classes_ = np.asarray(self.estimador_.classes_)
        self._memoria = None
        return self

    def predict_proba(self, X):
        """Probabilidades por clase, memorizadas para la última `X`."""
        huella = hashlib.sha1(np.ascontiguousarray(X).tobytes()).hexdigest()
        if self._memoria is None or self._memoria[0] != huella:
            probas = np.asarray(self.estimador_.predict_proba(X), dtype=np.float64)
            self._memoria = (huella, probas / probas.sum(axis=1, keepdims=True))
        return self._memoria[1]

    def predict(self, X):
        """Clase de máxima probabilidad."""
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def _cargar_credenciales():
    """Carga la clave de Prior Labs de `.env` (gitignorado) sin pisar el entorno."""
    from dotenv import load_dotenv

    load_dotenv(dir_raiz / ".env", override=False)


def _no_guardar_clave(token: str) -> None:
    """Reemplaza `tabpfn.browser_auth.save_token`: la clave vive solo en `.env`."""


def tabpfn_local(
    version: str = "v3.5",
    random_state: int | None = None,
    dispositivo: str | None = None,
):
    """TabPFN con pesos locales y configuración por defecto.

    `dispositivo` por omisión es la variable de entorno `FKDC_DISPOSITIVO` o, si no
    está definida, "auto" (MPS si está disponible; si no, CPU). La primera vez
    descarga los pesos, para lo cual la cuenta de Prior Labs debe haber aceptado la
    licencia de TabPFN-3.5.
    """
    _cargar_credenciales()
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")  # nunca abrir el navegador
    os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", "1")  # no recargar entre tareas
    from tabpfn import TabPFNClassifier, browser_auth
    from tabpfn.constants import ModelVersion

    browser_auth.save_token = _no_guardar_clave
    dispositivo = dispositivo or os.environ.get("FKDC_DISPOSITIVO", "auto")
    return TabPFNClassifier.create_default_for_version(
        ModelVersion(version), random_state=random_state, device=dispositivo
    )


def tabpfn_api(
    version: str = "v3.5", plus: bool = True, random_state: int | None = None
):
    """TabPFN por la API de Prior Labs (`tabpfn-client`), con configuración por defecto.

    Con `plus=True`, TabPFN-3.5-Plus: el modelo por defecto del cliente, que suma los
    sistemas de preprocesamiento y texto del servidor. Con `plus=False`
    (`paper_version`), el modelo base sin esos sistemas, comparable con
    `tabpfn_local`.
    """
    _cargar_credenciales()
    from tabpfn_client import TabPFNClassifier

    return TabPFNClassifier.create_default_for_version(
        version, paper_version=not plus, random_state=random_state
    )


clasificadores = Bunch(
    tabpfn=Fundacional("fkdc.fundacionales:tabpfn_local"),
    tabpfn_fast=Fundacional(
        "fkdc.fundacionales:tabpfn_local", {"version": "v3.5-fast"}
    ),
    tabpfn_api=Fundacional(
        "fkdc.fundacionales:tabpfn_api", {"plus": False}, local=False
    ),
    tabpfn_plus=Fundacional("fkdc.fundacionales:tabpfn_api", local=False),
)
grillas = {nombre: {} for nombre in clasificadores}


def instanciar(nombre: str, semilla: int | None = None) -> Fundacional:
    """Copia sin ajustar del clasificador `nombre`, con `random_state=semilla`."""
    return clone(clasificadores[nombre]).set_params(random_state=semilla)


def saldo_api() -> int:
    """Tokens que le quedan a la cuenta de la API (el menor saldo, diario o mensual)."""
    _cargar_credenciales()
    from tabpfn_client.client import ServiceClient
    from tabpfn_client.config import get_access_token

    uso = ServiceClient.get_api_usage(get_access_token())
    saldos = [
        int(uso[f"{periodo}_limit"]) - int(uso[f"{periodo}_usage"])
        for periodo in ("daily", "monthly")
        if int(uso[f"{periodo}_limit"]) != -1  # -1: sin límite
    ]
    return min(saldos, default=np.iinfo(np.int64).max)


def _partes(ruta: Path) -> dict:
    """Campos del nombre `{dataset}-{semilla_ds}-{clf}-{semilla_tarea}-{puntuacion}`."""
    campos = ["dataset", "semilla_ds", "clf", "semilla_tarea", "puntuacion"]
    return dict(zip(campos, ruta.stem.split("-"), strict=True))


def _rutas(directorio: Path, clfs, patrones=("*",), sufijo=".pkl") -> list[Path]:
    """Rutas de `directorio` que coinciden con algún patrón y son de algún clf."""
    rutas = {
        r
        for patron in patrones
        for r in directorio.glob(f"{patron}{sufijo}")
        if r.stem.count("-") == 4  # ignora archivos ajenos a la convención
    }
    return sorted(r for r in rutas if _partes(r)["clf"] in clfs)


app = typer.Typer(add_completion=False)


@app.command()
def configurar():
    """Escribe una config por tarea de la tesis y clasificador fundacional."""
    from fkdc.datasets import datasets as variantes

    bases = sorted({variante.removesuffix("_std") for variante in variantes})
    config.hacer_configuraciones(
        dir_datasets=dir_datasets,
        dir_configs=dir_configs,
        registro="fkdc.fundacionales",
        nombres_datasets=bases,
    )
    logger.info("Configs en %s", dir_configs)


@app.command()
def correr(
    clasificador: str,
    patron: Annotated[list[str] | None, typer.Option(help="Glob de configs")] = None,
    reserva_api: Annotated[int, typer.Option(help="Tokens a no consumir")] = 500_000,
):
    """Corre las tareas de `clasificador` que aún no tienen info.

    Las recorre repetición por repetición (todas las variantes con la primera
    semilla, luego con la segunda, etc.), de modo que una corrida interrumpida, por
    ejemplo al agotarse el saldo diario de la API, deja resultados parejos. Los
    modelos locales corren en MPS si está disponible y, si una tarea falla ahí, se
    reintenta en CPU; el info anota el `dispositivo` usado ("mps", "cpu" o "api").
    """
    from fkdc import process  # diferido: process importa este módulo

    def repeticion(ruta):
        partes = _partes(ruta)
        semilla_ds = partes["semilla_ds"]
        return semilla_ds if semilla_ds != "None" else partes["semilla_tarea"]

    dir_infos.mkdir(parents=True, exist_ok=True)
    rutas = _rutas(dir_configs, [clasificador], patron or ["*"], sufijo=".yaml")
    pendientes = sorted(
        (r for r in rutas if not (dir_infos / f"{r.stem}.pkl").exists()),
        key=lambda ruta: (repeticion(ruta), ruta.name),
    )
    logger.info(
        "%s: %d tareas, %d pendientes", clasificador, len(rutas), len(pendientes)
    )

    def correr_en(ruta, dispositivo):
        """Corre la tarea en `dispositivo` y lo anota en el info."""
        previo = os.environ.get("FKDC_DISPOSITIVO")
        if dispositivo != "api":
            os.environ["FKDC_DISPOSITIVO"] = dispositivo
        try:
            process.main(ruta, dir_trabajo=dir_infos)
        finally:
            os.environ.pop("FKDC_DISPOSITIVO", None)
            if previo is not None:
                os.environ["FKDC_DISPOSITIVO"] = previo
        ruta_info = dir_infos / f"{ruta.stem}.pkl"
        info = pickle.load(open(ruta_info, "rb"))
        if "r2" not in info[clasificador]:  # `Tarea.evaluar` traga excepciones
            raise RuntimeError(f"Evaluación incompleta en {ruta.stem}")
        info[clasificador].dispositivo = dispositivo
        pickle.dump(info, open(ruta_info, "wb"))

    usa_api = not clasificadores[clasificador].local
    dispositivos = ["api"] if usa_api else [_dispositivo_local()]
    if dispositivos == ["mps"]:
        dispositivos.append("cpu")  # respaldo, tarea por tarea, si MPS falla
    fallas = 0
    for i, ruta in enumerate(pendientes, start=1):
        if usa_api and (saldo := saldo_api()) < reserva_api:
            logger.warning("Saldo de la API (%d) bajo la reserva; me detengo", saldo)
            break
        logger.info("[%d/%d] %s", i, len(pendientes), ruta.stem)
        for dispositivo in dispositivos:
            try:
                correr_en(ruta, dispositivo)
                break
            except Exception:
                logger.exception("Falló %s en %s", ruta.stem, dispositivo)
                (dir_infos / f"{ruta.stem}.pkl").unlink(missing_ok=True)
        else:
            fallas += 1
            if fallas >= 3:
                logger.error("Tres fallas; me detengo")
                break


def _dispositivo_local() -> str:
    """Dispositivo para los modelos locales: `FKDC_DISPOSITIVO` o MPS si existe."""
    dispositivo = os.environ.get("FKDC_DISPOSITIVO", "auto")
    if dispositivo == "auto":
        import torch

        dispositivo = "mps" if torch.backends.mps.is_available() else "cpu"
    return dispositivo


def _cargar_basicos(directorio: Path, clfs, patrones=("*",)) -> pd.DataFrame:
    """Campos escalares de cada info de `directorio`, uno por fila."""
    campos = ["r2", "accuracy", "logvero", "t_entrenar", "t_probas", "t_evaluar"]
    registros = []
    for ruta in _rutas(directorio, clfs, patrones):
        partes = _partes(ruta)
        info = pickle.load(open(ruta, "rb"))
        registro = {campo: info[partes["clf"]].get(campo) for campo in campos}
        registros.append({**partes, **registro, "base_logvero": info.base.logvero})
    return pd.DataFrame(registros).astype(dict.fromkeys(campos, float))


@app.command()
def verificar(
    patron: Annotated[list[str], typer.Option(help="Glob de infos de la tesis")],
):
    """Contrasta la partición de `Tarea` con los infos de la tesis.

    Para cada info de `sandbox/v5/infos` que coincide con `patron`, rehace la
    partición y verifica que `base.logvero` coincida exactamente y que `y_eval`
    reproduzca exactamente la `accuracy` y la `logvero` guardadas.
    """
    from fkdc.datasets import Dataset  # noqa: F811
    from fkdc.tarea import Tarea

    for ruta in _rutas(config.dir_ejecucion, list(config.clasificadores), patron):
        partes = _partes(ruta)
        info = pickle.load(open(ruta, "rb"))
        nombre_ds = partes["dataset"].removesuffix("_std")
        if partes["semilla_ds"] != "None":
            nombre_ds += f"-{partes['semilla_ds']}"
        dataset = Dataset.cargar(dir_datasets / f"{nombre_ds}.pkl")
        tarea = Tarea(dataset, {}, semilla=int(partes["semilla_tarea"]))
        tarea.evaluar()  # sin clasificadores: solo la base
        info_clf = info[partes["clf"]]
        coincide = {
            "base_logvero": tarea.info.base.logvero == info.base.logvero,
            "accuracy": accuracy_score(info_clf.preds, tarea.y_eval)
            == info_clf.accuracy,
        }
        if "probas" in info_clf:
            logvero = -log_loss(
                tarea.y_eval,
                info_clf.probas,
                normalize=False,
                labels=dataset.etiquetas,
            )
            coincide["logvero"] = logvero == info_clf.logvero
        estado = "OK" if all(coincide.values()) else "DIFIERE"
        print(f"{estado} {ruta.stem} {coincide}")


@app.command()
def comparar(
    clf_a: str,
    clf_b: str,
    patron: Annotated[list[str] | None, typer.Option(help="Glob de infos")] = None,
):
    """Concordancia de probabilidades entre dos clasificadores fundacionales."""
    filas = []
    for ruta_a in _rutas(dir_infos, [clf_a], patron or ["*"]):
        ruta_b = ruta_a.with_name(ruta_a.name.replace(f"-{clf_a}-", f"-{clf_b}-"))
        if not ruta_b.exists():
            continue
        info_a = pickle.load(open(ruta_a, "rb"))[clf_a]
        info_b = pickle.load(open(ruta_b, "rb"))[clf_b]
        diferencia = np.abs(info_a.probas - info_b.probas)
        filas.append(
            {
                "tarea": ruta_a.stem.replace(f"-{clf_a}", ""),
                "dif_max": diferencia.max(),
                "dif_media": diferencia.mean(),
                "argmax_igual": (info_a.preds == info_b.preds).mean(),
                f"r2_{clf_a}": info_a.r2,
                f"r2_{clf_b}": info_b.r2,
            }
        )
    print(pd.DataFrame(filas).to_string(index=False))


@app.command()
def resumir(
    ruta_salida: Path = dir_fundacionales / "resumen-tabpfn.csv",
):
    """Mediana e IQR de R² y accuracy por variante de dataset y clasificador.

    Junta los clasificadores fundacionales con los ocho de la tesis y advierte si
    alguna tarea fundacional no reproduce la `base.logvero` de la tesis.
    """
    from fkdc.datasets import datasets as variantes

    tesis = _cargar_basicos(config.dir_ejecucion, list(config.clasificadores))
    nuevos = _cargar_basicos(dir_infos, list(clasificadores))
    clave = ["dataset", "semilla_ds", "semilla_tarea"]
    bases = tesis.drop_duplicates(clave).set_index(clave).base_logvero
    contraste = nuevos.join(bases, on=clave, rsuffix="_tesis")
    distintas = (contraste.base_logvero != contraste.base_logvero_tesis).sum()
    if distintas:
        logger.warning("%d tareas no reproducen la base.logvero de la tesis", distintas)
    else:
        logger.info("Las %d tareas reproducen la base.logvero de la tesis", len(nuevos))

    todos = pd.concat([tesis, nuevos], ignore_index=True)
    todos["t_inferencia"] = todos.t_probas.fillna(todos.t_evaluar)

    def iqr(serie):
        return serie.quantile(0.75) - serie.quantile(0.25)

    resumen = todos.groupby(["dataset", "clf"]).agg(
        n=("accuracy", "size"),
        r2_mediana=("r2", "median"),
        r2_iqr=("r2", iqr),
        accuracy_mediana=("accuracy", "median"),
        accuracy_iqr=("accuracy", iqr),
        t_entrenar_mediana=("t_entrenar", "median"),
        t_inferencia_mediana=("t_inferencia", "median"),
    )
    orden_clfs = [*config.clasificadores, *clasificadores]
    resumen = resumen.reset_index().sort_values(
        ["dataset", "clf"],
        key=lambda col: col.map(
            {
                n: i
                for i, n in enumerate(
                    variantes if col.name == "dataset" else orden_clfs
                )
            }
        ),
    )
    # Los de la tesis corren localmente; los fundacionales, según su metadato
    locales = {nombre: clf.local for nombre, clf in clasificadores.items()}
    resumen.insert(2, "local", resumen.clf.map(locales).fillna(True).astype(bool))
    resumen.to_csv(ruta_salida, index=False, float_format="%.4g")
    logger.info("Escribió %s", ruta_salida)


if __name__ == "__main__":
    import warnings

    from sklearn.exceptions import InconsistentVersionWarning

    # Los datasets se guardaron corriendo `fkdc/datasets.py` como script, así que
    # pickle busca `Dataset` en `__main__`.
    from fkdc.datasets import Dataset  # noqa: F401

    # Los infos de la tesis se guardaron con otra versión de scikit-learn
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    app()
