"""
Registro único de clasificadores: metadatos y fábricas perezosas.

Cada clasificador que puede correr sobre las tareas de la tesis tiene una entrada
`Clasificador` con su grupo, cómo construirlo, su grilla, dónde corre y si corre en
este equipo. Las entradas guardan rutas `"modulo:atributo"` y no objetos, así que
importar el registro no importa `torch`, `tabpfn` ni ningún módulo pesado: cada
fábrica se resuelve recién al instanciar.

Campos que deciden en qué comparaciones entra un clasificador:

- `local`: corre sin conexión en este equipo (pesos descargados una vez, cualquiera sea
  su licencia). La comparación crítica usa solo clasificadores locales; los que solo
  corren por una API (TabPFN Plus, Jev) son extras secundarios.
- `tesis`: forma parte de los resultados entregados (figuras y tablas de `fkdc/viz.py`).
- `ejecutor`: `"tarea"` si se evalúa con `fkdc.tarea.Tarea` (y por lo tanto con
  `fkdc.correr`); si no, el módulo que lo evalúa por su cuenta (texto).
- `carril`: `"cpu"` (muchas tareas a la vez, un hilo cada una), `"torch"` (modelos de
  `torch`, en serie en un proceso con varios hilos; TabPFN usa MPS) o `"api"` (servicio
  remoto con cuota).

La grilla de una entrada es un dict o una función `(nombre, n_muestras, n_atributos)`
que la adapta al dataset; las campañas pueden reemplazarla (`fkdc.campanias`). Los
oráculos construyen estimador y grilla según la variante (`por_variante`).
"""

from dataclasses import dataclass
from functools import cache
from importlib import import_module
from math import floor

from sklearn.base import clone

from fkdc import config

grupos = (
    "tesis",
    "compendio",
    "profundos",
    "fundacionales",
    "oraculo",
    "texto",
    "clasicos",
)
carriles = ("cpu", "torch", "api")


@dataclass(frozen=True)
class Clasificador:
    """Metadatos de un clasificador registrado.

    `fabrica` y `grilla` son rutas `"modulo:atributo"` (el atributo admite puntos:
    `"fkdc.config:grillas.kdc"`). La fábrica resuelve a un estimador sin ajustar (se
    clona), a un invocable sin argumentos que lo construye o, con `por_variante`, a
    `f(variante, n_muestras=..., **argumentos) -> (estimador, grilla)`. Con `semilla`,
    el estimador recibe como `random_state` la semilla de la tarea; `opciones` se suman
    a las de un `Fundacional` (p. ej., el dispositivo con que se barrió).
    """

    nombre: str
    grupo: str
    fabrica: str
    grilla: str | None = None
    local: bool = True
    carril: str = "cpu"
    semilla: bool = False
    por_variante: bool = False
    argumentos: tuple = ()
    opciones: tuple = ()
    ejecutor: str = "tarea"
    tesis: bool = False
    descripcion: str = ""

    def __post_init__(self):
        if self.grupo not in grupos:
            raise ValueError(f"Grupo desconocido: {self.grupo}")
        if self.carril not in carriles:
            raise ValueError(f"Carril desconocido: {self.carril}")


def resolver(ruta: str):
    """Objeto en `"modulo:atributo.subatributo"` (atributos o claves de dict)."""
    nombre_modulo, camino = ruta.split(":")
    objeto = import_module(nombre_modulo)
    for parte in camino.split("."):
        objeto = objeto[parte] if isinstance(objeto, dict) else getattr(objeto, parte)
    return objeto


def kdc_exacto():
    """kdc con la KDE euclídea exacta (`EuclideanKDE`) en lugar de `KernelDensity`."""
    from fkdc.fermat import KDClassifier

    return KDClassifier(metric="euclidea_exacta")


def aplica_oraculo(variante: str) -> bool:
    """Los oráculos existen para las familias sintéticas con variedad conocida."""
    from fkdc.oraculo import mapas_oraculo

    return not variante.endswith("_std") and variante.split("_")[0] in mapas_oraculo


descripciones_tesis = {
    "fkdc": "KDC con distancia de Fermat muestral",
    "kdc": "KDC euclídeo con sklearn.KernelDensity, como en la tesis",
    "fkn": "k vecinos con distancia de Fermat muestral",
    "kn": "k vecinos euclídeo",
    "gnb": "Bayes ingenuo gaussiano",
    "lr": "regresión logística",
    "svc": "máquina de soporte vectorial (RBF)",
    "gbt": "gradient boosting (HistGradientBoosting)",
}


def _entradas_texto() -> list[Clasificador]:
    """Laya y GLiNER (locales) y Jev (API) por régimen, como los nombra `fkdc.texto`.

    Se evalúan con `fkdc.texto` (no con `Tarea`): el registro solo los lista.
    """
    from fkdc.texto import familias, nombre_clf, regimenes

    entradas = []
    for regimen, abreviatura in regimenes.items():
        for familia in familias:
            for modelo, idioma in (
                ("multilingual", "es"),
                ("english", "en"),
                ("multilingual", "en"),
            ):
                entradas.append(
                    Clasificador(
                        nombre_clf(regimen, modelo, idioma, familia),
                        "texto",
                        "fkdc.texto:LayaClassifier"
                        if familia == "laya"
                        else "fkdc.texto:GlinerClassifier",
                        carril="torch",
                        ejecutor="fkdc.texto",
                        descripcion=f"{familia} ({modelo}, {idioma}), {regimen}",
                    )
                )
        entradas.append(
            Clasificador(
                f"jev_{abreviatura}",
                "texto",
                "scripts/jev",
                local=False,
                carril="api",
                ejecutor="scripts/jev",
                descripcion=f"Jev (TypeSafe, API), régimen {regimen}",
            )
        )
    return entradas


@cache
def registro() -> dict[str, Clasificador]:
    """Todas las entradas, por nombre."""
    entradas = [
        *(
            Clasificador(
                nombre,
                "tesis",
                f"fkdc.config:clasificadores.{nombre}",
                f"fkdc.config:grillas.{nombre}",
                tesis=True,
                descripcion=descripcion,
            )
            for nombre, descripcion in descripciones_tesis.items()
        ),
        Clasificador(
            "kdc_exacto",
            "tesis",
            "fkdc.registro:kdc_exacto",
            "fkdc.config:grillas.kdc",
            descripcion="kdc con la KDE euclídea exacta (EuclideanKDE)",
        ),
        *(
            Clasificador(
                nombre,
                "compendio",
                f"fkdc.compendio:clasificadores.{nombre}",
                "fkdc.compendio:grilla",
            )
            for nombre in (
                "qda",
                "lda",
                "gmm",
                "kdc_abramson",
                "isomap_kn",
                "isomap_kdc",
                "nca_kn",
                "label_spreading",
                "gpc",
                "rf",
                "et",
                "mlp",
            )
        ),
        *(
            Clasificador(
                nombre,
                "profundos",
                f"fkdc.profundos:clasificadores.{nombre}",
                carril="torch",
                semilla=True,
                # Como en su barrido (`scripts/profundos.py`): CPU con 4 hilos
                opciones=(("dispositivo", "cpu"), *hilos),
            )
            for nombre, hilos in (
                ("realmlp", (("hilos", 4),)),
                ("tabm", (("hilos", 4),)),
                ("tabicl", (("hilos", 4),)),
                ("tabdpt", ()),
            )
        ),
        *(
            Clasificador(
                nombre,
                "fundacionales",
                f"fkdc.fundacionales:clasificadores.{nombre}",
                carril="torch",
                semilla=True,
                descripcion=descripcion,
            )
            for nombre, descripcion in (
                ("tabpfn", "TabPFN-3.5 con pesos locales (licencia no comercial)"),
                ("tabpfn_fast", "TabPFN-3.5-fast con pesos locales"),
                ("tabpfn_v2", "TabPFN v2 con pesos locales (licencia priorlabs-1-1)"),
            )
        ),
        Clasificador(
            "tabpfn_ajustado",
            "fundacionales",
            "fkdc.tabpfn_ajustado:clasificador",
            carril="torch",
            semilla=True,
            descripcion="TabPFN-3.5 con pesos locales, ajustado a cada tarea",
        ),
        *(
            Clasificador(
                nombre,
                "fundacionales",
                f"fkdc.fundacionales:clasificadores.{nombre}",
                local=False,
                carril="api",
                semilla=True,
                descripcion="TabPFN por la API de Prior Labs",
            )
            for nombre in ("tabpfn_api", "tabpfn_plus")
        ),
        *(
            Clasificador(
                f"{base}_oraculo",
                "oraculo",
                "fkdc.oraculo:clasificador_oraculo",
                por_variante=True,
                argumentos=(("clf", base),),
                descripcion=f"{base} sobre las coordenadas intrínsecas φ",
            )
            for base in ("lr", "gnb", "kdc", "kn", "svc", "gbt")
        ),
        *_entradas_texto(),
        *(
            Clasificador(
                nombre,
                "clasicos",
                f"fkdc.clasicos:clasificadores.{nombre}",
                f"fkdc.clasicos:grillas.{nombre}",
                semilla=semilla,
                descripcion=descripcion,
            )
            for nombre, semilla, descripcion in (
                ("kdc_ard", False, "KDC con un ancho por variable (VC dejando uno)"),
                ("fkdc_ard", False, "fkdc sobre las variables divididas por kdc_ard"),
                ("gpc_ard", True, "proceso gaussiano con RBF anisótropo"),
                ("superlearner", True, "apilamiento convexo de clásicos rápidos"),
                ("bart", True, "BART probit (stochtree), uno contra el resto"),
            )
        ),
    ]
    resultado = {}
    for entrada in entradas:
        if entrada.nombre in resultado:
            raise ValueError(f"Nombre repetido en el registro: {entrada.nombre}")
        resultado[entrada.nombre] = entrada
    return resultado


def entrada(nombre: str) -> Clasificador:
    """La entrada de `nombre`."""
    try:
        return registro()[nombre]
    except KeyError:
        raise KeyError(f"Clasificador no registrado: {nombre}") from None


def nombres(
    grupos_elegidos=None, local: bool | None = None, ejecutor: str | None = "tarea"
) -> list[str]:
    """Nombres registrados, filtrados por grupo, por `local` y por ejecutor."""
    return [
        e.nombre
        for e in registro().values()
        if (grupos_elegidos is None or e.grupo in grupos_elegidos)
        and (local is None or e.local == local)
        and (ejecutor is None or e.ejecutor == ejecutor)
    ]


def aplica(nombre: str, variante: str) -> bool:
    """Si el clasificador `nombre` corre sobre la variante de dataset `variante`."""
    return aplica_oraculo(variante) if entrada(nombre).grupo == "oraculo" else True


def puntuacion(estimador) -> str:
    """Puntuación de la búsqueda: log-verosimilitud si hay probabilidades."""
    return "neg_log_loss" if hasattr(estimador, "predict_proba") else "accuracy"


def _filtrar_vecinos(grilla: dict, n_muestras: int, cv: int, split: float) -> dict:
    """Descarta los k que no caben en un pliego, como `config.hacer_configuraciones`."""
    n_ajuste = floor(n_muestras * (cv - 1) / cv * (1 - split))
    return {
        parametro: [k for k in valores if k < n_ajuste]
        if parametro.endswith("n_neighbors")
        else valores
        for parametro, valores in grilla.items()
    }


def preparar(
    nombre: str,
    n_muestras: int,
    n_atributos: int,
    variante: str | None = None,
    semilla: int | None = None,
    grilla: dict | None = None,
    cv: int = config.cv,
    split_evaluacion: float = config.split_evaluacion,
):
    """(estimador sin ajustar, grilla, puntuación) de `nombre` para un dataset.

    `grilla`, si se da, reemplaza la del registro (las campañas la fijan así); en ambos
    casos se descartan los números de vecinos que no caben en un pliego.
    """
    e = entrada(nombre)
    if e.ejecutor != "tarea":
        raise ValueError(f"{nombre} no se evalúa con Tarea, sino con {e.ejecutor}")
    fabrica = resolver(e.fabrica)
    if e.por_variante:
        if variante is None:
            raise ValueError(f"{nombre} necesita la variante del dataset")
        estimador, grilla_propia = fabrica(
            variante,
            n_muestras=n_muestras,
            cv=cv,
            split_evaluacion=split_evaluacion,
            **dict(e.argumentos),
        )
    else:
        estimador = clone(fabrica) if hasattr(fabrica, "get_params") else fabrica()
        if e.grilla is None:
            grilla_propia = {}
        else:
            objeto = resolver(e.grilla)
            grilla_propia = (
                objeto(nombre, n_muestras, n_atributos) if callable(objeto) else objeto
            )
    if e.semilla:
        estimador = estimador.set_params(random_state=semilla)
    if e.opciones:
        opciones = {
            **(estimador.get_params().get("opciones") or {}),
            **dict(e.opciones),
        }
        estimador = estimador.set_params(opciones=opciones)
    grilla = dict(grilla_propia if grilla is None else grilla)
    grilla = _filtrar_vecinos(grilla, n_muestras, cv, split_evaluacion)
    return estimador, grilla, puntuacion(estimador)
