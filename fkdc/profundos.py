"""
Clasificadores profundos y fundacionales tabulares como líneas de base adicionales.

Complementan a `fkdc.fundacionales` (TabPFN) con dos redes entrenadas desde cero
con hiperparámetros por defecto ajustados por sus autores (RealMLP y TabM, de
`pytabkit`) y dos modelos fundacionales de aprendizaje en contexto (TabICL y
TabDPT). Como allí, no forman parte de los resultados de la tesis: tienen registro
propio (`clasificadores`, `grillas` vacías) y se ajustan directamente sobre el
pliego de entrenamiento de cada tarea, sin búsqueda de hiperparámetros.

Cada clasificador es un `Fundacional`, que importa su fábrica (y con ella `torch`)
recién al ajustarse y recibe como `random_state` la semilla de la tarea. El
dispositivo es el argumento `dispositivo` de cada fábrica o, si es `None`, la
variable de entorno `FKDC_DISPOSITIVO`; "auto" (el valor por omisión) elige MPS si
está disponible y, si no, CPU.

El barrido sobre las tareas de la tesis está en `scripts/profundos.py`.
"""

import os

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch

from fkdc.fundacionales import Fundacional


def resolver_dispositivo(dispositivo: str | None = None) -> str:
    """Nombre del dispositivo de `torch`: "auto" elige MPS si está disponible."""
    dispositivo = dispositivo or os.environ.get("FKDC_DISPOSITIVO", "auto")
    if dispositivo == "auto":
        import torch

        dispositivo = "mps" if torch.backends.mps.is_available() else "cpu"
    return dispositivo


def realmlp(
    random_state: int | None = None,
    dispositivo: str | None = None,
    hilos: int | None = None,
):
    """RealMLP con los hiperparámetros por defecto ajustados (`RealMLP_TD`)."""
    from pytabkit import RealMLP_TD_Classifier

    return RealMLP_TD_Classifier(
        device=resolver_dispositivo(dispositivo),
        random_state=random_state,
        n_threads=hilos,
        verbosity=0,
    )


def tabm(
    random_state: int | None = None,
    dispositivo: str | None = None,
    hilos: int | None = None,
):
    """TabM con los hiperparámetros por defecto de sus autores (`TabM_D`)."""
    from pytabkit import TabM_D_Classifier

    return TabM_D_Classifier(
        device=resolver_dispositivo(dispositivo),
        random_state=random_state,
        n_threads=hilos,
        verbosity=0,
    )


def tabicl(
    random_state: int | None = None,
    dispositivo: str | None = None,
    hilos: int | None = None,
):
    """TabICL v2 con configuración por defecto (ensamble de 8 permutaciones).

    La primera vez descarga el punto de control de Hugging Face (`jingang/TabICL`,
    sin registro).
    """
    from tabicl import TabICLClassifier

    return TabICLClassifier(
        device=resolver_dispositivo(dispositivo),
        random_state=random_state,
        n_jobs=hilos,
    )


class TabDPTEnsamble(ClassifierMixin, BaseEstimator):
    """TabDPT con la predicción por defecto de sus autores, como estimador de sklearn.

    `TabDPTClassifier` exige etiquetas enteras 0..K-1, no expone `classes_` y su
    `predict_proba` es una sola pasada sin ensamble; su `predict` promedia
    `n_ensembles=8` pasadas con permutaciones de atributos y clases. Este adaptador
    codifica las etiquetas y usa ese mismo ensamble (`ensemble_predict_proba`) para
    las probabilidades, con `random_state` como semilla de las permutaciones. Los
    atributos pasan en simple precisión (la del modelo): `tabdpt` mueve los arreglos
    al dispositivo antes de convertirlos, y MPS no admite doble precisión.

    La primera vez descarga los pesos de Hugging Face (`Layer6/TabDPT`, sin
    registro).
    """

    def __init__(
        self,
        n_ensembles: int = 8,
        random_state: int | None = None,
        dispositivo: str | None = None,
    ):
        self.n_ensembles = n_ensembles
        self.random_state = random_state
        self.dispositivo = dispositivo

    def fit(self, X, y):
        """Codifica las etiquetas y guarda el contexto de entrenamiento."""
        from tabdpt import TabDPTClassifier

        self.codificador_ = LabelEncoder().fit(y)
        self.classes_ = self.codificador_.classes_
        self.modelo_ = TabDPTClassifier(
            device=resolver_dispositivo(self.dispositivo), verbose=False
        )
        self.modelo_.fit(
            np.asarray(X, dtype=np.float32),
            self.codificador_.transform(y).astype(np.int64),
        )
        return self

    def predict_proba(self, X):
        """Probabilidades promediadas sobre el ensamble de permutaciones."""
        return self.modelo_.ensemble_predict_proba(
            np.asarray(X, dtype=np.float32),
            n_ensembles=self.n_ensembles,
            seed=self.random_state,
        )

    def predict(self, X):
        """Clase de máxima probabilidad."""
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def tabdpt(random_state: int | None = None, dispositivo: str | None = None):
    """TabDPT v1.3 con ensamble de 8 pasadas (cf. `TabDPTEnsamble`)."""
    return TabDPTEnsamble(random_state=random_state, dispositivo=dispositivo)


clasificadores = Bunch(
    realmlp=Fundacional("fkdc.profundos:realmlp"),
    tabm=Fundacional("fkdc.profundos:tabm"),
    tabicl=Fundacional("fkdc.profundos:tabicl"),
    tabdpt=Fundacional("fkdc.profundos:tabdpt"),
)
grillas = {nombre: {} for nombre in clasificadores}


def instanciar(nombre: str, semilla: int | None = None, **opciones) -> Fundacional:
    """Copia sin ajustar del clasificador `nombre`, con `random_state=semilla`.

    `opciones` se suman a las de la fábrica (p. ej., `dispositivo="cpu"`).
    """
    base = clasificadores[nombre]
    return Fundacional(
        base.fabrica, {**(base.opciones or {}), **opciones}, random_state=semilla
    )
