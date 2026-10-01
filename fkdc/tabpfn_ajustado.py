"""
TabPFN-3.5 ajustado (_fine-tuned_) a cada tarea, como clasificador fundacional extra.

`tabpfn` (`fkdc.fundacionales`) usa los pesos congelados: aprende solo en contexto.
`tabpfn_ajustado` parte de los mismos pesos, los ajusta por descenso de gradiente sobre
el pliego de entrenamiento de la tarea con `tabpfn.finetuning.FinetunedTabPFNClassifier`
y predice en contexto con los pesos ajustados. Como no tiene grilla, `Tarea.entrenar`
le pasa solo `X_entr, y_entr`: la validación interna (para la detención temprana) sale
de ese pliego, estratificada y con la semilla de la tarea, y el pliego de evaluación
nunca se ve durante el ajuste.

Configuración: la de la biblioteca (30 épocas, AdamW con tasa 1e-5, calentamiento y
decaimiento coseno, 10 % de validación, paciencia 8, 2 estimadores al ajustar y validar,
8 al predecir, como el congelado), salvo la métrica de detención temprana: log-loss en
lugar de ROC AUC, porque la tesis puntúa por log-verosimilitud y porque con ~40 filas de
validación el ROC AUC se satura en 1 y la detención temprana no distingue épocas.
"""

import os
import warnings
from contextlib import contextmanager

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from fkdc.fundacionales import Fundacional, _dispositivo_local, _no_guardar_clave


@contextmanager
def _sin_modelo_compartido():
    """Desactiva, mientras dura el ajuste, la caché de modelos construidos de TabPFN.

    Con `TABPFN_MODEL_CACHE_SIZE` positivo (lo fija `tabpfn_local`), TabPFN comparte el
    modelo construido por referencia; el ajuste lo modificaría en el lugar y
    contaminaría las tareas siguientes del mismo proceso, ajustadas o congeladas.
    """
    previo = os.environ.get("TABPFN_MODEL_CACHE_SIZE")
    os.environ["TABPFN_MODEL_CACHE_SIZE"] = "0"
    try:
        yield
    finally:
        if previo is None:
            os.environ.pop("TABPFN_MODEL_CACHE_SIZE", None)
        else:
            os.environ["TABPFN_MODEL_CACHE_SIZE"] = previo


class TabPFNAjustado(ClassifierMixin, BaseEstimator):
    """TabPFN ajustado al pliego de entrenamiento con la configuración del módulo.

    `metrica` es la de la detención temprana ("log_loss" o "roc_auc"). `dispositivo`
    por omisión es el de `fkdc.fundacionales._dispositivo_local` (`FKDC_DISPOSITIVO` o
    MPS si existe). Los pesos se leen del caché local; si faltan, la descarga exige
    haber aceptado la licencia, como en `tabpfn_local`. `predict_proba` devuelve las
    probabilidades de TabPFN tal cual, con columnas en el orden de `classes_`: en el
    registro va envuelto en un `Fundacional`, que las renormaliza en doble precisión y
    las memoriza para `predict`.
    """

    def __init__(
        self,
        version: str = "v3.5",
        metrica: str = "log_loss",
        random_state: int | None = None,
        dispositivo: str | None = None,
    ):
        self.version = version
        self.metrica = metrica
        self.random_state = random_state
        self.dispositivo = dispositivo

    def fit(self, X, y):
        """Ajusta los pesos sobre `X, y` y prepara el modelo de inferencia."""
        os.environ.setdefault("TABPFN_NO_BROWSER", "1")  # nunca abrir el navegador
        from tabpfn import browser_auth
        from tabpfn.constants import ModelVersion
        from tabpfn.finetuning import FinetunedTabPFNClassifier

        browser_auth.save_token = _no_guardar_clave
        self.ajustador_ = FinetunedTabPFNClassifier(
            device=self.dispositivo or _dispositivo_local(),
            model_version=ModelVersion(self.version),
            eval_metric=self.metrica,
            random_state=self.random_state,
        )
        with _sin_modelo_compartido(), warnings.catch_warnings():
            # Sin `output_dir` no guarda puntos de control: es lo buscado
            warnings.filterwarnings("ignore", message="`output_dir` is not set")
            self.ajustador_.fit(X, y)
        inferencia = self.ajustador_.finetuned_inference_classifier_
        self.classes_ = np.asarray(inferencia.classes_)
        return self

    def predict_proba(self, X):
        """Probabilidades por clase, en el orden de `classes_`."""
        return self.ajustador_.predict_proba(X)

    def predict(self, X):
        """Clase de máxima probabilidad."""
        return self.classes_[np.asarray(self.predict_proba(X)).argmax(axis=1)]


# La entrada del registro: renormaliza en doble precisión y memoriza como `tabpfn`
clasificador = Fundacional("fkdc.tabpfn_ajustado:TabPFNAjustado")
