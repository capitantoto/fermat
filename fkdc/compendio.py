"""
Compendio de clasificadores clásicos como líneas de base adicionales.

No forman parte de los resultados de la tesis: tienen registro propio
(`clasificadores`, `grillas`) y resultados propios en `sandbox/fundacionales/infos`
(ver `scripts/compendio.py`), de modo que `config.clasificadores` y las figuras de
`viz.py` no cambian. Las tareas son las de la tesis: mismos datasets, semillas,
particiones, validación cruzada y reajuste parsimonioso.

Reajuste parsimonioso: cada hiperparámetro de estas grillas tiene en
`utils.reajustar_parsimoniosamente` una dirección de parsimonia propia, con un nombre
que no usa ningún clasificador de la tesis. Por eso los estimadores de sklearn cuyos
hiperparámetros comparten nombre con los de la tesis (p. ej. el `alpha` de MLP, una
penalización L2, frente al exponente `alpha` de fkdc) van dentro de un `Pipeline`, y
el nombre del paso prefija el del hiperparámetro (`mlp__alpha`).

Grillas dependientes del dataset: `grilla(nombre, n_muestras, n_atributos)` descarta
los vecinos que no caben en un pliego de entrenamiento (como
`config.hacer_configuraciones` con `n_neighbors`) y las dimensiones mayores que la del
espacio ambiente.

Isomap y NCA dominan el costo de sus pipelines y no dependen de los hiperparámetros
del clasificador que les sigue: `IsomapMemorizado` y `NCAMemorizada` memorizan en el
proceso su ajuste y sus transformaciones, de modo que la búsqueda los ajusta una vez
por pliego y combinación de sus propios hiperparámetros. `ParameterGrid` recorre las
claves en orden alfabético, con la primera como la de variación más lenta, así que
los pasos costosos llevan nombres que van antes que los de los pasos siguientes
(`isomap`, `nca` antes que `vecinos`) y basta una memoria chica.
"""

import hashlib
from collections import OrderedDict
from math import floor

import numpy as np
from scipy.special import logsumexp
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    DensityMixin,
    TransformerMixin,
    clone,
)
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.manifold import Isomap
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import (
    KNeighborsClassifier,
    NearestNeighbors,
    NeighborhoodComponentsAnalysis,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.semi_supervised import LabelSpreading
from sklearn.utils import Bunch
from sklearn.utils.multiclass import unique_labels

from fkdc import config
from fkdc.fermat import KDClassifier, euclidean

semilla = config.semilla_principal


def _probas_bayes(log_densidades, logpriors):
    """Regla de Bayes: normaliza log p(x|k) + log π_k con logsumexp, sin recortes."""
    logprobs = np.asarray(log_densidades).T + logpriors
    return np.exp(logprobs - logsumexp(logprobs, axis=1, keepdims=True))


class KDEAdaptativa(DensityMixin, BaseEstimator):
    """KDE gaussiana con ancho de banda adaptativo por punto muestral (Abramson, 1982).

    Cada observación X_i lleva su propio ancho de banda h·λ_i, con
    λ_i = (f̃(X_i) / g)^(-sensibilidad), donde f̃ es la KDE piloto de ancho fijo h
    evaluada en la muestra y g, la media geométrica de los f̃(X_i). Con
    `sensibilidad=0.5` es la ley de la raíz cuadrada de Abramson; con
    `sensibilidad=0`, la KDE de ancho fijo. Como en `FermatKDE`, la normalización
    usa la dimensión del espacio ambiente.

    Las log-densidades se calculan exactamente, con logsumexp sobre la matriz de
    distancias de `fermat.euclidean` (sin su caché en disco), y no con
    `sklearn.neighbors.KernelDensity`, cuyo recorrido del árbol devuelve
    log-densidades erróneas en las colas profundas (ver
    `scripts/reproduce_sklearn_kde_bug.py`).
    """

    def __init__(self, bandwidth: float = 1.0, sensibilidad: float = 0.5):
        self.bandwidth = bandwidth
        self.sensibilidad = sensibilidad

    def fit(self, X, y=None):
        self.X_ = np.asarray(X, dtype=float)
        self.N_, self.D_ = self.X_.shape
        log_piloto = self._log_densidad(euclidean.func(self.X_), np.zeros(self.N_))
        # log λ_i = -sensibilidad · (log f̃(X_i) - log g), con log g = media de log f̃
        self.log_lambda_ = -self.sensibilidad * (log_piloto - log_piloto.mean())
        return self

    def _log_densidad(self, distancias, log_lambda):
        """log f(x) para cada fila de `distancias` (x × muestra), con anchos h·λ_i."""
        log_anchos = np.log(self.bandwidth) + log_lambda
        log_nucleos = (
            -0.5 * (distancias / np.exp(log_anchos)) ** 2
            - self.D_ * log_anchos
            - self.D_ / 2 * np.log(2 * np.pi)
        )
        return logsumexp(log_nucleos, axis=1) - np.log(self.N_)

    def score_samples(self, X):
        distancias = euclidean.func(np.asarray(X, dtype=float), self.X_)
        return self._log_densidad(distancias, self.log_lambda_)

    def score(self, X, y=None):
        return self.score_samples(X).sum()


class KDClassifierAdaptativo(KDClassifier):
    """`KDClassifier` con una `KDEAdaptativa` (Abramson) por clase.

    Misma regla de Bayes, con las proporciones de entrenamiento como previas, pero
    normaliza con logsumexp sin recortar las log-densidades a
    [log(1e-323), log(máx)]: como son exactas y finitas, el recorte solo igualaría
    a las clases en las colas profundas. Con `sensibilidad=0` es el KDC euclídeo de
    ancho fijo, calculado exactamente.
    """

    def __init__(self, bandwidth: float = 1.0, sensibilidad: float = 0.5):
        self.bandwidth = bandwidth
        self.sensibilidad = sensibilidad

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        conjuntos = [X[y == yi] for yi in self.classes_]
        self.models_ = [
            KDEAdaptativa(self.bandwidth, self.sensibilidad).fit(Xi) for Xi in conjuntos
        ]
        self.logpriors_ = np.log([len(Xi) / len(X) for Xi in conjuntos])
        return self

    def predict_proba(self, X):
        log_densidades = [modelo.score_samples(X) for modelo in self.models_]
        return _probas_bayes(log_densidades, self.logpriors_)


class ClasificadorMezclas(ClassifierMixin, BaseEstimator):
    """Una mezcla gaussiana (`GaussianMixture`) por clase y regla de Bayes.

    Las previas son las proporciones de entrenamiento. Si una clase tiene menos
    observaciones que `n_components`, su mezcla usa tantas componentes como
    observaciones.
    """

    def __init__(
        self,
        n_components: int = 1,
        covariance_type: str = "full",
        reg_covar: float = 1e-6,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        conjuntos = [X[y == yi] for yi in self.classes_]
        self.models_ = [
            GaussianMixture(
                n_components=min(self.n_components, len(Xi)),
                covariance_type=self.covariance_type,
                reg_covar=self.reg_covar,
                random_state=self.random_state,
            ).fit(Xi)
            for Xi in conjuntos
        ]
        self.logpriors_ = np.log([len(Xi) / len(X) for Xi in conjuntos])
        return self

    def predict_proba(self, X):
        log_densidades = [modelo.score_samples(X) for modelo in self.models_]
        return _probas_bayes(log_densidades, self.logpriors_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class ClasificadorProcesoGaussiano(ClassifierMixin, BaseEstimator):
    """`GaussianProcessClassifier` con núcleo C·RBF isótropo, sin grilla.

    Los hiperparámetros del núcleo (C y la escala ℓ) se eligen maximizando la
    verosimilitud marginal (aproximación de Laplace; uno contra el resto si hay más
    de dos clases). El optimizador parte de ℓ igual a la mediana de las distancias
    entre observaciones de entrenamiento: desde ℓ = 1 (el valor por omisión), con
    distancias del orden de 10³ como en `mnist`, el núcleo es casi la identidad, el
    gradiente se anula y el clasificador queda en las proporciones de clase.
    """

    def __init__(self, random_state: int | None = None):
        self.random_state = random_state

    def fit(self, X, y):
        escala = np.median(euclidean.func(np.asarray(X, dtype=float)))
        nucleo = ConstantKernel(1.0) * RBF(length_scale=escala)
        self.estimador_ = GaussianProcessClassifier(
            kernel=nucleo, random_state=self.random_state
        ).fit(X, y)
        self.classes_ = self.estimador_.classes_
        return self

    def predict_proba(self, X):
        return self.estimador_.predict_proba(X)

    def predict(self, X):
        return self.estimador_.predict(X)


_memoria: OrderedDict = OrderedDict()
tamano_memoria = 32  # entradas (ajustes y transformaciones) por proceso


def _huella(*arrays) -> str:
    """Huella del contenido, la forma y el tipo de cada array (o de su ausencia)."""
    huella = hashlib.blake2b(digest_size=16)
    for array in arrays:
        if array is None:
            huella.update(b"None")
            continue
        array = np.asarray(array)
        if array.dtype == object:
            array = array.astype(str)
        array = np.ascontiguousarray(array)
        huella.update(f"{array.shape}{array.dtype.str}".encode())
        huella.update(array.tobytes())
    return huella.hexdigest()


def _recordar(clave, calcular):
    """Valor memorizado de `clave`; si falta, lo calcula y lo guarda (LRU)."""
    if clave in _memoria:
        _memoria.move_to_end(clave)
        return _memoria[clave]
    valor = _memoria[clave] = calcular()
    if len(_memoria) > tamano_memoria:
        _memoria.popitem(last=False)
    return valor


class _Memorizado:
    """Mixin que memoriza, dentro del proceso, el ajuste y las transformaciones.

    La clave del ajuste son la clase, los hiperparámetros y el contenido de (X, y);
    la de cada transformación, la del ajuste y el contenido de X. Las
    transformaciones memorizadas se devuelven de solo lectura, para que nadie las
    modifique en el lugar.
    """

    def fit(self, X, y=None):
        clave = (
            "ajuste",
            type(self).__name__,
            repr(sorted(self.get_params().items())),
            _huella(X, y),
        )

        def ajustar():
            super(_Memorizado, self).fit(X, y)
            return {k: v for k, v in vars(self).items() if k.endswith("_")}

        vars(self).update(_recordar(clave, ajustar))
        self.clave_ajuste_ = clave
        return self

    def transform(self, X):
        def transformar():
            resultado = np.array(super(_Memorizado, self).transform(X))
            resultado.flags.writeable = False
            return resultado

        return _recordar(
            ("transformacion", self.clave_ajuste_, _huella(X)), transformar
        )

    def fit_transform(self, X, y=None, **kwargs):
        return self.fit(X, y).transform(X)


class IsomapMemorizado(_Memorizado, Isomap):
    """`Isomap` que memoriza su ajuste y sus transformaciones en el proceso.

    `transform` es la extensión fuera de la muestra de Isomap: une cada punto nuevo
    a sus `n_neighbors` vecinos de entrenamiento y completa sus distancias
    geodésicas por el grafo de entrenamiento. También se aplica a los puntos de
    entrenamiento, en lugar de `embedding_`, para que entrenamiento y evaluación
    pasen por el mismo mapa.
    """


class NCAMemorizada(_Memorizado, NeighborhoodComponentsAnalysis):
    """`NeighborhoodComponentsAnalysis` que memoriza su ajuste en el proceso."""


class Recorte(TransformerMixin, BaseEstimator):
    """Primeras `n_componentes` columnas de X.

    Tras Isomap equivale a ajustarlo con `n_components = n_componentes`: sus
    coordenadas son los autovectores principales del núcleo geodésico, ordenados por
    autovalor (con `eigen_solver="dense"`, el mismo cálculo). Así, un único ajuste de
    Isomap por pliego sirve a toda la grilla de dimensiones.
    """

    def __init__(self, n_componentes: int = 2):
        self.n_componentes = n_componentes

    def fit(self, X, y=None):
        if self.n_componentes > X.shape[1]:
            raise ValueError(f"n_componentes={self.n_componentes} > {X.shape[1]}")
        return self

    def transform(self, X):
        return X[:, : self.n_componentes]


class EscalaIsotropa(TransformerMixin, BaseEstimator):
    """Divide X por la mediana de la distancia de cada punto a su vecino más cercano.

    Es una homotecia: conserva los cocientes de distancias y el orden de los
    vecinos. Pone en escala el softmax de NCA, que sobre los datos crudos (p. ej.
    `mnist` o `digitos`, con distancias del orden de 10³ y 10¹) queda saturado en el
    vecino más cercano y deja al optimizador clavado en la identidad.
    """

    def fit(self, X, y=None):
        distancias, _ = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X)
        positivas = distancias[:, 1][distancias[:, 1] > 0]
        self.escala_ = np.median(positivas) if positivas.size else 1.0
        return self

    def transform(self, X):
        return X / self.escala_


def _isomap():
    return IsomapMemorizado(eigen_solver="dense")


clasificadores = Bunch(
    qda=QuadraticDiscriminantAnalysis(),
    lda=LinearDiscriminantAnalysis(solver="lsqr"),
    gmm=ClasificadorMezclas(random_state=semilla),
    kdc_abramson=KDClassifierAdaptativo(),
    isomap_kn=Pipeline(
        [
            ("isomap", _isomap()),
            ("recorte", Recorte()),
            ("vecinos", KNeighborsClassifier()),
        ]
    ),
    isomap_kdc=Pipeline(
        [
            ("isomap", _isomap()),
            ("recorte", Recorte()),
            ("kdc", KDClassifierAdaptativo(sensibilidad=0)),
        ]
    ),
    nca_kn=Pipeline(
        [
            ("escala", EscalaIsotropa()),
            ("nca", NCAMemorizada(random_state=semilla)),
            ("vecinos", KNeighborsClassifier()),
        ]
    ),
    label_spreading=Pipeline(
        [("ls", LabelSpreading(kernel="knn", max_iter=1000))],
    ),
    gpc=ClasificadorProcesoGaussiano(random_state=semilla),
    rf=Pipeline(
        [("bosque", RandomForestClassifier(n_estimators=500, random_state=semilla))]
    ),
    et=Pipeline(
        [("bosque", ExtraTreesClassifier(n_estimators=500, random_state=semilla))]
    ),
    mlp=Pipeline(
        [("mlp", MLPClassifier(solver="lbfgs", max_iter=5000, random_state=semilla))]
    ),
)

vecinos_ralos = config.espacio_kn["n_neighbors"][::2]  # 1, 3, 6, …, 268, 499
dimensiones = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
dimensiones_nca = [1, 2, 4, 8, 16, 32]  # más D, el espacio ambiente completo
vecinos_isomap = [5, 10, 20, 40]
espacio_bosque = {
    "bosque__max_features": [0.1, 0.3, 0.6, 1.0],
    "bosque__min_samples_leaf": [1, 2, 4, 8],
}
grillas = {
    # 37 valores, como los 36 de lr; Σ_k ← (1 - r)Σ_k + r·I
    "qda": {"reg_param": np.logspace(-6, 0, 37)},
    # 37 valores; contrae la correlación común hacia la identidad
    "lda": {"shrinkage": np.logspace(-6, 0, 37)},
    # 6 × 2 × 4 = 48; "diag" y "full", cuyo orden alfabético (el que usa R1SD) es
    # también el de complejidad
    "gmm": {
        "n_components": [1, 2, 3, 4, 6, 8],
        "covariance_type": ["diag", "full"],
        "reg_covar": np.logspace(-6, 0, 4),
    },
    # La misma grilla de h que kdc (136 valores)
    "kdc_abramson": {"bandwidth": config.grillas["kdc"]["bandwidth"]},
    # 4 × ≤10 dims × 10 k × 2 pesos ≤ 800, entre los 494 de fkn y los 585 de fkdc
    "isomap_kn": {
        "isomap__n_neighbors": vecinos_isomap,
        "recorte__n_componentes": dimensiones,
        "vecinos__n_neighbors": vecinos_ralos,
        "vecinos__weights": ["uniform", "distance"],
    },
    # 4 × ≤10 dims × los 45 h de fkdc ≤ 1800 (un ajuste de Isomap por pliego y k)
    "isomap_kdc": {
        "isomap__n_neighbors": vecinos_isomap,
        "recorte__n_componentes": dimensiones,
        "kdc__bandwidth": config.grillas["fkdc"]["bandwidth"],
    },
    # ≤7 dims × 10 k × 2 pesos ≤ 140, como los 38 de kn por cada dimensión
    "nca_kn": {
        "nca__n_components": dimensiones_nca,
        "vecinos__n_neighbors": vecinos_ralos,
        "vecinos__weights": ["uniform", "distance"],
    },
    # Los 19 k de kn × 7 α = 133
    "label_spreading": {
        "ls__n_neighbors": config.espacio_kn["n_neighbors"],
        "ls__alpha": [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
    },
    # Sin grilla: los hiperparámetros del núcleo, por verosimilitud marginal
    "gpc": {},
    # 4 × 4 = 16, como los 12 de gbt; 500 árboles fijos
    "rf": espacio_bosque,
    "et": espacio_bosque,
    # 3 anchos de una capa oculta × 7 penalizaciones L2 = 21
    "mlp": {
        "mlp__hidden_layer_sizes": [16, 64, 256],
        "mlp__alpha": np.logspace(-4, 2, 7),
    },
}


def grilla(
    nombre: str,
    n_muestras: int,
    n_atributos: int,
    cv: int = config.cv,
    split_evaluacion: float = config.split_evaluacion,
) -> dict:
    """Grilla de `nombre` para un dataset de `n_muestras` × `n_atributos`.

    Como `config.hacer_configuraciones`, descarta los números de vecinos que no
    caben en un pliego de entrenamiento; además descarta las dimensiones mayores que
    `n_atributos`, suma a NCA la dimensión completa y ajusta Isomap con la mayor
    dimensión de la grilla, que `Recorte` reduce a las demás.
    """
    n_muestras_ajuste = floor(n_muestras * (cv - 1) / cv * (1 - split_evaluacion))
    resultado = {}
    for parametro, valores in grillas[nombre].items():
        valores = list(valores)
        if parametro.endswith("n_neighbors"):
            valores = [k for k in valores if k < n_muestras_ajuste]
        elif parametro == "recorte__n_componentes":
            valores = [d for d in valores if d <= n_atributos]
            resultado["isomap__n_components"] = [max(valores)]
        elif parametro == "nca__n_components":
            valores = [d for d in valores if d < n_atributos] + [n_atributos]
        resultado[parametro] = valores
    return resultado


def instanciar(nombre: str):
    """Copia sin ajustar del clasificador `nombre`."""
    return clone(clasificadores[nombre])
