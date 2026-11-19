from functools import partial

import numpy as np
from joblib import Memory
from scipy.sparse.csgraph import csgraph_from_dense, shortest_path
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.base import BaseEstimator, ClassifierMixin, DensityMixin
from sklearn.neighbors import KernelDensity, KNeighborsClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from fkdc import dir_cache

# Con `dir_cache` en None (biblioteca instalada, sin `FKDC_CACHE`) no memoriza en disco
memoria = Memory(dir_cache, verbose=0)

MIN_PUNTAJE_LOG = np.log(1e-323)  # Para numpy, 1e-324 == 0 y 1e-323 != 0
MAX_PUNTAJE_LOG = np.log(np.finfo("float64").max)


def _validar_prediccion(estimador, X):
    """Valida X contra un estimador ajustado, con los mensajes de scikit-learn."""
    check_is_fitted(estimador)
    X = check_array(X)
    if X.shape[1] != estimador.n_features_in_:
        raise ValueError(
            f"X has {X.shape[1]} features, but {type(estimador).__name__} "
            f"is expecting {estimador.n_features_in_} features as input"
        )
    return X


@memoria.cache
def euclidean(X, Y=None):
    """Matriz de distancias euclideans."""
    if Y is None:
        return squareform(pdist(X, metric="euclidean"))
    else:
        return cdist(X, Y, metric="euclidean")


@memoria.cache
def fermat_muestral(Q, alpha=1, valor_relleno=np.inf):
    """Distancia de Fermat muestral sobre el grafo completo de Q."""
    adyacencias = np.ma.masked_array(
        euclidean(Q) ** alpha,
        np.diag([True] * Q.shape[0]),
        fill_value=valor_relleno,
    )
    return shortest_path(csgraph_from_dense(adyacencias, valor_relleno), directed=False)


class SampleFermatDistance:
    """Distancia de Fermat muestral con soporte para grupos.

    Al construirse, calcula la distancia de Fermat muestral entre las observaciones
    de Q (dentro de cada grupo, si los hay). Invocada como `distancia(X)`, devuelve
    la distancia de cada fila de X a cada observación de Q, de forma
    `(len(X), len(Q))`; como `distancia(X, Y)`, la distancia entre cada fila de X y
    cada fila de Y.

    Parámetros
    ----------
    Q : ndarray de forma (N, D)
        Muestra que define la distancia.
    alpha : float, por defecto 1
        Exponente aplicado a las distancias euclídeas entre observaciones.
    grupos : array de largo N o None, por defecto None
        Etiqueta de grupo de cada observación; las trayectorias no cruzan de un
        grupo a otro. Con None, un solo grupo.
    """

    def __init__(self, Q, alpha: float = 1, grupos=None):
        self.Q = Q
        self.N, self.D = Q.shape
        self.grupos = np.array(np.zeros(self.N) if grupos is None else grupos)
        self.etiquetas = unique_labels(self.grupos)
        if len(self.grupos) != self.N:
            raise ValueError(
                "`grupos` debe ser None, o de igual número que las filas de Q"
            )
        self.alpha = alpha
        self.A = {
            lbl: fermat_muestral(Q[self.grupos == lbl], alpha) for lbl in self.etiquetas
        }

    def __setstate__(self, state):
        # Compatibilidad con pickles anteriores a la traducción al español
        if "groups" in state and "grupos" not in state:
            state["grupos"] = state.pop("groups")
        if "labels" in state and "etiquetas" not in state:
            state["etiquetas"] = state.pop("labels")
        self.__dict__.update(state)

    def _distancia_muestral(self, X):
        """Distancia de cada punto en X a cada punto en Q, vía la muestra."""
        distancias_muestrales = -np.ones((X.shape[0], self.N))

        for lbl in self.etiquetas:
            mascara_grupo = self.grupos == lbl
            a_Q_lbl = euclidean(X, self.Q[mascara_grupo]) ** self.alpha
            for i in range(len(X)):
                distancias_muestrales[i, mascara_grupo] = np.min(
                    a_Q_lbl[i].T + self.A[lbl], axis=1
                )
        assert np.all(distancias_muestrales >= 0)
        return distancias_muestrales

    def _distancia(self, X, Y):
        """Distancia de Fermat muestral entre cada par (x, y) en X × Y."""
        dfs_XQ = self._distancia_muestral(X)
        dfs_YQ = self._distancia_muestral(Y)
        euc_XY = euclidean(X, Y)
        nX, nY = X.shape[0], Y.shape[0]
        distancias = -np.ones((nX, nY))
        for i in range(nX):
            for j in range(nY):
                bypass_dfs = euc_XY[i, j] ** self.alpha
                cruce_dfs = np.min(dfs_XQ[i] + dfs_YQ[j])
                distancias[i, j] = np.min([bypass_dfs, cruce_dfs])
        assert np.all(distancias >= 0)
        return distancias

    def __call__(self, X, Y=None):
        if X.ndim == 1:  # en caso de que lo llamen con una sola observación en X
            X = X.reshape(1, self.D)
        if Y is None:
            return self._distancia_muestral(X)
        else:
            if Y.ndim == 1:  # en caso de que lo llamen con una sola observación en Y
                Y = Y.reshape(1, self.D)
            return self._distancia(X, Y)


class FermatKNeighborsClassifier(ClassifierMixin, BaseEstimator):
    """Clasificador de k vecinos más cercanos con distancia de Fermat muestral.

    Las distancias se calculan con `SampleFermatDistance` sobre la muestra de
    entrenamiento de cada clase por separado y se pasan precalculadas a
    `sklearn.neighbors.KNeighborsClassifier`.

    Parámetros
    ----------
    n_neighbors : int, por defecto 5
        Cantidad de vecinos.
    alpha : float, por defecto 1
        Exponente de la distancia de Fermat muestral; con `alpha=1` coincide con la
        euclídea.
    weights : {"uniform", "distance"} o invocable, por defecto "uniform"
        Pesos de los vecinos, como en `KNeighborsClassifier`.
    n_jobs : int, por defecto -1
        Procesos para la búsqueda de vecinos, como en `KNeighborsClassifier`.

    Atributos
    ---------
    classes_ : ndarray
        Etiquetas de clase.
    n_features_in_ : int
        Cantidad de variables vistas en `fit`.
    distance_ : SampleFermatDistance
        Distancia ajustada a la muestra de entrenamiento.
    classifier_ : KNeighborsClassifier
        Clasificador subyacente, ajustado sobre las distancias precalculadas.
    """

    def __init__(self, n_neighbors=5, alpha=1, weights="uniform", n_jobs=-1):
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.weights = weights
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        self.distance_ = SampleFermatDistance(Q=X, alpha=self.alpha, grupos=y)
        self.classifier_ = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric="precomputed",
            n_jobs=self.n_jobs,
        )
        self.classifier_.fit(self.distance_(X), y)
        return self

    def predict(self, X):
        X = _validar_prediccion(self, X)
        return self.classifier_.predict(self.distance_(X))

    def predict_proba(self, X):
        X = _validar_prediccion(self, X)
        return self.classifier_.predict_proba(self.distance_(X))


class FermatKDE(DensityMixin, BaseEstimator):
    """Estimador de densidad por núcleos con distancia de Fermat muestral.

    Núcleo gaussiano evaluado sobre la distancia de Fermat muestral a cada
    observación de entrenamiento, con la normalización de la KDE euclídea en
    dimensión `d`.

    Parámetros
    ----------
    alpha : float, por defecto 1
        Exponente de la distancia de Fermat muestral.
    bandwidth : float, por defecto 1
        Ancho de banda del núcleo.
    d : int, por defecto -1
        Dimensión usada en la normalización; con -1, la cantidad de variables de X.

    Atributos
    ---------
    n_features_in_ : int
        Cantidad de variables vistas en `fit`.
    d_ : int
        Dimensión efectiva de la normalización.
    distance_ : SampleFermatDistance
        Distancia ajustada a la muestra de entrenamiento.
    """

    def __init__(self, alpha: float = 1, bandwidth: float = 1, d: int = -1):
        self.bandwidth = bandwidth
        self.alpha = alpha
        self.d = d  # TODO: ¿Evitar completamente? Quitando el h^-d del score?

    def fit(self, X, y=None):
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        self.distance_ = SampleFermatDistance(Q=X, alpha=self.alpha)
        # `d` queda como lo pasó el usuario; la dimensión efectiva va en `d_`
        self.d_ = self.distance_.D if self.d == -1 else self.d
        return self

    def __setstate__(self, state):
        # Los pickles anteriores a `d_` guardaban la dimensión efectiva en `d`
        if "distance_" in state and "d_" not in state:
            state["d_"] = state["d"]
        super().__setstate__(state)

    def score_samples(self, X=None, log=True):
        """Log-densidad (o densidad, con `log=False`) en cada fila de X.

        Sin X, evalúa en la propia muestra de entrenamiento.
        """
        if X is None:
            check_is_fitted(self)
            distancias = self.distance_.A[0]
        else:
            distancias = self.distance_(_validar_prediccion(self, X))
        puntaje = np.exp(-0.5 * (distancias / self.bandwidth) ** 2).sum(1)
        if log:
            # En la cola profunda `puntaje` es 0: log(0) = -inf, que recorta el máximo
            with np.errstate(divide="ignore", invalid="ignore"):
                return (
                    -np.log(self.distance_.N)
                    - self.d_ * np.log(self.bandwidth)
                    - self.d_ / 2 * np.log(2 * np.pi)
                    + np.maximum(np.log(puntaje), MIN_PUNTAJE_LOG)
                )
        else:
            return (
                self.distance_.N**-1
                * (self.bandwidth**-self.d_)
                * (2 * np.pi) ** (-self.d_ / 2)
                * puntaje
            )

    def score(self, X=None, y=None):
        """Log-verosimilitud total de X (o de la muestra de entrenamiento)."""
        return self.score_samples(X).sum()


class DistanciaEuclidea:
    """Distancia euclídea con la interfaz mínima de `SampleFermatDistance`.

    Usa `euclidean.func`, el cómputo exacto con `cdist`/`pdist` sin el caché en disco:
    los valores son bit a bit los que se cachean y `dir_cache` no crece.
    """

    def __init__(self, Q):
        self.Q = Q
        self.N, self.D = Q.shape

    @property
    def A(self):
        # Solo para `score_samples(X=None)`: distancias de Q contra sí misma
        return {0: euclidean.func(self.Q)}

    def __call__(self, X, Y=None):
        if X.ndim == 1:  # en caso de que lo llamen con una sola observación
            X = X.reshape(1, self.D)
        return euclidean.func(X, self.Q if Y is None else Y)


class EuclideanKDE(FermatKDE):
    """Estimador de densidad por núcleos euclídeo exacto (fuerza bruta).

    Alternativa a `sklearn.neighbors.KernelDensity`, cuyo recorrido del árbol
    (KDTree/BallTree) devuelve log-densidades erróneas en la cola profunda: puntos
    dentro de la caja que envuelve a una clase pero lejos de todas sus observaciones.
    Hereda `score_samples` de `FermatKDE` (la suma exacta de núcleos sobre la matriz de
    distancias, con la misma normalización y el mismo recorte), de modo que
    `KDClassifier(metric="euclidea_exacta")` coincide bit a bit con
    `KDClassifier(metric="fermat", alpha=1)`.

    Parámetros
    ----------
    bandwidth : float, por defecto 1
        Ancho de banda del núcleo.
    d : int, por defecto -1
        Dimensión usada en la normalización; con -1, la cantidad de variables de X.
    """

    def __init__(self, bandwidth: float = 1, d: int = -1):
        # Sin `alpha`: no es parámetro de este estimador
        self.bandwidth = bandwidth
        self.d = d

    def fit(self, X, y=None):
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        self.distance_ = DistanciaEuclidea(X)
        self.d_ = self.distance_.D if self.d == -1 else self.d
        return self


class KDClassifier(ClassifierMixin, BaseEstimator):
    """Clasificador por densidad de núcleos: regla de Bayes con una KDE por clase.

    Estima la densidad de cada clase con una KDE y asigna la clase de máxima
    probabilidad a posteriori, con las proporciones de entrenamiento como previas.

    Parámetros
    ----------
    bandwidth : float, por defecto 1.0
        Ancho de banda común a todas las clases.
    metric : str, por defecto "euclidean"
        `"fermat"` usa `FermatKDE` con `alpha`; `"euclidea_exacta"` usa
        `EuclideanKDE`; cualquier otro valor se pasa a
        `sklearn.neighbors.KernelDensity`, sin `metric_params`.
    alpha : float, por defecto 1.0
        Exponente de la distancia de Fermat; solo se usa con `metric="fermat"`.

    Atributos
    ---------
    classes_ : ndarray
        Etiquetas de clase.
    n_features_in_ : int
        Cantidad de variables vistas en `fit`.
    models_ : list
        Una KDE ajustada por clase, en el orden de `classes_`.
    logpriors_ : list of float
        Logaritmo de la proporción de cada clase en el entrenamiento.
    """

    def __init__(self, bandwidth=1.0, metric="euclidean", alpha=1.0):
        self.bandwidth = bandwidth
        self.metric = metric
        self.alpha = alpha

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        conjuntos_entrenamiento = [X[y == yi] for yi in self.classes_]
        if self.metric == "fermat":
            fabrica_densidad = partial(FermatKDE, alpha=self.alpha)
        elif self.metric == "euclidea_exacta":
            # KDE euclídea exacta; "euclidean" sigue usando `KernelDensity`, como en la
            # tesis, para que sus números y figuras se reproduzcan tal cual
            fabrica_densidad = EuclideanKDE
        else:
            fabrica_densidad = partial(KernelDensity, metric=self.metric)
        self.models_ = [
            fabrica_densidad(bandwidth=self.bandwidth).fit(Xi)
            for Xi in conjuntos_entrenamiento
        ]
        self.logpriors_ = [
            np.log(Xi.shape[0] / X.shape[0]) for Xi in conjuntos_entrenamiento
        ]
        return self

    def predict_proba(self, X):
        X = _validar_prediccion(self, X)
        # Las log-densidades pueden ser -inf (cola profunda); el recorte las acota
        with np.errstate(divide="ignore", invalid="ignore"):
            logpuntajes = np.array(
                [modelo.score_samples(X) for modelo in self.models_]
            ).T
            logprobs = (logpuntajes + self.logpriors_).clip(
                MIN_PUNTAJE_LOG, MAX_PUNTAJE_LOG
            )
            # Tomo factor común de la máxima logprob para evitar problemas numéricos
            deltas = logprobs - logprobs.max(axis=1, keepdims=True)
            resultado = np.exp(deltas)
            return resultado / resultado.sum(axis=1, keepdims=True)

    def predict(self, X):
        probas = self.predict_proba(X)  # antes que `classes_`: valida que esté ajustado
        return self.classes_[np.argmax(probas, 1)]
