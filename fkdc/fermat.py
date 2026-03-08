from functools import partial

import numpy as np
from joblib import Memory
from scipy.sparse.csgraph import csgraph_from_dense, shortest_path
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.base import BaseEstimator, ClassifierMixin, DensityMixin
from sklearn.neighbors import KernelDensity, KNeighborsClassifier
from sklearn.utils.multiclass import unique_labels

from fkdc import dir_cache

# Ignorar advertencias para np.log(0) y similares
np.seterr(divide="ignore", invalid="ignore")

memoria = Memory(dir_cache, verbose=0)

MIN_PUNTAJE_LOG = np.log(1e-323)  # Para numpy, 1e-324 == 0 y 1e-323 != 0
MAX_PUNTAJE_LOG = np.log(np.finfo("float64").max)


@memoria.cache
def euclidiana(X, Y=None):
    """Matriz de distancias euclidianas."""
    if Y is None:
        return squareform(pdist(X, metric="euclidean"))
    else:
        return cdist(X, Y, metric="euclidean")


@memoria.cache
def fermat_muestral(Q, alpha=1, valor_relleno=np.inf):
    """Distancia de Fermat muestral sobre el grafo completo de Q."""
    adyacencias = np.ma.masked_array(
        euclidiana(Q) ** alpha,
        np.diag([True] * Q.shape[0]),
        fill_value=valor_relleno,
    )
    return shortest_path(csgraph_from_dense(adyacencias, valor_relleno), directed=False)


class DistanciaFermatMuestral:
    """Distancia de Fermat muestral con soporte para grupos."""

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

    def _distancia_muestral(self, X):
        """Distancia de cada punto en X a cada punto en Q, vía la muestra."""
        distancias_muestrales = -np.ones((X.shape[0], self.N))

        for lbl in self.etiquetas:
            mascara_grupo = self.grupos == lbl
            a_Q_lbl = euclidiana(X, self.Q[mascara_grupo]) ** self.alpha
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
        euc_XY = euclidiana(X, Y)
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


class ClasificadorFermatKVecinos(ClassifierMixin, BaseEstimator):
    """Clasificador K-vecinos con distancia de Fermat."""

    def __init__(self, n_neighbors=5, alpha=1, weights="uniform", n_jobs=-1):
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.weights = weights
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        self.distance_ = DistanciaFermatMuestral(Q=X, alpha=self.alpha, groups=y)
        self.classifier_ = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric="precomputed",
            n_jobs=self.n_jobs,
        )
        self.classifier_.fit(self.distance_(X), y)
        return self

    def predict(self, X):
        return self.classifier_.predict(self.distance_(X))

    def predict_proba(self, X):
        return self.classifier_.predict_proba(self.distance_(X))


class EDKFermat(BaseEstimator, DensityMixin):
    """Estimador de densidad kernel con distancia de Fermat."""

    def __init__(self, alpha: float = 1, bandwidth: float = 1, d: int = -1):
        self.bandwidth = bandwidth
        self.alpha = alpha
        self.d = d  # TODO: ¿Evitar completamente? Quitando el h^-d del score?

    def fit(self, X):
        self.distance_ = DistanciaFermatMuestral(Q=X, alpha=self.alpha)
        if self.d == -1:
            self.d = self.distance_.D
        return self

    def score_samples(self, X=None, log=True):
        if X is None:
            distancias = self.distance_.A[0]
        else:
            distancias = self.distance_(X)
        puntaje = np.exp(-0.5 * (distancias / self.bandwidth) ** 2).sum(1)
        if log:
            return (
                -np.log(self.distance_.N)
                - self.d * np.log(self.bandwidth)
                - self.d / 2 * np.log(2 * np.pi)
                + np.maximum(np.log(puntaje), MIN_PUNTAJE_LOG)
            )
        else:
            return (
                self.distance_.N**-1
                * (self.bandwidth**-self.d)
                * (2 * np.pi) ** (-self.d / 2)
                * puntaje
            )

    def score(self, X=None, y=None):
        return self.score_samples(X).sum()


class ClasificadorDensidadKernel(ClassifierMixin, BaseEstimator):
    """Clasificador por densidad kernel (Bayes ingenuo con KDE)."""

    def __init__(self, bandwidth=1.0, metric="euclidean", alpha=1.0):
        self.bandwidth = bandwidth
        self.metric = metric
        self.alpha = alpha

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        conjuntos_entrenamiento = [X[y == yi] for yi in self.classes_]
        if self.metric == "fermat":
            fabrica_densidad = partial(EDKFermat, alpha=self.alpha)
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
        logpuntajes = np.array([modelo.score_samples(X) for modelo in self.models_]).T
        logprobs = (logpuntajes + self.logpriors_).clip(
            MIN_PUNTAJE_LOG, MAX_PUNTAJE_LOG
        )
        # Tomo factor común de la máxima logprob para evitar problemas numéricos
        deltas = logprobs - logprobs.max(axis=1, keepdims=True)
        resultado = np.exp(deltas)
        return resultado / resultado.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]
