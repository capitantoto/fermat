from itertools import product
from numbers import Number

import numpy as np
from joblib import Memory
from scipy.sparse.csgraph import csgraph_from_dense, shortest_path
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.base import BaseEstimator, ClassifierMixin, DensityMixin
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.neighbors import KernelDensity, KNeighborsClassifier
from sklearn.utils.multiclass import unique_labels

# Ignore warnings for np.log(0) and siimilar
np.seterr(divide="ignore", invalid="ignore")

cachedir = "_cache"
memory = Memory(cachedir, verbose=0)


@memory.cache
def euclidean(X, Y=None):
    if Y is None:
        return squareform(pdist(X, metric="euclidean"))
    else:
        return cdist(X, Y, metric="euclidean")


@memory.cache
def sample_fermat(Q, alpha=1, fill_value=np.inf):
    adyacencias = np.ma.masked_array(
        euclidean(Q) ** alpha, np.diag([True] * Q.shape[0]), fill_value=fill_value
    )
    return shortest_path(csgraph_from_dense(adyacencias, fill_value), directed=False)


class Bundle(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __repr__(self):
        return "Bundle(" + ", ".join(f"{k}={v}" for k, v in self.items()) + ")"


def iqr(X):
    return np.percentile(X, 75) - np.percentile(X, 25)


def pilot_h(dists):
    return 0.9 * np.minimum(dists.std(), iqr(dists) / 1.34) * len(dists) ** (-1 / 5)
    # return np.mean(sample_fermat(X, alpha=alpha))


class FermatKDE(BaseEstimator, DensityMixin):
    # MAX_DIST = 38.6
    MIN_LOG_SCORE = -1e6

    def __init__(self, alpha: float = 1, bandwidth: float = 1, d: int = -1):
        self.bandwidth = bandwidth
        self.alpha = alpha
        self.d = d  # TODO: Evitar completamente? Quitando el h^-d del score?

    def fit(self, X):
        self.distance_ = SampleFermatDistance(Q=X, alpha=self.alpha)
        if self.d == -1:
            self.d = self.distance_.D
        return self

    def score_samples(self, X=None, log=True):
        if X is None:
            distances = self.distance_.A[0]
        else:
            distances = self.distance_(X)
        score = np.exp(-0.5 * (distances / self.bandwidth) ** 2).sum(1)
        if log:
            return (
                -np.log(self.distance_.N)
                - self.d * np.log(self.bandwidth)
                - self.d / 2 * np.log(2 * np.pi)
                + np.maximum(np.log(score), self.MIN_LOG_SCORE)
            )
        else:
            return (
                self.distance_.N**-1
                * (self.bandwidth**-self.d)
                * (2 * np.pi) ** (-self.d / 2)
                * score
            )

    def score(self, X=None):
        return self.score_samples(X).sum()


def lattice(a, b, step=1, dim=2, array=True):
    side = np.arange(a, b, step)
    if len(side) ** dim > 1e6:
        raise ValueError(
            f"Too many points ({len(side) ** dim:.2e} > 1e6). Try a bigger step or a smaller dim."
        )
    gen = product(*[side] * dim)
    return np.array([*gen]) if array else gen


class BaseKDClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = [
            KernelDensity(bandwidth=self.bandwidth, kernel="gaussian").fit(Xi)
            for Xi in training_sets
        ]
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0]) for Xi in training_sets]
        return self

    def predict_proba(self, X):
        logprobs = np.array([model.score_samples(X) for model in self.models_]).T
        result = np.exp(logprobs + self.logpriors_)
        return result / result.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]


class FermatKDClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, bandwidth: float = -1.0, alpha: float = 1.0):
        self.bandwidth = bandwidth
        self.alpha = alpha

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.bandwidths_ = self._pick_bandwidths(training_sets)

        self.models_ = [
            FermatKDE(bandwidth=hi, alpha=self.alpha).fit(Xi)
            for hi, Xi in zip(self.bandwidths_, training_sets)
        ]
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0]) for Xi in training_sets]
        return self

    def _pick_bandwidths(self, training_sets):
        if isinstance(self.bandwidth, Number):
            bandwidths = np.repeat(self.bandwidth, len(self.classes_))
        elif len(self.bandwidth) == len(self.classes_):
            bandwidths = np.array(self.bandwidth, dtype=float)
        else:
            raise ValueError(
                "bandwidth must be a scalar or a vector of exactly n_classes"
            )
        for i, bw in enumerate(bandwidths):
            Xi = training_sets[i]
            if bw == -1:  # automatic bandwidth selection
                pilot = pilot_h(sample_fermat(Xi, self.alpha))
                grid = {"bandwidth": pilot * np.logspace(-3, 3, 31)}
                cv = ShuffleSplit(n_splits=10, test_size=0.5)
                search = GridSearchCV(FermatKDE(alpha=self.alpha).fit(Xi), grid, cv=cv)
                search.fit(Xi)
                bandwidths[i] = search.best_params_["bandwidth"]
        return bandwidths

    def predict_proba(self, X):
        logprobs = np.array([model.score_samples(X) for model in self.models_]).T
        result = np.exp(logprobs + self.logpriors_)
        return result / result.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]


class SampleFermatDistance:
    def __init__(self, Q, alpha=1, groups=None):
        self.Q = Q
        self.N, self.D = Q.shape
        self.groups = np.array(np.zeros(self.N) if groups is None else groups)
        self.labels = unique_labels(self.groups)
        assert (
            len(self.groups) == self.N
        ), "`groups` debe ser None, o de la misma longitud que el número de filas de Q"
        self.alpha = alpha
        self.A = {
            lbl: sample_fermat(Q[self.groups == lbl], alpha) for lbl in self.labels
        }

    def _sample_distance(self, X):
        sample_distances = -np.ones((X.shape[0], self.N))

        for lbl in self.labels:
            group_mask = self.groups == lbl
            to_Q_lbl = euclidean(X, self.Q[group_mask]) ** self.alpha
            for i in range(len(X)):
                sample_distances[i, group_mask] = np.min(
                    to_Q_lbl[i].T + self.A[lbl], axis=1
                )
        assert np.all(sample_distances >= 0)
        return sample_distances

    def _distance(self, X, Y):
        sfd_XQ = self._sample_distance(X)
        sfd_YQ = self._sample_distance(Y)
        euc_XY = euclidean(X, Y)
        nX, nY = X.shape[0], Y.shape[0]
        distances = -np.ones((nX, nY))
        for i in range(nX):
            for j in range(nY):
                bypass_sfd = euc_XY[i, j] ** self.alpha
                cross_sfd = np.min(sfd_XQ[i] + sfd_YQ[j])
                distances[i, j] = np.min([bypass_sfd, cross_sfd])
        assert np.all(distances >= 0)
        return distances

    def __call__(self, X, Y=None):
        if Y is None:
            return self._sample_distance(X)
        else:
            return self._distance(X, Y)


class FermatKNeighborsClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_neighbors=5, alpha=1, weights="uniform", n_jobs=-1):
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.weights = weights
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        self.distance_ = SampleFermatDistance(Q=X, alpha=self.alpha, groups=y)
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
