from itertools import product
from numbers import Number

import numpy as np
from joblib import Memory
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.base import BaseEstimator, ClassifierMixin, DensityMixin
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.neighbors import KernelDensity

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
def sample_fermat(Q, alpha=1):
    return shortest_path(csr_matrix(euclidean(Q) ** alpha), directed=False)


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
        self.Q_ = X
        # A is the adjacency matrix with Fermat distances as edge weights
        self.A_ = sample_fermat(X, self.alpha)
        if self.d == -1:
            self.d = self.D
        return self

    @property
    def N(self):
        return self.Q_.shape[0]

    @property
    def D(self):
        return self.Q_.shape[1]

    def _sample_distances(self, X):
        to_Q = euclidean(X, self.Q_) ** self.alpha
        sample_distances = np.zeros((X.shape[0], self.N))
        for i in range(len(X)):
            sample_distances[i, :] = np.min(to_Q[i].T + self.A_, axis=1)
        return sample_distances

    def score_samples(self, X=None, log=True):
        if X is None:
            X = self.Q_

        score = np.exp(-0.5 * (self._sample_distances(X) / self.bandwidth) ** 2).sum(1)

        if log:
            return (
                -np.log(self.N)
                - self.d * np.log(self.bandwidth)
                - self.d / 2 * np.log(2 * np.pi)
                + np.maximum(np.log(score), self.MIN_LOG_SCORE)
            )

        else:
            return (
                self.N**-1
                * (self.bandwidth**-self.d)
                * (2 * np.pi) ** (-self.d / 2)
                * score
            )

    def score(self, X=None):
        if X is None:
            X = self.Q_
        return self.score_samples(X).sum()


def lattice(a, b, step=1, dim=2, array=True):
    side = np.arange(a, b, step)
    if len(side) ** dim > 1e6:
        raise ValueError(
            f"Too many points ({len(side) ** dim:.2e} > 1e6). Try a bigger step or a smaller dim."
        )
    gen = product(*[side] * dim)
    return np.array([*gen]) if array else gen


class BaseKDEClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
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


class FermatKDEClassifier(BaseEstimator, ClassifierMixin):
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
