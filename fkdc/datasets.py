from sklearn.utils import Bunch
import numpy as np
from sklearn.datasets import (
    load_iris,
    load_digits,
    make_moons,
    fetch_openml,
    make_circles,
)

def hacer_espiral_fermat(a, n, turns=4):
    phi = np.arange(n) / n * 2 * np.pi * turns
    arm = np.vstack([a * np.sqrt(phi) * np.cos(phi), a * np.sqrt(phi) * np.sin(phi)]).T
    X = np.vstack([arm, -arm])
    y = np.hstack([np.zeros(n), np.ones(n)])
    return X, y


class Dataset(Bunch):

    def __init__(self, nombre, X=None, y=None):
        self.nombre = nombre
        self.X = X
        self.y = y
        self.n, self.p = X.shape
        self.k = len(np.unique(y))


    def __repr__(self):
        return f"Dataset('{self.nombre}', n={self.n}, p={self.p}, k={self.k})"


datasets = [
    ("iris", *load_iris(return_X_y=True)),
    ("digitos", *load_digits(return_X_y=True)),
    ("lunas", *make_moons()),
    ("circuloas", *make_circles()),
    ("espirales", *hacer_espiral_fermat(1, 1000)),
    ("mnist", *fetch_openml("mnist_784", version=1, return_X_y=True)),
]
datasets = Bunch(**{nombre: Dataset(nombre, X, y) for nombre, X, y in datasets})
