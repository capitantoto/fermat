"""
Clasificadores clásicos, sin redes preentrenadas de ningún tipo, que podrían acercarse a
TabPFN en las tareas de la tesis.

Como el compendio (`fkdc.compendio`), no forman parte de los resultados de la tesis:
tienen registro propio (`clasificadores`, `grillas`) y resultados propios en
`sandbox/fundacionales/infos` (ver `scripts/clasicos.py`).

- `kdc_ard`: KDC con núcleo gaussiano producto y un ancho de banda por variable, que se
  eligen maximizando la log-verosimilitud dejando uno afuera de la probabilidad a
  posteriori P(y|x): la validación cruzada por verosimilitud para densidades
  condicionales (Hall, Racine y Li, 2004). Una variable irrelevante para y tiende a un
  ancho de banda infinito y queda suavizada por completo (relevancia automática, ARD).
  Cada clase mezcla su KDE con un fondo defensivo (la KDE de todas las clases con
  anchos 2h y peso de 5 pseudo-observaciones), que lejos de los datos devuelve la
  previa e impide que en datos separables los anchos colapsen.
- `fkdc_ard`: el KDC de Fermat de la tesis sobre los datos divididos por los anchos de
  `kdc_ard`, con la grilla α × h de fkdc: el método de la tesis con relevancia
  automática.
- `gpc_ard`: clasificador de procesos gaussianos con núcleo RBF anisótropo (una escala
  por variable) elegido por verosimilitud marginal.
- `superlearner`: apilamiento (Breiman, 1996; van der Laan, Polley y Hubbard, 2007) de
  clasificadores clásicos rápidos (entre ellos tres KDC de este módulo), con
  probabilidades fuera del pliego y la combinación convexa que maximiza su
  log-verosimilitud.
- `bart`: BART probit (Chipman, George y McCulloch, 2010) de `stochtree`; uno contra el
  resto si hay más de dos clases.

Todas las densidades por núcleos se calculan exactamente, con logsumexp sobre matrices
de distancias, y no con `sklearn.neighbors.KernelDensity` (ver
`scripts/reproduce_sklearn_kde_bug.py`).
"""

from functools import partial

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.special import logsumexp
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch
from sklearn.utils.multiclass import unique_labels

from fkdc import config
from fkdc.compendio import ClasificadorProcesoGaussiano, _huella, _recordar
from fkdc.fermat import KDClassifier


def _escalas(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Desvío estándar de cada columna y máscara de las no constantes."""
    escalas = X.std(axis=0)
    return escalas, escalas > 0


def _softmax_log(log_puntajes: np.ndarray) -> np.ndarray:
    """Probabilidades a partir de log-puntajes por clase, normalizando con logsumexp."""
    return np.exp(log_puntajes - logsumexp(log_puntajes, axis=1, keepdims=True))


# =====================================================================
# kdc_ard
# =====================================================================


def logvero_posterior_loo(
    log_anchos, X, codigos, gradiente=True, fondo=0.0, ancho_fondo=3.0
):
    """Media de log P̂₋ᵢ(yᵢ|Xᵢ) dejando uno afuera y su gradiente en log h.

    Con núcleo gaussiano producto K_h y previas iguales a las proporciones de clase,
    π_k f̂_k(x) = n⁻¹ Σ_{l: y_l=k} K_h(x - X_l): la normalización Π_j h_j se cancela y,
    sin fondo (η = `fondo` = 0), la probabilidad a posteriori es el cociente de
    Nadaraya–Watson Σ_{l: y_l=k} K_h(x - X_l) / Σ_l K_h(x - X_l). Con fondo, cada clase
    tiene densidad (1 - η) f̂_k + η ĝ, con ĝ la KDE de todas las clases y anchos c·h
    (c = `ancho_fondo`): una mezcla defensiva que, lejos de todos los datos en
    unidades de h, devuelve la previa. Dejar afuera a Xᵢ quita su término de todas
    las sumas y de las previas.

    Con a_il = -½ Σ_j (Δ_ilj/h_j)², F_ik = log(1-η) + log Σ_{l∈k} e^{a_il},
    G_ik = log η + log π_k - d log c + log Σ_l e^{a_il/c²} y log S_ik = logaddexp(F, G):
    ∂a_il/∂log h_j = (Δ_ilj/h_j)² y el gradiente es Σ_il M_il (Δ_ilj/h_j)², con
    M_il = (δ - q)_{i,k(l)} r_{i,k(l)} W_il + Σ_k (δ - q)_ik (1 - r_ik) V_il / c²,
    q la posterior, r = e^{F - log S}, W los pesos softmax de la fila i dentro de la
    clase de l y V los del fondo. Las observaciones únicas en su clase no entran.
    """
    n, d = X.shape
    Z = X / np.exp(log_anchos)
    log_nucleos = -0.5 * squareform(pdist(Z, "sqeuclidean"))
    np.fill_diagonal(log_nucleos, -np.inf)
    conteos = np.bincount(codigos)
    n_clases = len(conteos)
    uno = np.zeros((n, n_clases))
    uno[np.arange(n), codigos] = 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        lse_clase = np.column_stack(
            [logsumexp(log_nucleos[:, codigos == k], axis=1) for k in range(n_clases)]
        )
        if fondo > 0:
            log_previas = np.log(conteos - uno) - np.log(n - 1)
            lse_fondo = logsumexp(log_nucleos / ancho_fondo**2, axis=1)
            F = np.log1p(-fondo) + lse_clase
            G = np.log(fondo) - d * np.log(ancho_fondo) + lse_fondo[:, None]
            log_S = np.logaddexp(F, G + log_previas)
        else:
            log_S = lse_clase
        log_total = logsumexp(log_S, axis=1)
    log_propia = log_S[np.arange(n), codigos]
    validos = np.isfinite(log_propia) & (conteos[codigos] > 1)
    n_validos = validos.sum()
    valor = (log_propia[validos] - log_total[validos]).sum() / n_validos
    if not gradiente:
        return valor
    with np.errstate(invalid="ignore", over="ignore"):
        q = np.exp(log_S - log_total[:, None])
        r = np.exp(F - log_S) if fondo > 0 else np.ones_like(q)
        r = np.nan_to_num(r)
        coef = uno - q
        coef[~validos] = 0.0
        W = np.exp(log_nucleos - lse_clase[:, codigos])
        pesos = (coef * r)[:, codigos] * W
        if fondo > 0:
            V = np.exp(log_nucleos / ancho_fondo**2 - lse_fondo[:, None])
            pesos += (coef * (1 - r)).sum(axis=1)[:, None] * V / ancho_fondo**2
    pesos = np.nan_to_num(pesos) / n_validos
    # Σ_il M_il (Z_ij - Z_lj)²
    #   = Σ_i Z_ij² Σ_l M_il + Σ_l Z_lj² Σ_i M_il - 2 Σ_i Z_ij (MZ)_ij
    cuadrados = Z**2
    grad = (
        cuadrados.T @ pesos.sum(axis=1)
        + cuadrados.T @ pesos.sum(axis=0)
        - 2 * (Z * (pesos @ Z)).sum(axis=0)
    )
    return valor, grad


class KDClassifierARD(ClassifierMixin, BaseEstimator):
    """KDC gaussiano producto con un ancho de banda por variable (ARD).

    Los anchos h_j maximizan la log-verosimilitud dejando uno afuera de P̂(y|x)
    (`logvero_posterior_loo`; Hall, Racine y Li, 2004): una variable irrelevante para
    y lleva su h_j hacia infinito y deja de contar en las distancias. Se optimiza
    u_j = log(h_j / s_j), con s_j el desvío de la variable (el estimador es
    equivariante por escala), acotado a [-log cota, log cota]: en la cota superior la
    variable está quitada.

    Fondo defensivo: cada clase tiene densidad (1 - η) f̂_k + η ĝ, con ĝ la KDE de
    todas las clases con anchos c·h (c = `ancho_fondo`) y η = a / (n + a), el peso de
    `fondo` = a pseudo-observaciones. Cerca de los datos domina f̂_k; lejos de todos
    ellos, en unidades de h, domina ĝ (sus núcleos decaen más despacio) y la
    posterior vuelve a la previa. Sin fondo, en datos separables la verosimilitud
    dejando uno afuera tiende a 0 a medida que los anchos se achican, el óptimo
    degenera en h → 0 y un punto de evaluación lejos de todo recibe probabilidades
    como 1e-27 para su clase. Con fondo, achicar h por debajo del espaciado entre
    vecinos lleva las posteriores dejando uno afuera a la previa y se penaliza. En
    dimensión alta el factor c^(-d) hace al fondo despreciable cerca de los datos.

    Optimización (el problema no es cóncavo). Puntos de partida: la regla normal de
    Silverman por variable, h_j = s_j (4 / ((d + 2) n))^(1/(d+4)), por cada factor
    de `inicios_ard`. Primero, el KDC isótropo sobre los datos estandarizados: un
    factor común elegido en la grilla `factores_iniciales` y luego por L-BFGS-B. Con
    variables de ruido, ese óptimo suele ser el clasificador trivial (h → ∞), un
    punto estacionario del que ARD no sale; con `isotropo=True` el ajuste termina
    ahí. Después, L-BFGS-B ajusta cada h_j: (1) sin fondo, desde cada inicio de
    Silverman, y (2) con fondo, desde el óptimo de (1), el isótropo y cada inicio. Se
    queda con el mejor objetivo con fondo. La fase sin fondo hace falta porque con
    anchos chicos en muchas dimensiones el fondo domina todas las posteriores dejando
    uno afuera: el objetivo es plano, el gradiente nulo y L-BFGS-B no se mueve. Cada
    corrida hace a lo sumo `max_iter` iteraciones: en `mnist` y `digitos` las 500
    no mejoran la evaluación respecto de 100 y cuestan el triple.

    Con `escala_comun=True`, todas las s_j son la raíz de la varianza media de las
    variables: el KDC isótropo es entonces el de la tesis (un único h en las unidades
    crudas), que en coordenadas como las de PCA de `mnist` conserva el peso de cada
    componente.

    Las variables constantes no entran (ancho infinito). Las probabilidades son
    exactas: logsumexp de los log-núcleos y normalización entre clases, sin recortes.
    """

    def __init__(
        self,
        isotropo: bool = False,
        escala_comun: bool = False,
        fondo: float = 5.0,
        ancho_fondo: float = 2.0,
        inicios_ard: tuple = (1 / 3, 1.0, 3.0),
        cota: float = 1e3,
        max_iter: int = 100,
    ):
        self.isotropo = isotropo
        self.escala_comun = escala_comun
        self.fondo = fondo
        self.ancho_fondo = ancho_fondo
        self.inicios_ard = inicios_ard
        self.cota = cota
        self.max_iter = max_iter

    factores_iniciales = np.logspace(-1.5, 1.5, 13)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, codigos = np.unique(y, return_inverse=True)
        self.escalas_, self.activas_ = _escalas(X)
        if self.escala_comun:
            activas = self.escalas_[self.activas_]
            self.escalas_[self.activas_] = np.sqrt(np.mean(activas**2))
        X_activas = X[:, self.activas_]
        n, d = X_activas.shape
        log_s = np.log(self.escalas_[self.activas_])
        u_silverman = np.log(4 / ((d + 2) * n)) / (d + 4)
        limite = np.log(self.cota)
        self.peso_fondo_ = self.fondo / (n + self.fondo)

        def loo(log_anchos, gradiente=True, con_fondo=True):
            return logvero_posterior_loo(
                log_anchos,
                X_activas,
                codigos,
                gradiente=gradiente,
                fondo=self.peso_fondo_ if con_fondo else 0.0,
                ancho_fondo=self.ancho_fondo,
            )

        def objetivo(u, isotropo, con_fondo):
            valor, grad = loo(log_s + (u[0] if isotropo else u), con_fondo=con_fondo)
            return -valor, -(grad.sum(keepdims=True) if isotropo else grad)

        def optimizar(u, isotropo=False, con_fondo=True):
            resultado = minimize(
                objetivo,
                np.clip(u, -limite, limite),
                args=(isotropo, con_fondo),
                jac=True,
                method="L-BFGS-B",
                bounds=[(-limite, limite)] * len(u),
                options={"maxiter": self.max_iter},
            )
            self.iteraciones_ += resultado.nit
            return resultado

        self.iteraciones_ = 0
        candidatos = u_silverman + np.log(self.factores_iniciales)
        valores = [loo(log_s + u, gradiente=False) for u in candidatos]
        isotropo = optimizar([candidatos[int(np.argmax(valores))]], isotropo=True)
        mejor = Bunch(u=np.full(d, isotropo.x[0]), valor=-isotropo.fun)
        self.inicio_ = "isotropo"
        if not self.isotropo and d > 1:
            inicios = {
                f"silverman×{factor:.3g}": np.full(d, u_silverman + np.log(factor))
                for factor in self.inicios_ard
            }
            sin_fondo = min(
                (optimizar(u, con_fondo=False) for u in inicios.values()),
                key=lambda resultado: resultado.fun,
            )
            inicios = {"sin_fondo": sin_fondo.x, "isotropo": mejor.u, **inicios}
            for nombre, u in inicios.items():
                ard = optimizar(u)
                if -ard.fun > mejor.valor:
                    mejor = Bunch(u=ard.x, valor=-ard.fun)
                    self.inicio_ = nombre
        self.logvero_loo_isotropo_ = -isotropo.fun
        self.logvero_loo_ = mejor.valor
        self.anchos_ = np.full(X.shape[1], np.inf)
        self.anchos_[self.activas_] = np.exp(log_s + mejor.u)
        self.Z_ = X_activas / self.anchos_[self.activas_]
        self.codigos_ = codigos
        self.log_previas_ = np.log(np.bincount(codigos) / n)
        return self

    @property
    def relativos_(self) -> np.ndarray:
        """h_j / s_j de cada variable (infinito en las constantes)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.activas_, self.anchos_ / self.escalas_, np.inf)

    def log_puntajes(self, X) -> np.ndarray:
        """log S_k(x) por fila de X y clase k, salvo constantes comunes a las clases.

        S_k = (1 - η) Σ_{i: yᵢ=k} e^{aᵢ} + η π_k c^(-d) Σ_i e^{aᵢ/c²}, con
        aᵢ = -½ Σ_j ((x_j - X_ij)/h_j)², es n π_k [(1 - η) f̂_k(x) + η ĝ(x)] salvo el
        factor Π_j h_j (2π)^(d/2), común a todas las clases.
        """
        Z = np.asarray(X, dtype=float)[:, self.activas_] / self.anchos_[self.activas_]
        log_nucleos = -0.5 * cdist(Z, self.Z_, "sqeuclidean")
        lse_clase = np.column_stack(
            [
                logsumexp(log_nucleos[:, self.codigos_ == k], axis=1)
                for k in range(len(self.classes_))
            ]
        )
        if self.peso_fondo_ == 0:
            return lse_clase
        lse_fondo = logsumexp(log_nucleos / self.ancho_fondo**2, axis=1)
        G = (
            np.log(self.peso_fondo_)
            - Z.shape[1] * np.log(self.ancho_fondo)
            + lse_fondo[:, None]
            + self.log_previas_
        )
        return np.logaddexp(np.log1p(-self.peso_fondo_) + lse_clase, G)

    def predict_proba(self, X):
        return _softmax_log(self.log_puntajes(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# =====================================================================
# fkdc_ard
# =====================================================================


def _anchos_ard(X, y) -> np.ndarray:
    """Anchos de `KDClassifierARD` sobre (X, y), memorizados en el proceso.

    La búsqueda en grilla de `KDClassifierFermatARD` ajusta un clasificador por pliego
    y combinación (α, h), pero los anchos dependen solo del pliego: se calculan una vez.
    """
    return _recordar(
        ("anchos_ard", _huella(X, y)), lambda: KDClassifierARD().fit(X, y).anchos_
    )


class KDClassifierFermatARD(ClassifierMixin, BaseEstimator):
    """KDC de Fermat de la tesis sobre X_j / h_j, con h_j los anchos de `kdc_ard`.

    Los anchos se ajustan dentro de cada pliego (sin filtrar información de la
    validación a la escala) y dividen cada variable: las irrelevantes, con h_j en la
    cota, quedan casi constantes y no pesan en la distancia de Fermat. Sobre esas
    coordenadas, `KDClassifier(metric="fermat")` con sus `alpha` y `bandwidth`, los
    mismos hiperparámetros y la misma dirección de parsimonia que fkdc. Con α = 1 la
    distancia de Fermat muestral es la euclídea y, con h = 1, el clasificador es
    `kdc_ard` sin su fondo defensivo y con el recorte de log-densidades de
    `KDClassifier`.
    """

    def __init__(self, alpha: float = 1.0, bandwidth: float = 1.0):
        self.alpha = alpha
        self.bandwidth = bandwidth

    def _reescalar(self, X):
        X = np.asarray(X, dtype=float)
        finitos = np.isfinite(self.anchos_)
        return X[:, finitos] / self.anchos_[finitos]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.anchos_ = _anchos_ard(X, y)
        self.kdc_ = KDClassifier(
            bandwidth=self.bandwidth, metric="fermat", alpha=self.alpha
        ).fit(self._reescalar(X), y)
        self.classes_ = self.kdc_.classes_
        return self

    def predict_proba(self, X):
        return self.kdc_.predict_proba(self._reescalar(X))

    def predict(self, X):
        return self.kdc_.predict(self._reescalar(X))


# =====================================================================
# gpc_ard
# =====================================================================


def _lbfgs_con_reinicios(obj_func, initial_theta, bounds, factores=(1.0,)):
    """L-BFGS-B desde θ₀ y desde θ₀ con las escalas multiplicadas por cada factor.

    Para un núcleo C·RBF, θ = (log C, log ℓ_1, …, log ℓ_d): los reinicios desplazan
    todas las log-escalas a la vez y dejan log C. Devuelve el mejor óptimo.
    """
    mejor = None
    for factor in factores:
        theta = np.asarray(initial_theta, dtype=float).copy()
        theta[1:] += np.log(factor)
        theta = np.clip(theta, bounds[:, 0], bounds[:, 1])
        resultado = minimize(
            obj_func, theta, method="L-BFGS-B", jac=True, bounds=bounds
        )
        if mejor is None or resultado.fun < mejor.fun:
            mejor = resultado
    return mejor.x, mejor.fun


class ClasificadorProcesoGaussianoARD(ClassifierMixin, BaseEstimator):
    """`GaussianProcessClassifier` con núcleo C·RBF anisótropo, sin grilla.

    Una escala ℓ_j por variable, elegidas con C maximizando la verosimilitud marginal
    (aproximación de Laplace; uno contra el resto si hay más de dos clases, cada uno
    con sus propias escalas). Ajusta sobre las variables estandarizadas (sin las
    constantes) y parte, como `compendio.ClasificadorProcesoGaussiano`, de todas las
    escalas iguales a la mediana de las distancias entre observaciones, que allí son
    comparables; desde ℓ = 1 en las unidades crudas el núcleo es casi la identidad
    (p. ej. en `mnist`) y el gradiente se anula. Cotas de ℓ_j en unidades de desvío:
    `cotas_escala`; en la superior la variable está quitada. `reinicios` son factores
    que multiplican todas las escalas iniciales: el optimizador corre desde cada uno y
    se queda con la mayor verosimilitud marginal.
    """

    def __init__(
        self,
        reinicios: tuple = (1.0, 0.25),
        cotas_escala: tuple = (1e-2, 1e3),
        random_state: int | None = None,
    ):
        self.reinicios = reinicios
        self.cotas_escala = cotas_escala
        self.random_state = random_state

    def _estandarizar(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.activas_] / self.escalas_[self.activas_]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.escalas_, self.activas_ = _escalas(X)
        Z = self._estandarizar(X)
        mediana = np.median(pdist(Z))
        nucleo = ConstantKernel(1.0) * RBF(
            length_scale=np.full(Z.shape[1], mediana),
            length_scale_bounds=self.cotas_escala,
        )
        self.estimador_ = GaussianProcessClassifier(
            kernel=nucleo,
            optimizer=partial(_lbfgs_con_reinicios, factores=self.reinicios),
            random_state=self.random_state,
        ).fit(Z, y)
        self.classes_ = self.estimador_.classes_
        return self

    @property
    def escalas_longitud_(self) -> np.ndarray:
        """ℓ_j en unidades de desvío: una fila por clasificador binario."""
        base = self.estimador_.base_estimator_
        binarios = getattr(base, "estimators_", [base])
        return np.array([b.kernel_.k2.length_scale for b in binarios])

    def predict_proba(self, X):
        return self.estimador_.predict_proba(self._estandarizar(X))

    def predict(self, X):
        return self.estimador_.predict(self._estandarizar(X))


# =====================================================================
# superlearner
# =====================================================================


def _estandarizado(clasificador):
    return Pipeline([("escala", StandardScaler()), ("clf", clasificador)])


def biblioteca(semilla: int | None = None) -> dict:
    """Miembros del apilamiento: clasificadores clásicos rápidos, sin búsqueda.

    Cada uno trae sus hiperparámetros fijos o los elige por su cuenta (los KDC, por
    validación cruzada dejando uno afuera; los procesos gaussianos, por verosimilitud
    marginal). Tres KDC: isótropo sobre las variables estandarizadas, isótropo en las
    unidades crudas (el de la tesis) y ARD. Varios k de vecinos son miembros
    distintos, como en la biblioteca de un Super Learner. Los sensibles a la escala
    van estandarizados.
    """
    return {
        "lr": _estandarizado(LogisticRegression(max_iter=10_000)),
        "lda": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "qda": _estandarizado(QuadraticDiscriminantAnalysis(reg_param=0.1)),
        "gnb": GaussianNB(),
        "kn_5": _estandarizado(KNeighborsClassifier(5)),
        "kn_15": _estandarizado(KNeighborsClassifier(15)),
        "kn_45": _estandarizado(KNeighborsClassifier(45)),
        "kdc": KDClassifierARD(isotropo=True),
        "kdc_crudo": KDClassifierARD(isotropo=True, escala_comun=True),
        "kdc_ard": KDClassifierARD(),
        "gpc": _estandarizado(ClasificadorProcesoGaussiano(random_state=semilla)),
        "gbt": HistGradientBoostingClassifier(max_features=0.5, random_state=semilla),
        "et": ExtraTreesClassifier(n_estimators=300, random_state=semilla),
    }


def _probas_alineadas(clasificador, X, clases) -> np.ndarray:
    """`predict_proba` con columnas en el orden de `clases` (0 si falta alguna)."""
    probas = np.zeros((len(X), len(clases)))
    columnas = np.searchsorted(clases, clasificador.classes_)
    probas[:, columnas] = clasificador.predict_proba(X)
    return probas


def pesos_convexos(verosimilitudes: np.ndarray, tolerancia: float = 1e-10) -> tuple:
    """Pesos w ≥ 0, Σw = 1 que maximizan Σᵢ log Σ_m w_m p_im (EM para mezclas).

    `verosimilitudes[i, m]` es la probabilidad que el miembro m asignó, fuera del
    pliego, a la clase observada de i. El objetivo es cóncavo en w y la iteración EM
    w_m ← n⁻¹ Σᵢ w_m p_im / Σ_k w_k p_ik lo aumenta en cada paso hacia el máximo
    global. Las probabilidades se acotan por abajo en 1e-15, como `log_loss`.
    """
    p = np.clip(verosimilitudes, 1e-15, 1.0)
    pesos = np.full(p.shape[1], 1 / p.shape[1])
    for iteracion in range(1, 20_001):  # noqa: B007
        nuevos = pesos * (p / (p @ pesos)[:, None]).mean(axis=0)
        nuevos /= nuevos.sum()
        if np.abs(nuevos - pesos).max() < tolerancia:
            pesos = nuevos
            break
        pesos = nuevos
    return pesos, np.log(p @ pesos).mean(), iteracion


class SuperAprendiz(ClassifierMixin, BaseEstimator):
    """Apilamiento con combinación convexa de probabilidades (Super Learner).

    1. Probabilidades fuera del pliego de cada miembro de `biblioteca` sobre el pliego
       de entrenamiento (`pliegues` pliegues estratificados).
    2. Pesos convexos que maximizan la log-verosimilitud de esas probabilidades
       (`pesos_convexos`; el método NNloglik de van der Laan, Polley y Hubbard).
    3. Reajuste, sobre todo el pliego de entrenamiento, de los miembros con peso
       mayor que `peso_minimo` (los demás se descartan y los pesos se renormalizan).

    La probabilidad final es la mezcla de las de los miembros, válida por
    construcción. Un miembro que falla en algún pliego queda fuera.
    """

    def __init__(
        self,
        pliegues: int = 5,
        peso_minimo: float = 1e-4,
        random_state: int | None = None,
    ):
        self.pliegues = pliegues
        self.peso_minimo = peso_minimo
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_, codigos = np.unique(y, return_inverse=True)
        miembros = biblioteca(self.random_state)
        pliegues = StratifiedKFold(
            min(self.pliegues, np.bincount(codigos).min()),
            shuffle=True,
            random_state=self.random_state,
        )
        verosimilitudes, self.fallas_ = {}, {}
        for nombre, miembro in miembros.items():
            columna = np.empty(len(y))
            try:
                for entrenar, validar in pliegues.split(X, codigos):
                    ajustado = clone(miembro).fit(X[entrenar], y[entrenar])
                    probas = _probas_alineadas(ajustado, X[validar], self.classes_)
                    columna[validar] = probas[np.arange(len(validar)), codigos[validar]]
            except Exception as exc:  # un miembro que falla no detiene al resto
                self.fallas_[nombre] = f"{type(exc).__name__}: {exc}"
                continue
            if np.isfinite(columna).all():
                verosimilitudes[nombre] = columna
        nombres = list(verosimilitudes)
        pesos, self.logvero_oof_, self.iteraciones_ = pesos_convexos(
            np.column_stack([verosimilitudes[n] for n in nombres])
        )
        self.logvero_oof_miembros_ = {
            n: float(np.log(np.clip(v, 1e-15, 1)).mean())
            for n, v in verosimilitudes.items()
        }
        elegidos = pesos > self.peso_minimo
        self.pesos_ = dict(
            zip(
                [n for n, e in zip(nombres, elegidos, strict=True) if e],
                pesos[elegidos] / pesos[elegidos].sum(),
                strict=True,
            )
        )
        self.miembros_ = {n: clone(miembros[n]).fit(X, y) for n in self.pesos_}
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return sum(
            peso * _probas_alineadas(self.miembros_[nombre], X, self.classes_)
            for nombre, peso in self.pesos_.items()
        )

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# =====================================================================
# bart
# =====================================================================


class ClasificadorBART(ClassifierMixin, BaseEstimator):
    """BART probit (Chipman, George y McCulloch, 2010) con `stochtree`, sin grilla.

    P(y = 1 | x) = Φ(f(x)), f una suma de `n_arboles` árboles con la previa de
    regularización de BART (α = 0.95, β = 2) y el muestreador de `stochtree`:
    `n_gfr` iteraciones de arranque *grow-from-root* (XBART) y `n_mcmc` muestras
    retenidas de MCMC. La probabilidad es la media a posteriori de Φ(f(x)), la
    predictiva a posteriori. `stochtree` no tiene un modelo multinomial: con más de dos
    clases se ajusta un BART binario por clase contra el resto y se normaliza.
    """

    def __init__(
        self,
        n_arboles: int = 200,
        n_gfr: int = 10,
        n_mcmc: int = 100,
        random_state: int | None = None,
    ):
        self.n_arboles = n_arboles
        self.n_gfr = n_gfr
        self.n_mcmc = n_mcmc
        self.random_state = random_state

    def _binario(self, X, y01):
        from stochtree import BARTModel, OutcomeModel

        modelo = BARTModel()
        modelo.sample(
            X_train=X,
            y_train=y01.astype(float),
            num_gfr=self.n_gfr,
            num_burnin=0,
            num_mcmc=self.n_mcmc,
            general_params={
                "outcome_model": OutcomeModel(outcome="binary", link="probit"),
                "random_seed": -1 if self.random_state is None else self.random_state,
                "num_threads": 1,
            },
            mean_forest_params={"num_trees": self.n_arboles},
        )
        return modelo

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_ = unique_labels(y)
        y = np.asarray(y)
        clases = self.classes_[1:] if len(self.classes_) == 2 else self.classes_
        self.modelos_ = [self._binario(X, y == clase) for clase in clases]
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        probas = np.column_stack(
            [
                norm.cdf(
                    modelo.predict(X, type="posterior", terms="y_hat", scale="linear")
                ).mean(axis=1)
                for modelo in self.modelos_
            ]
        )
        if len(self.classes_) == 2:
            return np.column_stack([1 - probas[:, 0], probas[:, 0]])
        return probas / probas.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# =====================================================================
# Registro
# =====================================================================

clasificadores = Bunch(
    kdc_ard=KDClassifierARD(),
    fkdc_ard=KDClassifierFermatARD(),
    gpc_ard=ClasificadorProcesoGaussianoARD(),
    superlearner=SuperAprendiz(),
    bart=ClasificadorBART(),
)
grillas = {
    # Sin grilla: los anchos, por validación cruzada dejando uno afuera
    "kdc_ard": {},
    # La grilla de fkdc de la tesis (13 α × 45 h), sobre las coordenadas X_j / h_j
    "fkdc_ard": config.grillas["fkdc"],
    # Sin grilla: las escalas, por verosimilitud marginal
    "gpc_ard": {},
    # Sin grilla: los pesos, por log-verosimilitud fuera del pliego
    "superlearner": {},
    "bart": {},
}


def diagnostico(clasificador) -> dict:
    """Lo que eligió el clasificador ajustado, para guardar junto a su info."""
    if isinstance(clasificador, Pipeline):  # variantes `_std`
        clasificador = clasificador[-1]
    if isinstance(clasificador, KDClassifierARD):
        return {
            "anchos": clasificador.anchos_,
            "escalas": clasificador.escalas_,
            "relativos": clasificador.relativos_,
            "logvero_loo": clasificador.logvero_loo_,
            "logvero_loo_isotropo": clasificador.logvero_loo_isotropo_,
            "iteraciones": clasificador.iteraciones_,
        }
    if isinstance(clasificador, KDClassifierFermatARD):
        return {"anchos": clasificador.anchos_}
    if isinstance(clasificador, ClasificadorProcesoGaussianoARD):
        return {
            "escalas": clasificador.escalas_,
            "escalas_longitud": clasificador.escalas_longitud_,
        }
    if isinstance(clasificador, SuperAprendiz):
        return {
            "pesos": clasificador.pesos_,
            "logvero_oof": clasificador.logvero_oof_,
            "logvero_oof_miembros": clasificador.logvero_oof_miembros_,
            "fallas": clasificador.fallas_,
        }
    return {}
