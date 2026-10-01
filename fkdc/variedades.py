"""Variedades sintéticas con coordenadas intrínsecas conocidas, para la defensa.

Portado de `fkdc/variedades.py` de la rama `investigacion-v6` (commit 9ed8020, el
último que modifica ese archivo; punta de la rama: d13a243), recortado a las cinco
familias que se suman a la comparación: `paralelas`, `nudos`, `lissajous`, `toro` y
`esfera`. Los generadores reproducen bit a bit los datasets de aquella rama para la
misma semilla; cambian la firma (la de las fábricas de `fkdc.datasets`: `n_samples`,
`noise`, `random_state`, devuelven `(X, y)`) y los nombres de algunos datasets, que
siguen la convención `{familia}_{variante}` (p. ej., `nudos_200`, `toro_3`).

Convenciones de muestreo (heredadas de v6):

- Toda la aleatoriedad pasa por un único `np.random.Generator` derivado de
  `random_state`, de modo que cada semilla reproduce exactamente su dataset.
- El muestreo sobre la variedad usa la medida de longitud o de área correcta: ángulos
  uniformes o gaussianos solo en cartas planas de velocidad unitaria, inversión de la
  longitud de arco en las curvas cuyo parámetro no es de velocidad unitaria.
- El ruido es gaussiano isótropo en el espacio ambiente, de desvío `noise`, y se agrega
  después de muestrear sobre la variedad, como en `fkdc.datasets.hacer_eslabones`.
- Las clases quedan balanceadas y las filas, mezcladas.

Coordenadas intrínsecas: para cada familia, `theta_{familia}(X)` (registradas en
`thetas`) proyecta cada fila de X sobre la variedad generadora sin ruido y devuelve las
coordenadas de la carta con que se la genera. La carta es de forma cerrada en todas las
familias; su inversa también lo es en `paralelas`, `toro` y `esfera`, mientras que en
`nudos` y `lissajous` la proyección se resuelve numéricamente sobre una grilla fina de
cada curva. (En la rama v6, "θ" designaba la densidad de volumen de Pelletier, un objeto
distinto que este módulo no usa.)

Uso (desde la raíz del repo):

    .venv/bin/python -m fkdc.variedades generar --repeticiones 25
    .venv/bin/python -m fkdc.variedades graficar
"""

import logging
from collections.abc import Callable
from functools import cache
from inspect import signature
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.utils import Bunch

from fkdc import config, dir_raiz
from fkdc.datasets import Dataset

logger = logging.getLogger(__name__)

TAU = 2 * np.pi
dir_fundacionales = dir_raiz / "sandbox/fundacionales"


# ---------------------------------------------------------------------------
# Utilidades de muestreo
# ---------------------------------------------------------------------------


def _balance(n, k):
    """Reparte n observaciones en k clases lo más equilibradas posible."""
    base, resto = divmod(n, k)
    return [base + (i < resto) for i in range(k)]


def _ensamblar(rng, bloques, noise):
    """Apila los bloques por clase, agrega ruido ambiente N(0, noise²I) y mezcla."""
    X = np.vstack(bloques)
    y = np.concatenate([np.full(len(b), i, int) for i, b in enumerate(bloques)])
    if noise:
        X = X + rng.normal(scale=noise, size=X.shape)
    permutacion = rng.permutation(len(y))
    return X[permutacion], y[permutacion]


def _muestrear_curva(curva, rng, n, t_max=TAU, m=20001):
    """Muestra uniforme respecto de la longitud de arco sobre una curva cerrada.

    `curva: (m,) -> (m, D)` se evalúa en una grilla fina; se invierte la longitud de
    arco acumulada (interpolación lineal en el parámetro, error O((t_max/m)²)) y se
    evalúa la curva exacta en el parámetro obtenido: los puntos caen exactamente sobre
    la variedad y la densidad es uniforme en longitud de arco salvo el error de grilla.
    """
    ts = np.linspace(0.0, t_max, m)
    puntos = np.asarray(curva(ts))
    segmentos = np.linalg.norm(np.diff(puntos, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(segmentos)])
    u = rng.uniform(0.0, s[-1], size=n)
    return np.asarray(curva(np.interp(u, s, ts)))


@cache
def _grillas_curvas(familia: str, m: int = 20_000):
    """Grilla regular del parámetro y árbol k-d de cada curva de `familia`."""
    ts = np.linspace(0.0, TAU, m, endpoint=False)
    return [(ts, cKDTree(curva(ts))) for curva in curvas[familia]]


def _theta_curvas(X, familia: str):
    """(t₀, d₀, t₁, d₁): parámetro del punto más cercano de cada curva y distancia."""
    columnas = []
    for ts, arbol in _grillas_curvas(familia):
        distancia, indice = arbol.query(X)
        columnas += [ts[indice], distancia]
    return np.column_stack(columnas)


# ---------------------------------------------------------------------------
# 1. Paralelas: dos círculos a brecha constante sobre el cilindro unitario
# ---------------------------------------------------------------------------


def parametrizar_paralelas(t, z):
    """Cilindro unitario en coordenadas (t, z) ↦ (cos t, sin t, z), una isometría."""
    return np.column_stack([np.cos(t), np.sin(t), z])


def theta_paralelas(X):
    """θ = (t, z), coordenadas cilíndricas: ángulo en (-π, π] y altura.

    Ambas clases viven sobre el cilindro unitario, a alturas z = ±δ/2; t es la
    coordenada intrínseca a lo largo de cada círculo y z la que separa las clases.
    """
    return np.column_stack([np.arctan2(X[:, 1], X[:, 0]), X[:, 2]])


def hacer_paralelas(n_samples=800, m_brecha=4, noise=0.0039, random_state=None):
    """Dos círculos unitarios paralelos, a alturas z = ±δ/2, en R³.

    Variedad: γ_±(t) = (cos t, sin t, ±δ/2), d = 1, D = 3, longitud 2π fija. La
    etiqueta es el círculo de origen. La brecha ambiente entre clases es exactamente δ
    a lo largo de toda la curva: el vecino más cercano de la otra clase de cualquier
    punto está a distancia δ, mientras que dentro de la clase los vecinos están a un
    espaciado típico s = 2π/(n/2). El parámetro es δ = m_brecha · s: con m_brecha
    grande el vecino más cercano es siempre de la misma clase; con m_brecha ≈ 1 un
    salto entre círculos cuesta lo mismo que un paso dentro del círculo, y la
    distancia de Fermat (que abarata muchos pasos cortos frente a un salto largo)
    debería distinguir lo que la euclídea confunde. Las clases son trivialmente
    separables por z: el mecanismo bajo prueba es local, no la separabilidad.
    Ángulos uniformes, n balanceado, ruido σ = 0.0039 ≈ s/4 (para n = 800).
    """
    rng = np.random.default_rng(random_state)
    delta = m_brecha * TAU / (n_samples / 2)
    n0, n1 = _balance(n_samples, 2)
    t0 = rng.uniform(0, TAU, n0)
    t1 = rng.uniform(0, TAU, n1)
    bloques = [
        parametrizar_paralelas(t0, np.full(n0, +delta / 2)),
        parametrizar_paralelas(t1, np.full(n1, -delta / 2)),
    ]
    return _ensamblar(rng, bloques, noise)


# ---------------------------------------------------------------------------
# 2. Nudos: dos nudos tóricos enlazados en R³
# ---------------------------------------------------------------------------

R_NUDOS, R_TUBO_NUDOS = 2.0, 0.8  # radios mayor y menor de los toros portadores


def _nudo_torico(p, q, R, r):
    """Nudo tórico (p, q) sobre el toro de radios (R, r) alrededor del eje z."""

    def curva(t):
        rho = R + r * np.cos(q * t)
        return np.column_stack(
            [rho * np.cos(p * t), rho * np.sin(p * t), r * np.sin(q * t)]
        )

    return curva


def _mover_nudo_1(P, R):
    """Rota 90° sobre el eje x, (x, y, z) ↦ (x, -z, y), y traslada x += R."""
    return np.column_stack([P[:, 0] + R, -P[:, 2], P[:, 1]])


def curvas_nudos(R=R_NUDOS, r=R_TUBO_NUDOS):
    """Parametrizaciones t ∈ [0, 2π) ↦ R³ del nudo de cada clase, como se generan."""
    nudo_0 = _nudo_torico(2, 3, R, r)
    nudo_1 = _nudo_torico(3, 2, R, r)
    return nudo_0, lambda t: _mover_nudo_1(nudo_1(t), R)


def theta_nudos(X):
    """θ = (t₀, d₀, t₁, d₁): parámetro y distancia al punto más cercano de cada nudo.

    t_k es la coordenada intrínseca (el parámetro de `curvas_nudos`, no la longitud de
    arco) de la proyección sobre el nudo k, y d_k la distancia a él. Proyección
    numérica sobre una grilla de 20 000 puntos por nudo.
    """
    return _theta_curvas(X[:, :3], "nudos")


def hacer_nudos(
    n_samples=800, R=R_NUDOS, r=R_TUBO_NUDOS, noise=0.05, random_state=None
):
    """Nudo trébol (2, 3) contra nudo (3, 2), enlazados como eslabones, en R³.

    Variedad: dos curvas cerradas (nudos tóricos sobre toros R = 2, r = 0.8), d = 1,
    D = 3. La etiqueta es el nudo de origen. Clase 0: nudo (2, 3) alrededor del eje z,
    centrado en el origen. Clase 1: nudo (3, 2) rotado 90° sobre el eje x y trasladado
    x += R, enlazado con el primero como en `fkdc.datasets.hacer_eslabones`. Cada nudo
    se pliega sobre sí mismo y pasa cerca del otro: con muestreo ralo, el vecino
    euclídeo más cercano puede estar en otra vuelta o en el otro nudo. Muestreo
    uniforme en longitud de arco (el parámetro del nudo tórico no es de velocidad
    unitaria). n balanceado, ruido σ = 0.05.
    """
    rng = np.random.default_rng(random_state)
    n0, n1 = _balance(n_samples, 2)
    A = _muestrear_curva(_nudo_torico(2, 3, R, r), rng, n0)
    B = _mover_nudo_1(_muestrear_curva(_nudo_torico(3, 2, R, r), rng, n1), R)
    return _ensamblar(rng, [A, B], noise)


# ---------------------------------------------------------------------------
# 3. Lissajous: dos curvas cerradas en R⁵
# ---------------------------------------------------------------------------

FRECUENCIAS_LISSAJOUS = ((1, 2, 3, 4, 5), (5, 4, 3, 2, 1))
FASES_LISSAJOUS = ((0.0, 0.4, 0.8, 1.2, 1.6), (0.9, 1.7, 0.3, 1.1, 0.5))


def _curva_lissajous(frecuencias, fases):
    """γ(t) = (sin(a_j t + δ_j))_j, t ∈ [0, 2π)."""

    def curva(t):
        t = np.atleast_1d(t)
        pares = zip(frecuencias, fases, strict=True)
        return np.column_stack([np.sin(a * t + f) for a, f in pares])

    return curva


def curvas_lissajous():
    """Parametrizaciones t ∈ [0, 2π) ↦ R⁵ de la curva de cada clase."""
    pares = zip(FRECUENCIAS_LISSAJOUS, FASES_LISSAJOUS, strict=True)
    return tuple(_curva_lissajous(a, f) for a, f in pares)


def theta_lissajous(X):
    """θ = (t₀, d₀, t₁, d₁): parámetro y distancia al punto más cercano de cada curva.

    Proyección numérica sobre una grilla de 20 000 puntos por curva.
    """
    return _theta_curvas(X[:, :5], "lissajous")


def hacer_lissajous(n_samples=800, noise=0.05, random_state=None):
    """Dos curvas de Lissajous cerradas en R⁵.

    Variedad: γ_k(t) = (sin(a_j t + δ_j))_{j=1..5}, t ∈ [0, 2π), con frecuencias
    enteras (1, 2, 3, 4, 5) para la clase 0 y (5, 4, 3, 2, 1) para la 1, y fases fijas;
    d = 1, D = 5. La etiqueta es la curva de origen. Ambas curvas se pliegan sobre el
    cubo [-1, 1]⁵ y se acercan mutuamente sin tocarse: dimensión ambiente mayor sin
    dimensiones de ruido puro (a diferencia de las variantes `_12`), con d = 1.
    Muestreo uniforme en longitud de arco, n balanceado, ruido σ = 0.05.
    """
    rng = np.random.default_rng(random_state)
    bloques = [
        _muestrear_curva(curva, rng, m)
        for curva, m in zip(curvas_lissajous(), _balance(n_samples, 2), strict=True)
    ]
    return _ensamblar(rng, bloques, noise)


# ---------------------------------------------------------------------------
# 4. Toro plano T^d = (S¹)^d ⊂ R^{2d}
# ---------------------------------------------------------------------------


def parametrizar_toro(angulos):
    """(φ₁, …, φ_d) ↦ (cos φ₁, sin φ₁, …, cos φ_d, sin φ_d), carta plana."""
    return np.column_stack(
        [f(angulos[:, j]) for j in range(angulos.shape[1]) for f in (np.cos, np.sin)]
    )


def theta_toro(X):
    """θ = (φ₁, …, φ_d) ∈ (-π, π]^d, el ángulo de cada par de coordenadas."""
    return np.arctan2(X[:, 1::2], X[:, 0::2])


def hacer_toro(n_samples=800, d=3, sigma_clase=0.8, noise=0.05, random_state=None):
    """Dos lomas gaussianas envueltas sobre el toro plano T^d ⊂ R^{2d}.

    Variedad: producto de d círculos unitarios, D = 2d, intrínsecamente plana. La
    etiqueta es la loma de origen: clase 0 centrada en (0, …, 0) y clase 1 en
    (π, …, π), con ángulos gaussianos independientes de desvío `sigma_clase`
    (envueltos). La carta es de velocidad unitaria, así que muestrear los ángulos
    muestrea la variedad sin jacobianos. La frontera de Bayes pasa por el valle entre
    las lomas. n balanceado, ruido σ = 0.05.
    """
    rng = np.random.default_rng(random_state)
    n0, n1 = _balance(n_samples, 2)
    angulos_0 = rng.normal(0.0, sigma_clase, size=(n0, d))
    angulos_1 = rng.normal(np.pi, sigma_clase, size=(n1, d))
    bloques = [parametrizar_toro(angulos_0), parametrizar_toro(angulos_1)]
    return _ensamblar(rng, bloques, noise)


# ---------------------------------------------------------------------------
# 5. Esfera S^d ⊂ R^{d+1} con clases von Mises-Fisher antipodales
# ---------------------------------------------------------------------------


def parametrizar_esfera(angulos):
    """Ángulos hiperesféricos (ϑ, ω₂, …, ω_d) ↦ S^d ⊂ R^{d+1}.

    ϑ ∈ [0, π] es el ángulo polar desde el polo norte e_{d+1}; ω₂, …, ω_{d-1} ∈ [0, π]
    y ω_d ∈ (-π, π] recorren las coordenadas restantes en el orden usual.
    """
    n, d = angulos.shape
    senos = np.cumprod(np.column_stack([np.ones(n), np.sin(angulos)]), axis=1)
    y = np.empty((n, d + 1))
    y[:, :d] = senos[:, :d] * np.cos(angulos)
    y[:, d] = senos[:, d]
    return np.column_stack([y[:, 1:], y[:, 0]])


def theta_esfera(X):
    """θ = (ϑ, ω₂, …, ω_d), ángulos hiperesféricos de x/‖x‖ (inversa de la carta).

    ϑ es la distancia geodésica al polo norte, centro de la clase 0: la frontera de
    Bayes es el ecuador ϑ = π/2.
    """
    y = np.column_stack([X[:, -1], X[:, :-1]])
    colas = np.sqrt(np.cumsum(y[:, ::-1] ** 2, axis=1)[:, ::-1])
    angulos = []
    for j in range(y.shape[1] - 2):
        coseno = np.divide(
            y[:, j], colas[:, j], out=np.ones(len(y)), where=colas[:, j] > 0
        )
        angulos.append(np.arccos(np.clip(coseno, -1.0, 1.0)))
    angulos.append(np.arctan2(y[:, -1], y[:, -2]))
    return np.column_stack(angulos)


def _von_mises_fisher(rng, mu, kappa, n):
    """Muestra vMF(μ, κ) sobre S^{p-1} ⊂ R^p con el algoritmo de Wood (1994).

    Rechazo exacto del coseno w = <x, μ> (envolvente Beta) más una dirección tangente
    uniforme.
    """
    mu = np.asarray(mu, float)
    p = mu.size
    d = p - 1
    b = d / (2 * kappa + np.sqrt(4 * kappa**2 + d**2))
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + d * np.log(1 - x0**2)
    ws = []
    faltan = n
    while faltan > 0:
        m = max(2 * faltan, 64)
        z = rng.beta(d / 2, d / 2, size=m)
        w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
        u = rng.uniform(size=m)
        acepta = kappa * w + d * np.log(1 - x0 * w) - c >= np.log(u)
        ws.append(w[acepta])
        faltan -= int(acepta.sum())
    w = np.concatenate(ws)[:n]
    v = rng.normal(size=(n, p))
    v -= np.outer(v @ mu, mu)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return w[:, None] * mu + np.sqrt(1 - w**2)[:, None] * v


@cache
def _kappa_para_error_bayes(d, error):
    """Concentración κ de dos vMF antipodales sobre S^d con error de Bayes `error`.

    Con clases vMF(±μ, κ) equiprobables, la frontera de Bayes es el ecuador
    {<x, μ> = 0} y el error es P(t < 0), con t = <x, μ> de densidad marginal
    ∝ exp(κ t) (1 - t²)^{(d-2)/2} en [-1, 1]. Se resuelve κ numéricamente.
    """

    def _error(kappa):
        # e^{κ(t-1)} por estabilidad numérica (el factor constante se cancela)
        def densidad(t):
            return np.exp(kappa * (t - 1)) * (1 - t**2) ** ((d - 2) / 2)

        numerador, _ = quad(densidad, -1, 0)
        denominador, _ = quad(densidad, -1, 1)
        return numerador / denominador - error

    return brentq(_error, 1e-3, 300.0)


def hacer_esfera_vmf(
    n_samples=800, d=2, error_bayes=0.05, kappa=None, noise=0.05, random_state=None
):
    """Dos casquetes von Mises-Fisher antipodales sobre la esfera unitaria S^d.

    Variedad: S^d ⊂ R^{d+1}, curvatura constante +1. La etiqueta es el casquete de
    origen: vMF(+e_{d+1}, κ) para la clase 0 y vMF(-e_{d+1}, κ) para la 1,
    equiprobables. La frontera de Bayes es el ecuador; si `kappa` es None se calibra
    para que el error de Bayes (sin ruido) sea `error_bayes`, de modo que el
    solapamiento entre clases no cambia con d. n balanceado, ruido σ = 0.05.
    """
    rng = np.random.default_rng(random_state)
    if kappa is None:
        kappa = _kappa_para_error_bayes(d, error_bayes)
    mu = np.zeros(d + 1)
    mu[-1] = 1.0
    n0, n1 = _balance(n_samples, 2)
    bloques = [
        _von_mises_fisher(rng, mu, kappa, n0),
        _von_mises_fisher(rng, -mu, kappa, n1),
    ]
    return _ensamblar(rng, bloques, noise)


# ---------------------------------------------------------------------------
# Registro
# ---------------------------------------------------------------------------

curvas = {"nudos": curvas_nudos(), "lissajous": curvas_lissajous()}

thetas: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "paralelas": theta_paralelas,
    "nudos": theta_nudos,
    "lissajous": theta_lissajous,
    "toro": theta_toro,
    "esfera": theta_esfera,
}


def _variedad(fabrica, dim_intrinseca, nombre_v6, theta, mecanismo, **params):
    """Entrada del registro: fábrica y sus parámetros, d, nombre en v6 y descripción.

    `theta` dice si la inversa de la carta (`thetas[familia]`) es de forma "cerrada" o
    "numérica"; `mecanismo` describe en una línea cómo se asigna la etiqueta. Los
    parámetros no fijados toman el valor por omisión de la fábrica y quedan explícitos,
    para que `Dataset.params` los registre todos.
    """
    por_omision = {
        nombre: parametro.default
        for nombre, parametro in signature(fabrica).parameters.items()
        if nombre != "random_state"
    }
    return Bunch(
        fabrica=fabrica,
        d=dim_intrinseca,
        nombre_v6=nombre_v6,
        theta=theta,
        mecanismo=mecanismo,
        params={**por_omision, **params},
    )


def _mecanismo_paralelas(m_brecha, n_samples=800):
    delta = m_brecha * TAU / (n_samples / 2)
    return (
        f"círculo de origen; paralelos a brecha δ = {m_brecha}·2π/(n/2) ≈ {delta:.4f}"
    )


_mecanismo_nudos = "nudo de origen: trébol (2,3) o nudo (3,2), enlazados"
_mecanismo_esfera = (
    "casquete de origen: vMF en el polo norte o sur; error de Bayes 0.05"
)
variedades = {
    **{
        f"paralelas_{m}": _variedad(
            hacer_paralelas,
            1,
            f"paralelas_{m}",
            "cerrada",
            _mecanismo_paralelas(m),
            n_samples=800,
            noise=0.0039,
            m_brecha=m,
        )
        for m in (2, 4, 8)
    },
    "nudos_200": _variedad(
        hacer_nudos, 1, "nudos_ralo", "numérica", _mecanismo_nudos, n_samples=200
    ),
    "nudos_800": _variedad(
        hacer_nudos, 1, "nudos", "numérica", _mecanismo_nudos, n_samples=800
    ),
    "lissajous": _variedad(
        hacer_lissajous,
        1,
        "lissajous",
        "numérica",
        "curva de origen: Lissajous de frecuencias (1,…,5) o (5,…,1)",
        n_samples=800,
    ),
    "toro_3": _variedad(
        hacer_toro,
        3,
        "toro3",
        "cerrada",
        "loma de origen: gaussiana envuelta centrada en (0,0,0) o en (π,π,π)",
        n_samples=800,
        d=3,
        sigma_clase=0.8,
    ),
    **{
        f"esfera_{d}": _variedad(
            hacer_esfera_vmf,
            d,
            f"esfera_{d}",
            "cerrada",
            _mecanismo_esfera,
            n_samples=800,
            d=d,
            error_bayes=0.05,
        )
        for d in (2, 4)
    },
}


def familia(nombre: str) -> str:
    """Familia de un dataset de `variedades` (p. ej., `nudos_200` → `nudos`)."""
    return nombre.split("-")[0].split("_")[0]


assert all(familia(nombre) in thetas for nombre in variedades)


# ---------------------------------------------------------------------------
# Generación y diagnóstico
# ---------------------------------------------------------------------------

app = typer.Typer(add_completion=False)


@app.command()
def generar(
    repeticiones: int = config.repeticiones,
    semilla_principal: int = config.semilla_principal,
    dir_datos: Path = dir_fundacionales / "datasets",
    archivo_resumen: Path = dir_fundacionales / "datasets-nuevos.csv",
    nombres: list[str] | None = None,
):
    """Genera `{nombre}-{semilla}.pkl` por dataset y semilla, y una tabla resumen.

    Las semillas son `config._obtener_semillas(semilla_principal, repeticiones)`, las
    mismas que usa `config.hacer_configuraciones` (con 25, las de la tesis; los
    conjuntos de 25, 64 y 100 semillas están anidados). Las configuraciones deben
    generarse con las mismas `repeticiones`, o las semillas de dataset que no estén en
    su lista se tratarán como datasets reales.
    """
    logging.basicConfig(level=logging.INFO)
    dir_datos.mkdir(parents=True, exist_ok=True)
    semillas = config._obtener_semillas(semilla_principal, repeticiones)
    filas = []
    for nombre in nombres or list(variedades):
        variedad = variedades[nombre]
        conteos = []
        for semilla in semillas:
            ds = Dataset.de_fabrica(
                variedad.fabrica, nombre=nombre, random_state=semilla, **variedad.params
            )
            ds.guardar(dir_datos / f"{nombre}-{semilla}.pkl")
            conteos.append(np.unique(ds.y, return_counts=True)[1])
        logger.info("%s: %d semillas en %s", nombre, len(semillas), dir_datos)
        extra = {
            clave: valor
            for clave, valor in variedad.params.items()
            if clave not in ("n_samples", "noise") and valor is not None
        }
        filas.append(
            {
                "nombre": nombre,
                "familia": familia(nombre),
                "nombre_v6": variedad.nombre_v6,
                "n": ds.n,
                "D": ds.p,
                "d": variedad.d,
                "K": ds.k,
                "ruido": variedad.params["noise"],
                "parametros": " ".join(f"{k}={v}" for k, v in extra.items()),
                "n_por_clase": "/".join(map(str, np.unique(np.vstack(conteos)))),
                "theta": variedad.theta,
                "semillas": len(semillas),
                "mecanismo": variedad.mecanismo,
            }
        )
    resumen = pd.DataFrame(filas)
    resumen.to_csv(archivo_resumen, index=False)
    logger.info("Escribió %s", archivo_resumen)


# Columnas de θ que se grafican por familia, con sus rótulos
_ejes_theta = {
    "paralelas": ((0, 1), ("t", "z")),
    "nudos": ((1, 3), ("d₀ (distancia al nudo 0)", "d₁ (distancia al nudo 1)")),
    "lissajous": (
        (1, 3),
        ("d₀ (distancia a la curva 0)", "d₁ (distancia a la curva 1)"),
    ),
    "toro": ((0, 1), ("φ₁", "φ₂")),
    "esfera": ((0, -1), ("ϑ (ángulo polar)", "ω_d")),
}
_colores = ("#2a78d6", "#eb6834")


@app.command()
def graficar(
    dir_datos: Path = dir_fundacionales / "datasets",
    dir_img: Path = dir_fundacionales / "img",
    nombres: list[str] | None = None,
):
    """Por dataset (primera semilla): proyección 3D de X y dispersión de θ, en PNG.

    La proyección es X mismo si D = 3 y sus tres primeras componentes principales si
    D > 3.
    """
    logging.basicConfig(level=logging.INFO)
    dir_img.mkdir(parents=True, exist_ok=True)
    for nombre in nombres or list(variedades):
        ruta = sorted(dir_datos.glob(f"{nombre}-*.pkl"))[0]
        ds = Dataset.cargar(ruta)
        proyeccion = ds.X if ds.p == 3 else PCA(3).fit_transform(ds.X)
        (i, j), (rotulo_i, rotulo_j) = _ejes_theta[familia(nombre)]
        theta = thetas[familia(nombre)](ds.X)
        figura = plt.figure(figsize=(11, 5))
        eje_3d = figura.add_subplot(1, 2, 1, projection="3d")
        eje_theta = figura.add_subplot(1, 2, 2)
        for etiqueta, color in zip(ds.etiquetas, _colores, strict=False):
            filas = ds.y == etiqueta
            eje_3d.scatter(*proyeccion[filas].T, s=4, color=color, label=etiqueta)
            eje_theta.scatter(
                theta[filas, i], theta[filas, j], s=4, color=color, label=etiqueta
            )
        eje_3d.set_title("X" if ds.p == 3 else f"X (PCA 3 de {ds.p})")
        eje_theta.set(title="θ", xlabel=rotulo_i, ylabel=rotulo_j)
        eje_theta.legend(title="Clase", markerscale=3)
        figura.suptitle(f"{ruta.stem}: n={ds.n}, D={ds.p}, d={variedades[nombre].d}")
        figura.tight_layout()
        figura.savefig(dir_img / f"{nombre}.png", dpi=110)
        plt.close(figura)
        logger.info("Graficó %s", nombre)


if __name__ == "__main__":
    app()
