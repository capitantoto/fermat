"""
Clasificadores "oráculo": los de la tesis sobre coordenadas intrínsecas.

Para cada familia sintética de `fkdc.datasets` se define un mapa de atributos φ que
invierte el proceso generador conocido y devuelve las coordenadas intrínsecas que
distinguen a las clases. φ es una función fija del código generador y de los
parámetros con que lo invoca `hacer_datasets`: nunca se ajusta a los datos, se
aplica igual a las variantes con más o menos ruido (el ruido se conserva) y descarta
las dimensiones de ruido puro que `agregar_dims_ruido` agrega al final en las
variantes `_12`.

Criterio común: bajo el modelo generador (ruido gaussiano isótropo de desvío σ),
φ se elige de modo que el logaritmo del cociente de probabilidades a posteriori de
Bayes ("log-odds") sea (aproximadamente) afín en φ. Así, la regresión logística
sobre φ puede alcanzar el clasificador de Bayes y sirve de cota de referencia para
un clasificador que *conoce* la variedad. Los demás clasificadores de la tesis se
pueden montar sobre el mismo φ; φ no es una carta isométrica y sus componentes no
comparten escala, así que los métodos basados en distancias (kdc, kn, svc) ven
una geometría distinta de la del espacio ambiente.
"""

from functools import cache
from math import floor

import numpy as np
from scipy.spatial import cKDTree
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from fkdc import config

# Parámetros de los generadores tal como los usa `fkdc.datasets.hacer_datasets`
# (valores por omisión de cada fábrica; `noise` no interviene en φ).
FACTOR_CIRCULOS = 0.8  # `make_circles(factor=0.8)`: radio de la clase 1
A_ESPIRALES = 1  # `hacer_espirales(a=1)`
CENTRO_HELICES, RADIO_HELICES, ALTURA_HELICES = (0, 0), 1, 1  # `hacer_helices`
CENTRO_ESLABONES, RADIO_ESLABONES = (0, 0), 1  # `hacer_eslabones`
ESCALA_Y_PIONONO = 21  # `hacer_pionono`: y = 21 · s₁
SEPARACION_ANTEOJOS = 3  # `hacer_anteojos(separacion=3)`
RADIO_ANTEOJOS, ALTURA_PUENTE_ANTEOJOS = 1, 0.2  # `fkdc.utils.anteojos`


def _polares(X):
    """Radio y ángulo en (-π, π] de las dos primeras columnas de X."""
    return np.hypot(X[:, 0], X[:, 1]), np.arctan2(X[:, 1], X[:, 0])


def _dist2_arco(X, centro, radio, angulo_ini, angulo_fin):
    """Distancia al cuadrado de cada fila de X (2D) a un arco de circunferencia.

    Si el ángulo del punto cae en el arco, el punto más cercano es su proyección
    radial; si no, el extremo más cercano.
    """
    centro = np.asarray(centro, dtype=float)
    rho, theta = _polares(X - centro)
    dentro = np.mod(theta - angulo_ini, 2 * np.pi) <= angulo_fin - angulo_ini
    extremos = [
        centro + radio * np.array([np.cos(angulo), np.sin(angulo)])
        for angulo in (angulo_ini, angulo_fin)
    ]
    d2_extremos = np.min([((X - e) ** 2).sum(axis=1) for e in extremos], axis=0)
    return np.where(dentro, (rho - radio) ** 2, d2_extremos)


def _dist2_anillo(X, centro, normal, radio):
    """Distancia al cuadrado de cada fila de X (3D) a una circunferencia.

    En coordenadas toroidales, con h la altura sobre el plano del anillo y ρ la
    distancia al eje, d² = h² + (ρ - radio)².
    """
    D = X - np.asarray(centro, dtype=float)
    normal = np.asarray(normal, dtype=float)
    h = D @ normal
    rho = np.linalg.norm(D - np.outer(h, normal), axis=1)
    return h**2 + (rho - radio) ** 2


def phi_lunas(X):
    """φ = (d₀², d₁²), distancias al cuadrado a cada semicircunferencia de `make_moons`.

    Clase 0: centro (0, 0), mitad superior; clase 1: centro (1, 0.5), mitad inferior.
    Proyectar sobre el arco invierte el generador y d_k es la norma del residuo; con
    ruido gaussiano log p(x|k) ≈ -d_k²/2σ² + cte (aproximación de tubo, más gruesa
    con σ = 0.5), así que el log-odds es ≈ afín en φ.
    """
    X = X[:, :2]
    return np.column_stack(
        [
            _dist2_arco(X, (0, 0), 1, 0, np.pi),
            _dist2_arco(X, (1, 0.5), 1, np.pi, 2 * np.pi),
        ]
    )


def phi_circulos(X):
    """φ = ‖x‖, el radio.

    El ángulo es uniforme y común a ambas clases, y el radio (1 en la clase 0,
    `FACTOR_CIRCULOS` en la 1) es estadístico suficiente; con ruido gaussiano el
    log-odds es ≈ afín en ‖x‖ (asintótica de Bessel de la densidad de Rice).
    """
    return np.hypot(X[:, 0], X[:, 1])[:, None]


def phi_espirales(X):
    """φ = (cos ψ, sin ψ), con ψ = ‖x‖²/a² - θ la fase respecto de la espiral 0.

    El generador fija ψ ≡ 0 (mód 2π) en la clase 0 y ψ ≡ π en la 1, rotada media
    vuelta; con ruido, el log-odds es ≈ afín en cos ψ (von Mises). Aproximación: el
    ruido isótropo le da a ψ varianza σ²(4r²/a⁴ + 1/r²), que crece con el radio.
    """
    r, theta = _polares(X[:, :2])
    psi = (r / A_ESPIRALES) ** 2 - theta
    return np.column_stack([np.cos(psi), np.sin(psi)])


def phi_helices(X):
    """φ = (cos ψ, sin ψ), con ψ = θ - 2πz/altura la fase respecto de la hélice 0.

    El generador fija ψ ≡ 0 (mód 2π) en la clase 0 y ψ ≡ π en la 1, rotada 180° sobre
    el eje Z. Sobre el cilindro de radio fijo |∇ψ|² es constante, así que el ruido de
    ψ es homoscedástico y el log-odds (≈ von Mises) es afín en cos ψ.
    """
    theta = np.arctan2(X[:, 1] - CENTRO_HELICES[1], X[:, 0] - CENTRO_HELICES[0])
    psi = theta - 2 * np.pi * X[:, 2] / ALTURA_HELICES
    return np.column_stack([np.cos(psi), np.sin(psi)])


def phi_eslabones(X):
    """φ = (d₀², d₁²), distancias al cuadrado a cada anillo generador.

    Clase 0: centro (0, 0, 0) en el plano z = 0; clase 1: centro (radio, 0, 0) en el
    plano y = 0. d_k es la norma del residuo del generador y, como ambos anillos
    tienen igual longitud, log-odds ≈ (d₁² - d₀²)/2σ², afín en φ.
    """
    cx, cy = CENTRO_ESLABONES
    return np.column_stack(
        [
            _dist2_anillo(X[:, :3], (cx, cy, 0), (0, 0, 1), RADIO_ESLABONES),
            _dist2_anillo(
                X[:, :3], (cx + RADIO_ESLABONES, 0, cy), (0, 1, 0), RADIO_ESLABONES
            ),
        ]
    )


def phi_hueveras(X):
    """φ = z·g(x, y), con g(x, y) = sin x · sin y.

    La clase 0 es la superficie z = g y la 1 es z = -g; con ruido gaussiano en los
    residuos verticales z ∓ g, log-odds = [(z + g)² - (z - g)²]/2σ² = 2zg/σ², afín en
    φ. Aproximación: ignora el ruido en (x, y), que propagado por g infla la varianza
    del residuo a σ²(1 + |∇g|²). Sobre las líneas nodales (g = 0) ambas superficies se
    cortan y las clases son indistinguibles.
    """
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    return (z * np.sin(x) * np.sin(y))[:, None]


def phi_pionono(X):
    """φ = (s₀, s₁), las coordenadas intrínsecas del pionono.

    Invierte x = t cos t, z = t sin t, y = 21 s₁ con t = 2π(1 + 2 s₀): t es el ángulo
    de (x, z) desenrollado con su radio, ‖(x, z)‖ ≈ t. En (s₀, s₁) las clases son
    gaussianas isótropas de igual covarianza, así que el log-odds es afín en φ.
    """
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    rho, theta = np.hypot(x, z), np.mod(np.arctan2(z, x), 2 * np.pi)
    t = theta + 2 * np.pi * np.round((rho - theta) / (2 * np.pi))
    return np.column_stack([(t / (2 * np.pi) - 1) / 2, y / ESCALA_Y_PIONONO])


@cache
def _marco_anteojos(n_por_tramo: int = 20_000) -> cKDTree:
    """Discretización fina y determinista del marco de `fkdc.utils.anteojos`.

    Replica la geometría de `anteojos` (dos arcos de radio r y los dos medios arcos
    de elipse del puente) evaluando `arco` sobre una grilla regular de ángulos en
    lugar de ángulos aleatorios.
    """
    r, separacion, altura = RADIO_ANTEOJOS, SEPARACION_ANTEOJOS, ALTURA_PUENTE_ANTEOJOS
    excluir_theta = np.arcsin(altura / r)
    angulo_efectivo = np.pi - excluir_theta
    c_x1, c_x2 = -separacion / 2, separacion / 2
    desplazamiento_tunel = r * np.cos(excluir_theta)
    diametro_tunel = separacion - 2 * desplazamiento_tunel
    tunel_c_x = c_x1 + desplazamiento_tunel + diametro_tunel / 2

    def tramo(centro, radios, angulo_max_abs, desplazamiento):
        theta = np.linspace(-angulo_max_abs, angulo_max_abs, n_por_tramo)
        theta = theta - desplazamiento
        return np.column_stack(
            [
                centro[0] + radios[0] * np.cos(theta),
                centro[1] + radios[1] * np.sin(theta),
            ]
        )

    tramos = [
        tramo((c_x1, 0), (r, r), angulo_efectivo, np.pi),
        tramo((c_x2, 0), (r, r), angulo_efectivo, 0),
        *(
            tramo(
                (tunel_c_x, signo * altura),
                (diametro_tunel / 2, altura / 2),
                np.pi / 2,
                signo * np.pi / 2,
            )
            for signo in (+1, -1)
        ),
    ]
    return cKDTree(np.vstack(tramos))


def phi_anteojos(X):
    """φ = (d², x₁, x₁², x₂²), con d la distancia al marco de los anteojos.

    Marco (clase 0): d se obtiene proyectando numéricamente sobre las curvas de
    `fkdc.utils.anteojos`, y log p(x|0) ≈ -d²/2σ² + cte como en `lunas`. Pupilas
    (clases 1 y 2): gaussianas de desvíos (1/6, 1/3) centradas en (∓separación/2, 0),
    sin variedad que invertir; x₁, x₁² y x₂² son sus estadísticos suficientes y
    hacen afines sus log-densidades. Aproximación de tubo en el marco.
    """
    X = X[:, :2]
    d, _ = _marco_anteojos().query(X)
    return np.column_stack([d**2, X[:, 0], X[:, 0] ** 2, X[:, 1] ** 2])


mapas_oraculo = {
    "lunas": phi_lunas,
    "circulos": phi_circulos,
    "espirales": phi_espirales,
    "helices": phi_helices,
    "eslabones": phi_eslabones,
    "hueveras": phi_hueveras,
    "pionono": phi_pionono,
    "anteojos": phi_anteojos,
}


def familia(variante: str) -> str:
    """Familia sintética de una variante (p. ej. `helices_12` → `helices`)."""
    nombre = variante.split("_")[0]
    if nombre not in mapas_oraculo:
        raise ValueError(f"Sin oráculo para '{variante}': no es una familia sintética")
    return nombre


def clasificador_oraculo(
    variante: str,
    clf: str = "lr",
    n_muestras: int | None = None,
    cv: int = config.cv,
    split_evaluacion: float = config.split_evaluacion,
) -> tuple[Pipeline, dict]:
    """Clasificador `clf` de la tesis sobre φ para `variante`, listo para `Tarea`.

    Devuelve el par (clasificador, grilla) que espera `Tarea`: un `Pipeline` con
    φ (`FunctionTransformer`, sin parámetros ajustables) seguido del estimador
    `config.clasificadores[clf]`, y su grilla `config.grillas[clf]` con prefijo
    `clf__`. Si la grilla tiene `n_neighbors`, se filtra como en
    `config.hacer_configuraciones`, para lo que hace falta `n_muestras` (el tamaño
    del dataset completo).
    """
    pipeline = Pipeline(
        [
            ("oraculo", FunctionTransformer(mapas_oraculo[familia(variante)])),
            ("clf", clone(config.clasificadores[clf])),
        ]
    )
    grilla = dict(config.grillas[clf])
    if "n_neighbors" in grilla:
        if n_muestras is None:
            raise ValueError("Filtrar `n_neighbors` requiere `n_muestras`")
        n_muestras_ajuste = floor(n_muestras * (cv - 1) / cv * (1 - split_evaluacion))
        grilla["n_neighbors"] = [
            n for n in grilla["n_neighbors"] if n < n_muestras_ajuste
        ]
    return pipeline, {f"clf__{param}": valores for param, valores in grilla.items()}
