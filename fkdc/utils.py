"""
Funciones auxiliares varias.
`_longitud_elipse`, `anteojos` y `arco` son autoría del Ing. Diego Battochio.
"""

import hashlib

import numpy as np
import pandas as pd
import yaml
from scipy.special import ellipe
from scipy.stats import truncnorm

# Usados para dumpear a yaml legible para humanos objetos de numpy
yaml.add_representer(np.int64, lambda dumper, num: dumper.represent_int(num))
yaml.add_representer(np.float64, lambda dumper, num: dumper.represent_float(num))
yaml.add_representer(np.ndarray, lambda dumper, array: dumper.represent_list(array))


def iqr(X):
    """Rango intercuartílico."""
    return np.percentile(X, 75) - np.percentile(X, 25)


def h_piloto(dists):
    """Ancho de banda piloto según regla de Silverman."""
    return 0.9 * np.minimum(dists.std(), iqr(dists) / 1.34) * len(dists) ** (-1 / 5)


def _longitud_elipse(d1, d2) -> float:
    """Longitud de una elipse con diámetros d1 y d2."""
    r1 = d1 / 2
    r2 = d2 / 2
    a_elipse = max(r1, r2)
    b_elipse = min(r1, r2)
    e_elipse = 1.0 - b_elipse**2 / a_elipse**2
    return 4 * a_elipse * ellipe(e_elipse)


def anteojos(
    centro=(0, 0),
    r=1,
    separacion=3,
    n=500,
    altura_puente=0.2,
    excluir_theta=None,
    random_state=None,
):
    """Genera puntos en forma de anteojos (dos arcos con puente)."""
    rng = np.random.default_rng(random_state)
    excluir_theta = excluir_theta or np.arcsin(altura_puente / r)
    angulo_efectivo = np.pi - excluir_theta

    c_x1 = centro[0] - separacion / 2
    c_x2 = centro[0] + separacion / 2
    c_y = centro[1]

    desplazamiento_tunel = r * np.cos(excluir_theta)
    diametro_tunel = separacion - 2 * desplazamiento_tunel
    longitud_tunel = _longitud_elipse(altura_puente, diametro_tunel) / 2

    longitud_arco = 2 * angulo_efectivo * r
    longitud_total = 2 * longitud_arco + 2 * longitud_tunel

    n_tunel = round(longitud_tunel / longitud_total * n)
    n_arco = round(longitud_arco / longitud_total * n)
    n_resto = n - 2 * n_tunel - 2 * n_arco

    circulo1 = arco(
        centro=(c_x1, c_y),
        r=r,
        n=n_arco + n_resto,
        angulo_max_abs=angulo_efectivo,
        desplazamiento_angulo=np.pi,
        random_state=rng,
    )
    circulo2 = arco(
        centro=(c_x2, c_y),
        r=r,
        n=n_arco,
        angulo_max_abs=angulo_efectivo,
        random_state=rng,
    )

    tunel_c_x = c_x1 + desplazamiento_tunel + diametro_tunel / 2
    tunel_superior, tunel_inferior = (
        arco(
            centro=(tunel_c_x, c_y + signo * altura_puente),
            r=(diametro_tunel / 2, altura_puente / 2),
            n=n_tunel,
            angulo_max_abs=np.pi / 2,
            desplazamiento_angulo=signo * np.pi / 2,
            random_state=rng,
        )
        for signo in (+1, -1)
    )

    return np.vstack((circulo1, circulo2, tunel_superior, tunel_inferior))


def arco(
    centro=(0, 0),
    r=1,
    n=500,
    muestreo="uniform",
    angulo_max_abs=np.pi,
    desplazamiento_angulo=0,
    random_state=None,
):
    """Genera puntos sobre un arco de circunferencia o elipse."""
    rng = np.random.default_rng(random_state)
    if muestreo == "uniform":
        theta = rng.uniform(-angulo_max_abs, angulo_max_abs, n)
    elif muestreo == "normal":
        desv_angulo = angulo_max_abs / 1.50
        theta = truncnorm(
            -angulo_max_abs, angulo_max_abs, loc=0, scale=desv_angulo
        ).rvs(size=n, random_state=rng)
    else:
        raise ValueError("`muestreo` debe ser 'uniform' o 'normal'")
    r = [r] if not hasattr(r, "__getitem__") else r
    x = centro[0] + r[0] * np.cos(theta - desplazamiento_angulo)
    y = centro[1] + r[len(r) - 1] * np.sin(theta - desplazamiento_angulo)
    return np.column_stack((x, y))


def muestra(*arrays, n_muestras, random_state=None):
    """Submuestreo aleatorio de arrays con igual dimensión 0."""
    rng = np.random.default_rng(random_state)
    n_arrays = arrays[0].shape[0]
    assert all(
        array.shape[0] == n_arrays for array in arrays
    ), "Todo elemento en *arrays debe tener igual dimensión 0 ('n')."
    if isinstance(n_muestras, float):
        assert 0 <= n_muestras <= 1, "El ratio debe estar entre 0 y 1"
        n_muestras = int(n_arrays * n_muestras)
    if isinstance(n_muestras, int):
        assert (
            0 <= n_muestras <= n_arrays
        ), "El nro de muestras debe estar entre 0 y la longitud de los arrays"
    idxs = rng.choice(range(n_arrays), size=n_muestras, replace=False)
    return [array[idxs] for array in arrays]


def hash_dict(D, largo=16):
    """Hash MD5 truncado de un diccionario."""
    return hashlib.md5(((k, v) for k, v in D.items())).hexdigest()[:largo]


def reajustar_parsimoniosamente(cv_results_: dict, ratio_std: float = 1) -> int:
    """Reajuste parsimonioso: elige el modelo más simple dentro de 1 std del mejor."""
    regularizar_ascendente = [
        ("param_alpha", True),
        ("param_bandwidth", False),
        ("param_n_neighbors", False),
        ("param_C", True),
        ("param_logreg__C", True),
        ("param_var_smoothing", False),
        ("param_max_depth", True),
    ]
    resultados = pd.DataFrame(cv_results_)
    puntaje_max = resultados.mean_test_score.max()
    std_max = resultados[
        resultados.mean_test_score.eq(puntaje_max)
    ].std_test_score.min()
    umbral_puntaje = puntaje_max - ratio_std * std_max
    resultados = resultados[resultados.mean_test_score.ge(umbral_puntaje)]
    regularizadores = [
        reg for reg in regularizar_ascendente if reg[0] in resultados.columns
    ]
    by, ascending = map(list, zip(*regularizadores, strict=True))
    return resultados.sort_values(by=by, ascending=ascending).index[0]


# Alias para compatibilidad con pickles anteriores a la traducción al español
refit_parsimoniously = reajustar_parsimoniosamente
