"""Funciones auxiliares varias. `_ellipse_length`, `eyeglasses` y `arc` son autoría del Ing. Diego Battochio."""

import hashlib
import pickle
from itertools import product
from pathlib import Path

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
    return np.percentile(X, 75) - np.percentile(X, 25)


def pilot_h(dists):
    return 0.9 * np.minimum(dists.std(), iqr(dists) / 1.34) * len(dists) ** (-1 / 5)


def lattice(a, b, step=1, dim=2, array=True):
    side = np.arange(a, b, step)
    if len(side) ** dim > 1e6:
        raise ValueError(
            f"Too many points ({len(side) ** dim:.2e} > 1e6). Try a bigger step or a smaller dim."
        )
    gen = product(*[side] * dim)
    return np.array([*gen]) if array else gen


def _ellipse_length(d1, d2) -> float:
    r1 = d1 / 2
    r2 = d2 / 2
    a_ellipse = max(r1, r2)
    b_ellipse = min(r1, r2)
    e_ellipse = 1.0 - b_ellipse**2 / a_ellipse**2
    return 4 * a_ellipse * ellipe(e_ellipse)


def eyeglasses(
    center=(0, 0),
    r=1,
    separation=3,
    n=500,
    bridge_height=0.2,
    exclude_theta=None,
    random_state=None,
):
    rng = np.random.default_rng(random_state)
    exclude_theta = exclude_theta or np.arcsin(bridge_height / r)
    effective_angle = np.pi - exclude_theta

    c_x1 = center[0] - separation / 2
    c_x2 = center[0] + separation / 2
    c_y = center[1]

    tunnel_offset = r * np.cos(exclude_theta)
    tunnel_diameter = separation - 2 * tunnel_offset
    tunnel_length = _ellipse_length(bridge_height, tunnel_diameter) / 2

    arc_length = 2 * effective_angle * r
    total_length = 2 * arc_length + 2 * tunnel_length

    n_tunnel = round(tunnel_length / total_length * n)
    n_arc = round(arc_length / total_length * n)
    n_reminder = n - 2 * n_tunnel - 2 * n_arc

    circle1 = arc(
        center=(c_x1, c_y),
        r=r,
        n=n_arc + n_reminder,
        max_abs_angle=effective_angle,
        angle_shift=np.pi,
        random_state=rng,
    )
    circle2 = arc(
        center=(c_x2, c_y),
        r=r,
        n=n_arc,
        max_abs_angle=effective_angle,
        random_state=rng,
    )

    tunnel_c_x = c_x1 + tunnel_offset + tunnel_diameter / 2
    top_tunnel, bottom_tunnel = (
        arc(
            center=(tunnel_c_x, c_y + sign * bridge_height),
            r=(tunnel_diameter / 2, bridge_height / 2),
            n=n_tunnel,
            max_abs_angle=np.pi / 2,
            angle_shift=sign * np.pi / 2,
            random_state=rng,
        )
        for sign in (+1, -1)
    )

    return np.vstack((circle1, circle2, top_tunnel, bottom_tunnel))


def arc(
    center=(0, 0),
    r=1,
    n=500,
    sampling="uniform",
    max_abs_angle=np.pi,
    angle_shift=0,
    random_state=None,
):
    rng = np.random.default_rng(random_state)
    if sampling == "uniform":
        theta = rng.uniform(-max_abs_angle, max_abs_angle, n)
    elif sampling == "normal":
        angle_sd = max_abs_angle / 1.50
        theta = truncnorm(-max_abs_angle, max_abs_angle, loc=0, scale=angle_sd).rvs(
            size=n, random_state=rng
        )
    else:
        raise ValueError("Sampling should be either 'uniform' or 'normal'")
    r = [r] if not hasattr(r, "__getitem__") else r
    x = center[0] + r[0] * np.cos(theta - angle_shift)
    y = center[1] + r[len(r) - 1] * np.sin(theta - angle_shift)
    return np.column_stack((x, y))


def sample(*arrays, n_samples, random_state=None):
    rng = np.random.default_rng(random_state)
    n_arrays = arrays[0].shape[0]
    assert all(
        array.shape[0] == n_arrays for array in arrays
    ), "Todo elemento en *arrays deben tener igual dimensión 0 ('n')."
    if isinstance(n_samples, float):
        assert (0 <= n_samples) and (n_samples <= 1), "El ratio debe estar entre 0 y 1"
        n_samples = int(n_arrays * n_samples)
    if isinstance(n_samples, int):
        assert (0 <= n_samples) and (
            n_samples <= n_arrays
        ), "El nro de muestras debe estar entre 0 y la longitud de los arrays"
    idxs = rng.choice(range(n_arrays), size=n_samples, replace=False)
    return [array[idxs] for array in arrays]


def dict_hasher(D, len=16):
    return hashlib.md5(((k, v) for k, v in D.items())).hexdigest()[:len]


def refit_parsimoniously(cv_results_: dict, std_ratio: float = 1) -> int:
    regularize_ascending = [
        ("param_alpha", True),
        ("param_bandwidth", False),
        ("param_n_neighbors", False),
        ("param_C", True),
        ("param_logreg__C", True),
        ("param_var_smoothing", False),
        ("param_max_depth", True),
    ]
    results = pd.DataFrame(cv_results_)
    top_score = results.mean_test_score.max()
    top_std = results[results.mean_test_score.eq(top_score)].std_test_score.min()
    score_threshold = top_score - std_ratio * top_std
    results = results[results.mean_test_score.ge(score_threshold)]
    regularizers = [reg for reg in regularize_ascending if reg[0] in results.columns]
    by, ascending = map(list, zip(*regularizers))
    return results.sort_values(by=by, ascending=ascending).index[0]


def load_infos(dir_or_paths: list[Path] | Path):
    if isinstance(dir_or_paths, Path):
        assert dir_or_paths.is_dir()
        paths = dir_or_paths.glob("*.pkl")
    elif isinstance(dir_or_paths, list):
        paths = dir_or_paths
    else:
        raise ValueError(
            "`dir_or_paths` debe ser un directorio con pickles de info o una lista de rutas a pickles de info"
        )
    return {tuple(fn.stem.split("-")): pickle.load(open(fn, "rb")) for fn in paths}


def parse_basic_info(infos: dict, main_seed: int | None = None):
    basic_fields = ["accuracy", "r2", "logvero"]
    basic_infos = {}
    for k, v in infos.items():
        clf = k[2]
        basic_infos[k] = {k: v for k, v in v[clf].items() if k in basic_fields}
        if clf == "fkdc":
            basic_infos[(k[0], k[1], "base", k[3], k[4])] = {
                k: v for k, v in v["base"].items() if k in basic_fields
            }

    basic_info = pd.DataFrame.from_records(
        list(basic_infos.values()),
        index=pd.MultiIndex.from_tuples(
            basic_infos.keys(), names=["dataset", "ds_seed", "clf", "run_seed", "score"]
        ),
    ).reset_index()
    if main_seed:  # Valida que las semillas se correspondan
        assert all(
            (basic_info.ds_seed == "None") | (basic_info.run_seed == str(main_seed))
        )
    basic_info["semilla"] = np.where(
        basic_info.ds_seed == "None", basic_info.run_seed, basic_info.ds_seed
    ).astype(int)
    return basic_info.drop(columns=["ds_seed", "run_seed"])
