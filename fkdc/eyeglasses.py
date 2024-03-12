import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import ellipe
from scipy.stats import truncnorm

plt.style.use("seaborn-v0_8")


def plot_dataset(X, title="Conjunto de datos", ax=None):
    df = pd.DataFrame(X, columns=["x", "y"])
    if ax is None:
        ax = plt.gca()
    sns.kdeplot(data=df, x="x", y="y", fill=True, cut=1, cmap="Blues", ax=ax)
    ax.scatter(x=df["x"], y=df["y"], alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


def _ellipse_length(d1, d2) -> float:
    r1 = d1 / 2
    r2 = d2 / 2
    a_ellipse = max(r1, r2)
    b_ellipse = min(r1, r2)
    e_ellipse = 1.0 - b_ellipse**2 / a_ellipse**2
    return 4 * a_ellipse * ellipe(e_ellipse)


def eyeglasses(center=(0, 0), r=1, separation=3, n=500, bridge_height=0.2, exclude_theta=None):
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
    )
    circle2 = arc(center=(c_x2, c_y), r=r, n=n_arc, max_abs_angle=effective_angle)

    tunnel_c_x = c_x1 + tunnel_offset + tunnel_diameter / 2
    top_tunnel = arc(
        center=(tunnel_c_x, c_y + bridge_height),
        r=(tunnel_diameter / 2, bridge_height / 2),
        n=n_tunnel,
        max_abs_angle=np.pi / 2,
        angle_shift=np.pi / 2,
    )
    bottom_tunnel = arc(
        center=(tunnel_c_x, c_y - bridge_height),
        r=(tunnel_diameter / 2, bridge_height / 2),
        n=n_tunnel,
        max_abs_angle=np.pi / 2,
        angle_shift=-np.pi / 2,
    )

    return np.vstack((circle1, circle2, top_tunnel, bottom_tunnel))


def arc(center=(0, 0), r=1, n=500, sampling="uniform", max_abs_angle=np.pi, angle_shift=0):
    if sampling == "uniform":
        theta = np.random.uniform(-max_abs_angle, max_abs_angle, n)
    elif sampling == "normal":
        angle_sd = max_abs_angle / 1.50
        theta = truncnorm.rvs(-max_abs_angle, max_abs_angle, loc=0, scale=angle_sd, size=n)
    else:
        raise ValueError("Sampling should be either 'uniform' or 'normal'")
    r = [r] if not hasattr(r, "__getitem__") else r
    x = center[0] + r[0] * np.cos(theta - angle_shift)
    y = center[1] + r[len(r) - 1] * np.sin(theta - angle_shift)
    return np.column_stack((x, y))


def filled_circle(center=(0, 0), max_r=1, r_power=4, n=500):
    theta = np.random.uniform(-np.pi, np.pi, n)
    r = np.random.uniform(0, 1, n) ** (1 / r_power) * max_r
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.column_stack((x, y))


def rectangle(n: int, length_x: float, length_y: float):
    cum_lengths = np.cumsum([length_y, length_x, length_y, length_x])
    samples = np.random.uniform(0, max(cum_lengths), size=n)
    result_samples = np.empty(shape=(n, 2))
    for i, x in enumerate(samples):
        if x < cum_lengths[0]:
            result_samples[i] = [0, x]
        elif x < cum_lengths[1]:
            result_samples[i] = [x - cum_lengths[0], 0]
        elif x < cum_lengths[2]:
            result_samples[i] = [length_x, x - cum_lengths[1]]
        elif x < cum_lengths[3]:
            result_samples[i] = [x - cum_lengths[2], length_y]
    return pd.DataFrame(result_samples, columns=["x", "y"])


def football_sensor(n: int, tag_id: int, second_half: bool = False):
    root_path = pathlib.Path(__file__).parent.parent.resolve()
    df = pd.read_csv(f"{root_path}/datasets/2013-11-03_tromso_stromsgodset_raw_first.csv")
    if second_half:
        second_df = pd.read_csv("../datasets/2013-11-03_tromso_stromsgodset_raw_second.csv")
        df = pd.concat([df, second_df])
    df.columns = [
        "timestamp",
        "tag_id",
        "x",
        "y",
        "heading",
        "direction",
        "energy",
        "speed",
        "total_distance",
    ]
    df.query("tag_id == @tag_id", inplace=True)
    return df[["x", "y"]].sample(n)


def add_noise(X, sd=1):
    n, d = X.shape
    noise = np.random.normal(0, sd, (n, d))
    return X + noise


def add_outliers(X, frac=0.05, iqr_factor=1.5):
    n, d = X.shape
    amount = int(round(frac * n))
    qs = np.percentile(X, [25, 75], axis=0)
    iqr = qs[1] - qs[0]
    iqr_sign = np.random.choice([-1, 1], amount)
    ixs = np.random.choice(n, amount, replace=True)
    X[ixs, :] = X[ixs, :] + iqr_factor * iqr_sign[:, np.newaxis] * np.tile(iqr, (amount, 1))
    return X


def add_dummy_dimensions(X, d=1):
    n, _ = X.shape
    new_columns = np.random.normal(0, 1, (n, d))
    return np.hstack((X, new_columns))
