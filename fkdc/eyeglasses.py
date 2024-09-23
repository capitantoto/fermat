"""Funciones auxiliares graciosamente donadas por Ing. Diego Battochio."""

import numpy as np
from scipy.special import ellipe
from scipy.stats import truncnorm


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
