"""Nube de puntos de fondo para la invitación: `lunas.svg`, transparente."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from sklearn.datasets import make_moons

out_dir = Path(sys.argv[1])
X, y = make_moons(n_samples=800, noise=0.15, random_state=7)
# La luna de arriba va en el amarillo de los globos; la de abajo, en azul tenue
estilo = [("#FBD979", 0.35), ("#4A90D9", 0.09)]
fig, ax = plt.subplots(figsize=(3, 3))
ax.scatter(X[:, 0], X[:, 1], c=[to_rgba(*estilo[k]) for k in y], s=12, linewidths=0)
ax.set_aspect("equal")
ax.axis("off")
fig.savefig(
    out_dir / "lunas.svg", transparent=True, bbox_inches="tight", pad_inches=0.02
)
plt.close(fig)
