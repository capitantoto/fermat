"""
Ilustraciones de la primera parte de la defensa.

- `cajita-d{1,2,3}.svg`: un segmento, un cuadrado y un cubo de lado 1 con la cajita
  centrada de lado h resaltada, para la maldición de la dimensionalidad.
- `paseos-espirales-{euclidea,fermat}.svg`: cien paseos al azar sobre el grafo completo
  de `espirales_lo`, desde un mismo punto, saltando a cada observación con probabilidad
  inversamente proporcional a su distancia. Cada salto de Fermat se dibuja por su
  camino mínimo. El euclídeo cambia de brazo seguido; el de Fermat, casi nunca.
- `resumen-{fkdc,fkn}.svg`: en los 20 datasets de la tesis, si el clasificador es
  mejor, igual o peor que su versión euclídea y que el mejor de los demás. Como en las
  tablas resumen de la tesis, A es peor que B si la mediana de A queda por debajo del
  primer cuartil de B; diferencias de medianas menores a `tolerancia` son empates.
- `anim/vecindario-espirales-NN.svg`: cuadros de una animación que `docs/defensa.typ`
  recorre uno por clic. Los k vecinos más cercanos de un mismo punto, según la
  distancia euclídea y según la de Fermat muestral, para k creciente. La euclídea
  cruza a la otra espiral; la de Fermat sigue el brazo.
- `densidades-espirales.svg`: la densidad estimada de cada clase y su solapamiento
  sobre una muestra nueva, con la distancia euclídea y con la de Fermat, cada una
  con su ancho de banda elegido por validación cruzada.

Todo se escribe en `docs/img/defensa/`. Uso, desde la raíz del repositorio:
    uv run python scripts/ilustraciones_defensa.py
"""

import logging
import sys

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse.csgraph import shortest_path  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402

from fkdc import config, dir_raiz  # noqa: E402
from fkdc.datasets import hacer_espirales  # noqa: E402
from fkdc.fermat import KDClassifier, euclidean  # noqa: E402

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger("ilustraciones_defensa")

dir_img = dir_raiz / "docs/img/defensa"
dir_anim = dir_img / "anim"
semilla = 1312
alfa_fermat = 3
vecinos = [1, 2, 3, 5, 8, 12, 18, 25, 35, 50, 70, 100]
tinta, regla = "#1b1f23", "#d9dcdf"
colores_clase = ["#2c7bb6", "#d7191c"]


def caminos_fermat(X, origen, alfa):
    """Distancias de Fermat muestrales desde `origen` y sus predecesores."""
    adyacencias = euclidean(X) ** alfa
    return shortest_path(
        adyacencias, directed=False, indices=origen, return_predecessors=True
    )


def panel(ax, X, y, origen, orden, k, titulo, predecesores=None):
    """Dibuja los k vecinos de `origen`; con predecesores, también sus caminos."""
    cercanos = orden[1 : k + 1]
    ax.scatter(*X.T, c=[colores_clase[c] for c in y], s=10, alpha=0.18, lw=0)
    if predecesores is not None:
        for j in cercanos:
            while predecesores[j] >= 0:
                i = predecesores[j]
                ax.plot(*X[[i, j]].T, color=tinta, lw=0.8, alpha=0.5, zorder=2)
                j = i
    ax.scatter(
        *X[cercanos].T,
        c=[colores_clase[c] for c in y[cercanos]],
        s=28,
        edgecolors=tinta,
        linewidths=0.6,
        zorder=3,
    )
    ax.scatter(*X[origen], marker="*", s=260, c="gold", edgecolors=tinta, zorder=4)
    intrusos = int((y[cercanos] != y[origen]).sum())
    ax.set_title(f"{titulo}: {k} vecinos, {intrusos} de la otra espiral", fontsize=12)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for borde in ax.spines.values():
        borde.set_color(regla)


def vecindario_espirales():
    # Los parámetros de `espirales_lo` en la tesis
    X, y = hacer_espirales(n_samples=config.n_muestras, noise=0.1, random_state=semilla)
    radios = np.linalg.norm(X, axis=1)
    # Un punto a media altura del brazo, donde las espirales están más cerca
    candidatos = np.flatnonzero(y == 0)
    origen = candidatos[np.argmin(np.abs(radios[candidatos] - np.median(radios)))]
    orden_euclideo = np.argsort(euclidean(X)[origen])
    distancias, predecesores = caminos_fermat(X, origen, alfa_fermat)
    orden_fermat = np.argsort(distancias)
    dir_anim.mkdir(parents=True, exist_ok=True)
    for cuadro, k in enumerate(vecinos):
        fig, (izq, der) = plt.subplots(1, 2, figsize=(11, 5.4))
        panel(izq, X, y, origen, orden_euclideo, k, "Euclídea")
        panel(der, X, y, origen, orden_fermat, k, "Fermat", predecesores)
        fig.tight_layout()
        ruta = dir_anim / f"vecindario-espirales-{cuadro:02d}.svg"
        fig.savefig(ruta, bbox_inches="tight")
        plt.close(fig)
        logger.info("Escrito %s", ruta.relative_to(dir_raiz))


def densidades_espirales():
    """Densidad de cada clase y su solapamiento, euclídea contra Fermat.

    Las densidades se evalúan sobre una muestra nueva, es decir, sobre la variedad.
    La validación cruzada de Fermat tiene dos modos: uno de casi un vecino (h ~ 1e-3),
    que en el plano es un manto de agujas, y uno suave (h ~ 1e-1); la grilla se limita
    al suave, que es el que ilustra la idea.
    """
    # Entrenamiento y evaluación de `espirales_lo`: 400 puntos cada una
    X, y = hacer_espirales(n_samples=400, noise=0.1, random_state=semilla)
    X_eval, _ = hacer_espirales(n_samples=400, noise=0.1, random_state=semilla + 1)
    candidatos = {
        "Euclídea": (KDClassifier(metric="euclidea_exacta"), np.logspace(-2, 0, 17)),
        "Fermat": (KDClassifier(metric="fermat", alpha=2), np.logspace(-1.5, -0.5, 9)),
    }
    fig, ejes = plt.subplots(2, 3, figsize=(13, 8.4))
    for fila, (nombre, (clf, anchos)) in enumerate(candidatos.items()):
        busqueda = GridSearchCV(
            clf, {"bandwidth": anchos}, scoring="neg_log_loss", cv=5
        ).fit(X, y)
        logger.info("%s: %s", nombre, busqueda.best_params_)
        logdens = [m.score_samples(X_eval) for m in busqueda.best_estimator_.models_]
        # Escala logarítmica relativa al máximo, con piso en e^-10
        relativas = [np.maximum(ld - ld.max(), -10) for ld in logdens]
        # Solapamiento: cociente entre la menor y la mayor densidad, en [0, 1]
        solapamiento = np.exp(-np.abs(logdens[0] - logdens[1]))
        paneles = [
            (relativas[0], "Blues", "densidad de la clase 0"),
            (relativas[1], "Reds", "densidad de la clase 1"),
            (solapamiento, "Purples", f"solapamiento medio {solapamiento.mean():.2f}"),
        ]
        for ax, (z, cmap, titulo) in zip(ejes[fila], paneles, strict=True):
            orden = np.argsort(z)  # los valores altos, encima
            ax.scatter(*X_eval[orden].T, c=z[orden], cmap=cmap, s=16, lw=0)
            ax.set_title(f"{nombre}: {titulo}", fontsize=13, color=tinta)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for borde in ax.spines.values():
                borde.set_color(regla)
    fig.tight_layout()
    ruta = dir_img / "densidades-espirales.svg"
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    logger.info("Escrito %s", ruta.relative_to(dir_raiz))


def guardar(fig, nombre):
    ruta = dir_img / nombre
    fig.savefig(ruta, bbox_inches="tight", transparent=True)
    plt.close(fig)
    logger.info("Escrito %s", ruta.relative_to(dir_raiz))


def cajitas(h=0.5):
    """La cajita de lado h dentro de la de lado 1, en una, dos y tres dimensiones."""
    resalte, contorno = "#e66101", tinta
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 0], color=contorno, lw=3, solid_capstyle="butt")
    ax.plot(
        [(1 - h) / 2, (1 + h) / 2], [0, 0], color=resalte, lw=9, solid_capstyle="butt"
    )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.5, 0.5)
    ax.set_axis_off()
    guardar(fig, "cajita-d1.svg")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec=contorno, lw=2))
    ax.add_patch(plt.Rectangle(((1 - h) / 2,) * 2, h, h, fc=resalte, ec=resalte))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_axis_off()
    guardar(fig, "cajita-d2.svg")

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    def caras(lo, hi):
        v = np.array([[x, y, z] for x in (lo, hi) for y in (lo, hi) for z in (lo, hi)])
        indices = [
            [0, 1, 3, 2],
            [4, 5, 7, 6],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [0, 2, 6, 4],
            [1, 3, 7, 5],
        ]
        return [v[i] for i in indices]

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(projection="3d")
    ax.add_collection3d(
        Poly3DCollection(
            caras((1 - h) / 2, (1 + h) / 2), fc=resalte, ec=tinta, lw=0.5, alpha=0.95
        )
    )
    ax.add_collection3d(
        Poly3DCollection(caras(0, 1), fc=(0.6, 0.6, 0.6, 0.08), ec=contorno, lw=1.2)
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-58)
    ax.set_axis_off()
    guardar(fig, "cajita-d3.svg")


def paseos_espirales(paseos=100, pasos=15, alfa=4):
    """Paseos al azar con saltos de probabilidad proporcional a 1 / distancia."""
    X, y = hacer_espirales(n_samples=config.n_muestras, noise=0.1, random_state=semilla)
    radios = np.linalg.norm(X, axis=1)
    candidatos = np.flatnonzero(y == 0)
    origen = candidatos[np.argmin(np.abs(radios[candidatos] - np.median(radios)))]
    euclidea = euclidean(X)
    fermat, predecesores = shortest_path(
        euclidea**alfa, directed=False, return_predecessors=True
    )
    for nombre, distancias in [("euclidea", euclidea), ("fermat", fermat)]:
        pesos = np.divide(
            1, distancias, out=np.zeros_like(distancias), where=distancias > 0
        )
        transicion = pesos / pesos.sum(1, keepdims=True)
        rng = np.random.default_rng(semilla)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(*X.T, c=[colores_clase[c] for c in y], s=10, alpha=0.35, lw=0)
        cambios = 0
        for _ in range(paseos):
            i = origen
            for _ in range(pasos):
                j = rng.choice(len(X), p=transicion[i])
                cambios += y[j] != y[i]
                tramo = [j]
                if nombre == "fermat":
                    while tramo[-1] != i:
                        tramo.append(predecesores[i, tramo[-1]])
                else:
                    tramo.append(i)
                ax.plot(*X[tramo].T, color=tinta, lw=0.6, alpha=0.06, zorder=2)
                i = j
        ax.scatter(*X[origen], marker="*", s=320, c="gold", edgecolors=tinta, zorder=4)
        logger.info("%s: %.1f cambios de brazo por paseo", nombre, cambios / paseos)
        ax.set_aspect("equal")
        ax.set_axis_off()
        guardar(fig, f"paseos-espirales-{nombre}.svg")


crudas_tesis = [
    "lunas_lo", "circulos_lo", "espirales_lo", "lunas_hi", "circulos_hi",
    "espirales_hi", "eslabones_0", "helices_0", "hueveras_0", "pionono_0",
    "eslabones_12", "helices_12", "hueveras_12", "pionono_12", "anteojos",
    "iris", "pinguinos", "vino", "digitos", "mnist",
]  # fmt: skip


# Los rótulos de `docs/defensa.typ`
rotulos = {
    "fkdc": "fKDC", "kdc": "KDC", "fkn": "fkNN", "kn": "kNN",
    "gnb": "GNB", "lr": "LR", "gbt": "GBT", "svc": "SVC",
}  # fmt: skip


def veredicto(a, b, tolerancia=0.005):
    """Mejor, igual o peor, para los R² de A contra los de B."""
    if abs(np.median(a) - np.median(b)) < tolerancia:
        return "igual"
    if np.median(b) < np.percentile(a, 25):
        return "mejor"
    if np.median(a) < np.percentile(b, 25):
        return "peor"
    return "igual"


def resumen(clf, par):
    """Fila de veredictos de `clf` contra `par` y contra el mejor de los demás."""
    corridas = pd.read_csv(dir_raiz / "sandbox/heroica/corridas-v5.csv")
    marcas = {
        "mejor": ("^", "#1a9641"),
        "igual": ("o", "#9a9ea4"),
        "peor": ("v", "#d7191c"),
    }
    fig, ax = plt.subplots(figsize=(13, 3.4))
    for x, dataset in enumerate(crudas_tesis):
        r2 = {
            k: g.r2.dropna().to_numpy()
            for k, g in corridas[corridas.dataset.eq(dataset)].groupby("clf")
        }
        medianas = {k: np.median(v) for k, v in r2.items() if len(v) and k != clf}
        mejor_otro = max(medianas, key=medianas.get)
        for fila, rival in [(1, par), (0, mejor_otro)]:
            marcador, color = marcas[veredicto(r2[clf], r2[rival])]
            ax.scatter(x, fila, marker=marcador, s=260, color=color, lw=0)
        ax.text(
            x,
            -0.42,
            rotulos[mejor_otro],
            ha="center",
            va="top",
            fontsize=10,
            color=tinta,
            family="monospace",
        )
    for corte in [2.5, 5.5, 9.5, 13.5]:
        ax.axvline(corte, color=regla, lw=1)
    ax.set_xticks(range(len(crudas_tesis)))
    ax.set_xticklabels(
        crudas_tesis, rotation=45, ha="left", family="monospace", fontsize=11
    )
    ax.xaxis.tick_top()
    ax.set_yticks(
        [1, 0], [f"vs. {rotulos[par]}", "vs. el mejor\nde los demás"], fontsize=12
    )
    ax.set_ylim(-0.75, 1.5)
    ax.set_xlim(-0.6, len(crudas_tesis) - 0.4)
    ax.tick_params(length=0)
    for borde in ax.spines.values():
        borde.set_visible(False)
    guardar(fig, f"resumen-{clf}.svg")


ilustraciones = {
    "vecindario": lambda: vecindario_espirales(),
    "densidades": lambda: densidades_espirales(),
    "cajitas": lambda: cajitas(),
    "paseos": lambda: paseos_espirales(),
    "resumen": lambda: (resumen("fkdc", "kdc"), resumen("fkn", "kn")),
}


def main():
    """Sin argumentos, todas; si no, solo las nombradas."""
    for nombre in sys.argv[1:] or ilustraciones:
        ilustraciones[nombre]()


if __name__ == "__main__":
    main()
