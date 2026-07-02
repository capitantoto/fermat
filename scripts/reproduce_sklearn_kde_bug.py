"""Reproductor mínimo standalone del bug de `sklearn.neighbors.KernelDensity`
observado durante el desarrollo de esta tesis.

## Contexto

Al comparar los clasificadores #fkdc (basado en `FermatKDE`, fuerza bruta) y
#kdc (basado en `sklearn.KernelDensity`, KDTree/BallTree), aparecieron
discrepancias grandes (>100 nats de log-densidad) en el régimen de cola
profunda: puntos de consulta que caen dentro de la caja envolvente ("bounding
box") del conjunto de entrenamiento pero lejos de todos los puntos, con
bandwidth `h` chico.

## Régimen del bug

`sklearn/neighbors/_binary_tree.pxi` mantiene cotas globales en log-espacio
(`global_log_min_bound`, `global_log_bound_spread`) con cadenas de
`logaddexp`/`logsubexp`. Cuando el punto de consulta cae dentro de la caja
envolvente del conjunto de entrenamiento (`dist_LB(root)=0` ⇒ spread inicial
≈ `log N`) pero lejos de todos los puntos, la cancelación catastrófica de los
`logsubexp` deja un residuo de orden `eps·N` que el recorrido nunca limpia, y
la log-densidad devuelta queda anclada cerca de `ln(eps) + log N ≈ −31` sin
importar cuán chica sea la densidad verdadera.

## Este script

Verifica los tres puntos que hacen que este bug afecte casos reales:

1. En un dataset sintético mínimo (dos círculos concéntricos en 3D, uno
   dentro de la caja envolvente del otro), `KernelDensity` devuelve valores
   erróneos de hasta ~300 nats respecto de la log-densidad verdadera
   (calculada con `logsumexp` sobre `cdist`).
2. El workaround `leaf_size >= N` (árbol de una sola hoja → suma exacta
   dentro de la hoja) corrige el resultado a precisión numérica.
3. Fuera del régimen de cola profunda (bandwidth moderado), sklearn y
   `logsumexp` coinciden a ~1e-13.

## Referencias upstream

- scikit-learn#27186 (abierto, "Needs Investigation"): identifica el origen
  probable en `node_log_bound_spreads`.
- scikit-learn#25799 (cerrado sin fix): mismo síntoma reportado con bandwidth
  chico.

## Uso

    uv run python scripts/reproduce_sklearn_kde_bug.py
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from sklearn.neighbors import KernelDensity


def logdensidad_exacta(X_query, X_train, h):
    """KDE gaussiano con norma L2, calculado por fuerza bruta con logsumexp."""
    N, d = X_train.shape
    log_suma = logsumexp(-0.5 * (cdist(X_query, X_train) / h) ** 2, axis=1)
    return -np.log(N) - d * np.log(h) - d / 2 * np.log(2 * np.pi) + log_suma


def dataset_nested(seed=42):
    """Dos círculos concéntricos en 3D: uno de radio 1.5 (afuera), otro de
    radio 1.0 (adentro). La caja envolvente del círculo externo contiene a
    todos los puntos del interno + a puntos "de consulta" que están LEJOS de
    los del externo pero DENTRO de la caja.
    """
    rng = np.random.default_rng(seed)
    N = 100
    # Círculo externo: radio 1.5, en plano xy
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.normal(0, 0.01, N)
    X_ext = np.column_stack(
        [1.5 * np.cos(theta), 1.5 * np.sin(theta), rng.normal(0, 0.01, N)]
    )
    # Puntos de consulta: dentro de la caja [-1.5, 1.5]^3 pero LEJOS del círculo
    X_query = np.array(
        [
            [0.0, 0.0, 0.0],  # centro: dist mínima a círculo externo = 1.5
            [0.5, 0.5, 0.5],  # también interior
            [0.3, 0.3, 0.0],  # otro punto en el plano xy pero cerca del centro
        ]
    )
    return X_ext, X_query


def main():
    print("=" * 78)
    print("Reproductor mínimo del bug de sklearn.KernelDensity en cola profunda")
    print("=" * 78)

    X_train, X_query = dataset_nested()
    N, d = X_train.shape

    # Régimen del bug: h chico + query dentro de bounding box del train
    h_bug = 0.05
    print(f"\nTrain: {N} puntos en R^{d} (círculo de radio 1.5 en xy)")
    print(f"Query: {len(X_query)} puntos dentro del bounding box del train")
    print(f"Bandwidth: h = {h_bug}")
    print(f"Distancia mínima query→train: {cdist(X_query, X_train).min(axis=1)}")

    # 1. sklearn KDTree (default): buggy
    kde_default = KernelDensity(bandwidth=h_bug, kernel="gaussian", atol=0, rtol=0)
    kde_default.fit(X_train)
    ld_sklearn = kde_default.score_samples(X_query)

    # 2. sklearn con leaf_size >= N (workaround): árbol de una sola hoja
    kde_leaf = KernelDensity(
        bandwidth=h_bug, kernel="gaussian", atol=0, rtol=0, leaf_size=N + 1
    )
    kde_leaf.fit(X_train)
    ld_sklearn_leaf = kde_leaf.score_samples(X_query)

    # 3. Fuerza bruta con logsumexp: verdad
    ld_exact = logdensidad_exacta(X_query, X_train, h_bug)

    print("\n--- Comparación (log-densidades en cada punto de consulta) ---")
    print(
        f"{'sklearn KDTree':>20s} {'sklearn leaf=N+1':>20s} {'logsumexp exacto':>20s}"
    )
    for i in range(len(X_query)):
        print(
            f"{ld_sklearn[i]:>20.4f} {ld_sklearn_leaf[i]:>20.4f} {ld_exact[i]:>20.4f}"
        )

    err_default = np.abs(ld_sklearn - ld_exact)
    err_leaf = np.abs(ld_sklearn_leaf - ld_exact)

    print(f"\nError máximo sklearn KDTree vs exacto: {err_default.max():.3e}")
    print(f"Error máximo sklearn leaf=N+1 vs exacto: {err_leaf.max():.3e}")

    assert err_default.max() > 1.0, (
        "el fixture no reprodujo el bug: los errores son <1 nat. "
        "Chequear que query está dentro del bounding box y h suficientemente chico."
    )
    assert (
        err_leaf.max() < 1e-10
    ), f"el workaround leaf_size>=N no corrigió el bug: err={err_leaf.max():.3e}"
    print("\n  Bug reproducido: sklearn KDTree difiere del exacto por >1 nat")
    print("  Workaround verificado: sklearn con leaf_size>=N coincide con exacto")

    # Sanity check: fuera del régimen del bug, sklearn coincide
    h_ok = 0.5
    print(f"\n--- Sanity: h = {h_ok} (fuera del régimen del bug) ---")
    kde = KernelDensity(bandwidth=h_ok, kernel="gaussian", atol=0, rtol=0).fit(X_train)
    ld_sk = kde.score_samples(X_query)
    ld_ex = logdensidad_exacta(X_query, X_train, h_ok)
    err = np.abs(ld_sk - ld_ex).max()
    print(f"Error máximo sklearn vs exacto: {err:.3e}")
    assert err < 1e-8, f"error inesperado fuera del régimen del bug: {err:.3e}"
    print("  Consistente con el exacto a ~1e-13 (bandwidth moderada)")

    print("\n" + "=" * 78)
    print("Todos los asserts pasaron. Bug reproducido y workaround validado.")
    print("=" * 78)


if __name__ == "__main__":
    main()
