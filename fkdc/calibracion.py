"""
Calibración por temperatura a posteriori (fuera de la validación cruzada).

Protocolo de la corrida v6: las probabilidades de un clasificador sobre la mitad de
evaluación se templan, p̃ ∝ p^γ, con γ elegido por máxima log-verosimilitud en una
mitad estratificada de la evaluación (semilla fija) y medido en la otra; se promedian
los R² de McFadden de ambas mitades. Así el R² compara clasificadores tras corregir
su sub- o sobreconfianza, que de otro modo domina las diferencias de R².
"""

import numpy as np
from scipy.special import logsumexp
from sklearn.model_selection import train_test_split

gammas = np.logspace(np.log10(0.25), np.log10(8), 41)
semilla_particion = 0
# Recorte como el de `sklearn.metrics.log_loss`, para que γ = 1 reproduzca el R² crudo
_eps = float(np.finfo(np.float64).eps)


def _curva_logvero(log_probas, indices_y, mascara, gammas):
    """Log-verosimilitud de las probabilidades templadas sobre `mascara`, por γ."""
    sub = log_probas[mascara]
    clases = indices_y[mascara]
    templadas = gammas[:, None, None] * sub[None, :, :]
    templadas = templadas - logsumexp(templadas, axis=2, keepdims=True)
    correctas = templadas[:, np.arange(len(clases)), clases]
    return np.clip(correctas, np.log(_eps), 0.0).sum(axis=1)


def templar(
    probas,
    y_eval,
    clases,
    frecuencias_entrenamiento,
    gammas=gammas,
    semilla: int = semilla_particion,
) -> dict:
    """R² crudo y templado de unas probabilidades sobre la evaluación.

    `probas` tiene una columna por clase en el orden de `clases`, y
    `frecuencias_entrenamiento` las proporciones de clase del entrenamiento (el
    clasificador nulo del R² de McFadden), en ese mismo orden. Devuelve `r2_crudo`
    (γ = 1), `r2_temp` (promedio de las dos mitades), `gamma` (media de los γ elegidos),
    `gamma_mitades` y `r2_mitades`.
    """
    probas = np.asarray(probas, float)
    y_eval = np.asarray(y_eval)
    indice = {c: i for i, c in enumerate(np.asarray(clases))}
    indices_y = np.array([indice[y] for y in y_eval])
    frecuencias = np.asarray(frecuencias_entrenamiento, float)
    n = len(y_eval)
    log_probas = np.log(np.clip(probas, _eps, 1.0))
    logvero_nula = float(np.log(frecuencias[indices_y]).sum())
    correctas = np.clip(log_probas[np.arange(n), indices_y], np.log(_eps), 0.0)
    r2_crudo = 1 - float(correctas.sum()) / logvero_nula
    mitad_a, mitad_b = train_test_split(
        np.arange(n), test_size=0.5, random_state=semilla, stratify=y_eval
    )
    r2_mitades, gamma_mitades = [], []
    for ajuste, prueba in ((mitad_a, mitad_b), (mitad_b, mitad_a)):
        en_ajuste = np.zeros(n, bool)
        en_ajuste[ajuste] = True
        en_prueba = np.zeros(n, bool)
        en_prueba[prueba] = True
        curva = _curva_logvero(log_probas, indices_y, en_ajuste, gammas)
        elegido = int(np.argmax(curva))
        logvero = _curva_logvero(log_probas, indices_y, en_prueba, gammas)[elegido]
        nula = float(np.log(frecuencias[indices_y[en_prueba]]).sum())
        r2_mitades.append(1 - logvero / nula)
        gamma_mitades.append(float(gammas[elegido]))
    return {
        "r2_crudo": r2_crudo,
        "r2_temp": float(np.mean(r2_mitades)),
        "gamma": float(np.mean(gamma_mitades)),
        "gamma_mitades": gamma_mitades,
        "r2_mitades": [float(r) for r in r2_mitades],
    }
