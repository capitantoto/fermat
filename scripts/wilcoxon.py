"""
Contrastes de Wilcoxon por dataset sobre una campaña (por omisión, la heroica).

Análisis exploratorio, fuera del prerregistro. Para cada variante y cada métrica (R²
crudo y R² templado, siempre del mismo tipo contra el mismo tipo) compara de a pares
los clasificadores sobre las mismas semillas:

- prueba de rangos con signo de Wilcoxon, bilateral, con los ceros por el método de
  Pratt (las semillas en que dos clasificadores empatan exactamente son frecuentes);
- corrección de Holm dentro de cada variante, por separado para el conjunto de la
  tesis (sus siete clasificadores suaves: svc no da probabilidades ni R²) y para el
  ampliado (más el compendio, los fundacionales locales y los clásicos de
  `fkdc.clasicos`);
- estimador de Hodges–Lehmann de la diferencia y su intervalo del 95 % (promedios de
  Walsh, aproximación normal);
- prueba de Friedman por variante y conjunto, como ómnibus;
- para los pares del conjunto de la tesis, la prueba bayesiana de rangos con signo
  con ROPE ±0,01 (Benavoli, Corani, Demšar y Zaffalon, 2017): probabilidad de que
  gane cada uno o de que sean prácticamente equivalentes.

En los datasets de datos fijos (anteojos, iris, vino, pinguinos, digitos y sus
`_std`) cada semilla es otra partición de los mismos datos: los resultados están
correlacionados y los p-valores salen optimistas. Se marcan como tales.

Escribe `{dir}/wilcoxon/resultados.json` (insumo de la página) y `pares.csv`.

Uso, desde la raíz del repositorio:
    .venv/bin/python scripts/wilcoxon.py heroica
"""

import json
import logging
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from scipy import stats

from fkdc import campanias, registro

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("wilcoxon")

metricas = {"r2": "R² crudo", "r2_temp": "R² templado"}
rope = 0.01  # margen de equivalencia práctica del prerregistro
prior_bayes = 0.5  # fuerza de la pseudo-observación en 0 (valor por omisión de baycomp)
muestras_bayes = 20_000
semilla_bayes = 1312
tolerancia_empate = 1e-9
bases_fijas = {"anteojos", "iris", "vino", "pinguinos", "digitos"}
tesis = ["fkdc", "kdc_exacto", "fkn", "kn", "gnb", "lr", "gbt"]
conjuntos_campania = ("tesis", "compendio", "fundacionales", "clasicos")
excluidos = {"svc", "kdc"}  # svc: sin R²; kdc de sklearn: sensibilidad, no método
etiquetas = {"kdc_exacto": "kdc"}
fermat = {"fkdc", "fkn"}


def holm(p: np.ndarray) -> np.ndarray:
    """p-valores ajustados por Holm (paso descendente, monótonos)."""
    p = np.asarray(p, float)
    ajustados = np.empty_like(p)
    acumulado = 0.0
    for rango, i in enumerate(np.argsort(p)):
        acumulado = max(acumulado, (len(p) - rango) * p[i])
        ajustados[i] = min(1.0, acumulado)
    return ajustados


def hodges_lehmann(d: np.ndarray, nivel: float = 0.95) -> tuple[float, float, float]:
    """Mediana de los promedios de Walsh y su IC por la cota normal de Wilcoxon."""
    n = len(d)
    i, j = np.triu_indices(n)
    walsh = np.sort((d[i] + d[j]) / 2)
    z = stats.norm.ppf(1 - (1 - nivel) / 2)
    k = int(np.floor(n * (n + 1) / 4 - z * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)))
    k = min(max(k, 0), len(walsh) - 1)
    return float(np.median(walsh)), float(walsh[k]), float(walsh[-1 - k])


def bayes_rangos_signo(d: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """P(gana b), P(ROPE), P(gana a) para d = a − b (Benavoli et al., 2017).

    Proceso de Dirichlet con una pseudo-observación en 0 de peso `prior_bayes`; en
    cada muestra de pesos w, θ_izq = Σ wᵢwⱼ 1{zᵢ+zⱼ < −2r}, θ_rope = Σ wᵢwⱼ
    1{|zᵢ+zⱼ| ≤ 2r} y θ_der el resto; se informa la frecuencia con que gana cada uno.
    """
    z = np.concatenate([[0.0], d])
    suma = z[:, None] + z[None, :]
    izquierda = (suma < -2 * rope).astype(float)
    centro = (np.abs(suma) <= 2 * rope).astype(float)
    pesos = rng.dirichlet(
        np.concatenate([[prior_bayes], np.ones(len(d))]), muestras_bayes
    )
    t_izq = ((pesos @ izquierda) * pesos).sum(axis=1)
    t_rope = ((pesos @ centro) * pesos).sum(axis=1)
    t_der = 1.0 - t_izq - t_rope
    gana = np.argmax(np.stack([t_izq, t_rope, t_der], axis=1), axis=1)
    return np.bincount(gana, minlength=3) / muestras_bayes


def comparar(a: np.ndarray, b: np.ndarray, rng, bayes: bool) -> dict:
    """Wilcoxon, Hodges–Lehmann y (si corresponde) la prueba bayesiana de a − b."""
    validos = ~(np.isnan(a) | np.isnan(b))
    d = a[validos] - b[validos]
    # Polvo numérico: empate, como en el prerregistro
    d[np.abs(d) < tolerancia_empate] = 0.0
    gana_a, gana_b = int((d > 0).sum()), int((d < 0).sum())
    fila = {
        "n": int(len(d)),
        "no_nulos": gana_a + gana_b,
        "gana_a": gana_a,
        "gana_b": gana_b,
        "empates": int((d == 0).sum()),
    }
    if fila["no_nulos"] == 0:
        fila.update(p=1.0, hl=0.0, hl_bajo=0.0, hl_alto=0.0)
    else:
        prueba = stats.wilcoxon(d, zero_method="pratt", alternative="two-sided")
        fila["p"] = float(prueba.pvalue)
        fila["hl"], fila["hl_bajo"], fila["hl_alto"] = hodges_lehmann(d)
    if bayes:
        fila["bayes_b"], fila["bayes_rope"], fila["bayes_a"] = bayes_rangos_signo(
            d, rng
        ).tolist()
    return fila


def main(
    nombre: Annotated[str, typer.Argument(help="Campaña")] = "heroica",
    conjunto: Annotated[
        list[str] | None,
        typer.Option(help="Conjuntos de `fkdc.campanias.conjuntos` (repetible)"),
    ] = None,
):
    """Contrastes de Wilcoxon por variante, en el conjunto de la tesis y el ampliado."""
    t0 = time.time()
    c = campanias.campania(nombre, tuple(conjunto or conjuntos_campania))
    corridas = campanias.corridas(c)
    logger.info("%d corridas leídas en %.0f s", len(corridas), time.time() - t0)
    ampliado = [clf for clf in c.clasificadores if clf not in excluidos]
    conjuntos = {"tesis": tesis, "ampliado": ampliado}
    rng = np.random.default_rng(semilla_bayes)
    salida_dir = c.dir / "wilcoxon"
    salida_dir.mkdir(parents=True, exist_ok=True)

    resultado = {
        "meta": {
            "campania": nombre,
            "semillas": c.repeticiones,
            "generado": time.strftime("%Y-%m-%d %H:%M"),
            "rope": rope,
            "prior_bayes": prior_bayes,
            "muestras_bayes": muestras_bayes,
        },
        "clasificadores": [
            {
                "id": clf,
                "etiqueta": etiquetas.get(clf, clf),
                "familia": registro.entrada(clf).grupo,
                "fermat": clf in fermat,
            }
            for clf in ampliado
        ],
        "conjuntos": conjuntos,
        "metricas": metricas,
        "variantes": [
            {
                "id": v,
                "nuevo": v in campanias._variantes_nuevas(),
                "fijo": v.removesuffix("_std") in bases_fijas,
            }
            for v in c.variantes
        ],
        "valores": {},
        "friedman": {},
        "pares": {},
    }
    filas_csv = []
    for metrica in metricas:
        resultado["valores"][metrica] = {}
        resultado["friedman"][metrica] = {k: {} for k in conjuntos}
        resultado["pares"][metrica] = {}
        for variante in c.variantes:
            tabla = (
                corridas[corridas.variante.eq(variante) & corridas.clf.isin(ampliado)]
                .pivot_table(index="semilla", columns="clf", values=metrica)
                .reindex(columns=ampliado)
                .sort_index()
            )
            resultado["valores"][metrica][variante] = {
                clf: [None if np.isnan(x) else round(float(x), 4) for x in tabla[clf]]
                for clf in ampliado
            }
            for nombre_conjunto, miembros in conjuntos.items():
                completas = tabla[miembros].dropna()
                prueba = stats.friedmanchisquare(*[completas[m] for m in miembros])
                resultado["friedman"][metrica][nombre_conjunto][variante] = {
                    "chi2": round(float(prueba.statistic), 3),
                    "p": float(prueba.pvalue),
                    "n": len(completas),
                }
            pares = []
            for i, a in enumerate(ampliado):
                for b in ampliado[i + 1 :]:
                    fila = comparar(
                        tabla[a].to_numpy(),
                        tabla[b].to_numpy(),
                        rng,
                        bayes=a in tesis and b in tesis,
                    )
                    pares.append({"a": a, "b": b, **fila})
            for nombre_conjunto, miembros in conjuntos.items():
                indices = [
                    k
                    for k, par in enumerate(pares)
                    if par["a"] in miembros and par["b"] in miembros
                ]
                ajustados = holm([pares[k]["p"] for k in indices])
                for k, p_holm in zip(indices, ajustados, strict=True):
                    pares[k][f"holm_{nombre_conjunto}"] = float(p_holm)
            resultado["pares"][metrica][variante] = pares
            filas_csv += [
                {"metrica": metrica, "variante": variante, **p} for p in pares
            ]
        logger.info("%s listo (%.0f s)", metrica, time.time() - t0)

    ruta_json = salida_dir / "resultados.json"
    temporal = ruta_json.with_name(f".{ruta_json.name}.tmp")
    temporal.write_text(
        json.dumps(resultado, ensure_ascii=False, separators=(",", ":"))
    )
    temporal.rename(ruta_json)
    pd.DataFrame(filas_csv).to_csv(salida_dir / "pares.csv", index=False)
    escribir_pagina(resultado, salida_dir / "index.html")
    logger.info(
        "Escribió %s, pares.csv e index.html en %.0f s", ruta_json, time.time() - t0
    )


def _sig(x) -> float | None:
    """Tres cifras significativas; None para faltantes (JSON no admite NaN)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return float(f"{x:.3g}")


def datos_pagina(resultado: dict) -> dict:
    """Versión compacta de los resultados para incrustar en la página."""
    clfs = [x["id"] for x in resultado["clasificadores"]]
    indice = {clf: i for i, clf in enumerate(clfs)}
    variantes = [v["id"] for v in resultado["variantes"]]

    def fila_par(p):
        bayes = [p.get(k) for k in ("bayes_b", "bayes_rope", "bayes_a")]
        return [
            indice[p["a"]],
            indice[p["b"]],
            _sig(p["p"]),
            _sig(p.get("holm_tesis")),
            _sig(p["holm_ampliado"]),
            round(p["hl"], 4),
            round(p["hl_bajo"], 4),
            round(p["hl_alto"], 4),
            p["gana_a"],
            p["gana_b"],
            p["empates"],
            *[None if x is None else round(x, 3) for x in bayes],
        ]

    return {
        "meta": resultado["meta"],
        "clasificadores": resultado["clasificadores"],
        "conjuntos": {
            k: [indice[c] for c in v] for k, v in resultado["conjuntos"].items()
        },
        "metricas": resultado["metricas"],
        "variantes": resultado["variantes"],
        "valores": {
            m: [
                [
                    [None if x is None else round(x, 3) for x in por_var[v][c]]
                    for c in clfs
                ]
                for v in variantes
            ]
            for m, por_var in resultado["valores"].items()
        },
        "friedman": {
            m: {
                k: [[f[v]["chi2"], _sig(f[v]["p"])] for v in variantes]
                for k, f in por_conjunto.items()
            }
            for m, por_conjunto in resultado["friedman"].items()
        },
        "pares": {
            m: [[fila_par(p) for p in por_var[v]] for v in variantes]
            for m, por_var in resultado["pares"].items()
        },
    }


def escribir_pagina(resultado: dict, destino: Path) -> Path:
    """Incrusta los datos en la plantilla `scripts/wilcoxon.html`."""
    plantilla = (Path(__file__).parent / "wilcoxon.html").read_text(encoding="utf-8")
    datos = json.dumps(
        datos_pagina(resultado), ensure_ascii=False, separators=(",", ":")
    )
    pagina = plantilla.replace("/*__DATOS__*/null", datos.replace("</", "<\\/"))
    temporal = destino.with_name(f".{destino.name}.tmp")
    temporal.write_text(pagina, encoding="utf-8")
    temporal.rename(destino)
    return destino


if __name__ == "__main__":
    typer.run(main)
