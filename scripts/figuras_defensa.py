"""
Figuras y tablas de la segunda parte de la defensa (posteriores a la entrega).

Lee la corrida heroica (`sandbox/heroica`: réplica prerregistrada, contrastes de
Wilcoxon e infos) y los barridos de 25 semillas (`sandbox/fundacionales`), y escribe
SVG en `docs/img/defensa/` y CSV en `docs/data/defensa/`, que `docs/defensa.typ` usa.

Uso, desde la raíz del repositorio, después de `scripts/wilcoxon.py`:
    .venv/bin/python scripts/figuras_defensa.py
"""

import json
import logging

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fkdc import campanias, dir_raiz  # noqa: E402

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger("figuras_defensa")

dir_heroica = dir_raiz / "sandbox/heroica"
dir_fundacionales = dir_raiz / "sandbox/fundacionales"
dir_img = dir_raiz / "docs/img/defensa"
dir_data = dir_raiz / "docs/data/defensa"
crudas_tesis = [
    "lunas_lo", "circulos_lo", "espirales_lo", "lunas_hi", "circulos_hi",
    "espirales_hi", "eslabones_0", "helices_0", "hueveras_0", "pionono_0",
    "eslabones_12", "helices_12", "hueveras_12", "pionono_12", "mnist",
    "anteojos", "iris", "pinguinos", "vino", "digitos",
]  # fmt: skip
alfa = 0.05
tinta, tinta_2, regla = "#1b1f23", "#55595f", "#d9dcdf"
colores = {  # paleta categórica de referencia, orden fijo
    "fkdc": "#2a78d6",
    "tabpfn": "#eb6834",
    "kdc_ard": "#1baf7a",
    "superlearner": "#4a3aa7",
}
marcadores = {"fkdc": "o", "tabpfn": "D", "kdc_ard": "s", "superlearner": "^"}
etiquetas = {
    "fkdc": "fKDC",
    "tabpfn": "TabPFN-3.5",
    "kdc_ard": "KDC con un ancho por variable",
    "superlearner": "superlearner clásico",
}

plt.rcParams.update(
    {
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.edgecolor": regla,
        "axes.labelcolor": tinta_2,
        "xtick.color": tinta_2,
        "ytick.color": tinta_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def veredictos():
    """Conteo de veredictos de la réplica prerregistrada, primaria y sensibilidades."""
    analisis = {
        "primaria": "replica-heroica.csv",
        "kdc de sklearn": "replica-heroica-kdc-sklearn.csv",
        "selección ciega a α": "replica-heroica-seleccion-v6.csv",
        "R² templado": "replica-heroica-r2-templado.csv",
        "25 semillas de la tesis": "replica-heroica-semillas-25.csv",
    }
    orden = ["replica", "no concluyente", "no replica", "contradice"]
    filas = []
    for nombre, archivo in analisis.items():
        conteo = pd.read_csv(dir_heroica / archivo).veredicto.value_counts()
        filas.append({"análisis": nombre, **{v: int(conteo.get(v, 0)) for v in orden}})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(dir_data / "replica-veredictos.csv", index=False)
    return tabla


def pares_wilcoxon():
    return pd.read_csv(dir_heroica / "wilcoxon/pares.csv")


def par(pares, metrica, variante, a, b):
    """Fila del par (a, b) orientada como a − b."""
    q = pares[pares.metrica.eq(metrica) & pares.variante.eq(variante)]
    directo = q[q.a.eq(a) & q.b.eq(b)]
    if len(directo):
        return directo.iloc[0].to_dict()
    r = q[q.a.eq(b) & q.b.eq(a)].iloc[0].to_dict()
    r["hl"], r["hl_bajo"], r["hl_alto"] = -r["hl"], -r["hl_alto"], -r["hl_bajo"]
    return r


def fkdc_kdc_crudo_templado(pares):
    """Diferencia de Hodges–Lehmann fkdc − kdc con IC 95 %, crudo y templado."""
    filas = []
    for v in crudas_tesis:
        for metrica in ("r2", "r2_temp"):
            r = par(pares, metrica, v, "fkdc", "kdc_exacto")
            filas.append(
                {
                    "dataset": v,
                    "metrica": metrica,
                    "hl": r["hl"],
                    "bajo": r["hl_bajo"],
                    "alto": r["hl_alto"],
                    "holm": r["holm_tesis"],
                }
            )
    tabla = pd.DataFrame(filas)
    tabla.to_csv(dir_data / "fkdc-kdc-crudo-templado.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ys = np.arange(len(crudas_tesis))[::-1]
    for desplazo, metrica, relleno, nombre in (
        (0.17, "r2", True, "R² crudo"),
        (-0.17, "r2_temp", False, "R² templado"),
    ):
        t = tabla[tabla.metrica.eq(metrica)].set_index("dataset").loc[crudas_tesis]
        y = ys + desplazo
        ax.hlines(y, t.bajo, t.alto, color=colores["fkdc"], lw=1.6)
        signif = t.holm < alfa
        ax.scatter(
            t.hl,
            y,
            s=38,
            marker="o",
            facecolor=colores["fkdc"] if relleno else "white",
            edgecolor=colores["fkdc"],
            lw=1.6,
            zorder=3,
            label=nombre,
        )
        for yi, hi, sig in zip(y, t.alto, signif, strict=True):
            if sig:
                ax.text(hi + 0.004, yi, "*", va="center", color=tinta, fontsize=12)
    ax.axvline(0, color=tinta_2, lw=1, ls=(0, (3, 3)))
    ax.axvspan(-0.01, 0.01, color="#e9e8e4", zorder=0)
    ax.set_yticks(ys, crudas_tesis, fontfamily="monospace", fontsize=10)
    ax.set_xlabel("fKDC − KDC exacto (Hodges–Lehmann, IC 95 %)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color=regla, lw=0.6)
    fig.tight_layout()
    fig.savefig(dir_img / "fkdc-kdc-crudo-templado.svg")
    plt.close(fig)


def grupo_lider(pares):
    """Veces en el grupo líder (no peor que la mejor mediana, Holm) por clasificador."""
    resultado = json.loads((dir_heroica / "wilcoxon/resultados.json").read_text())
    filas = []
    for conjunto in ("tesis", "ampliado"):
        miembros = resultado["conjuntos"][conjunto]
        for metrica in ("r2", "r2_temp"):
            cuenta = dict.fromkeys(miembros, 0)
            for v in [x["id"] for x in resultado["variantes"]]:
                valores = resultado["valores"][metrica][v]
                medianas = {
                    c: float(
                        np.median([round(x, 3) for x in valores[c] if x is not None])
                    )
                    for c in miembros
                }
                mejor = sorted(miembros, key=lambda c: -medianas[c])[0]
                for c in miembros:
                    if c == mejor:
                        cuenta[c] += 1
                        continue
                    r = par(pares, metrica, v, mejor, c)
                    if r[f"holm_{conjunto}"] >= alfa:
                        cuenta[c] += 1
            for c, n in cuenta.items():
                filas.append(
                    {"conjunto": conjunto, "metrica": metrica, "clf": c, "n": n}
                )
    tabla = pd.DataFrame(filas).pivot_table(
        index=["conjunto", "clf"], columns="metrica", values="n"
    )
    tabla = tabla.reset_index().sort_values(
        ["conjunto", "r2"], ascending=[False, False]
    )
    tabla["clf"] = tabla.clf.replace({"kdc_exacto": "kdc"})
    tabla.to_csv(dir_data / "grupo-lider.csv", index=False)
    return tabla


def mapa_tabpfn():
    """Mediana de R² crudo por dataset: fkdc, dos clásicos y TabPFN (100 semillas)."""
    c = campanias.campania(
        "heroica", ("tesis", "compendio", "fundacionales", "clasicos")
    )
    corridas = campanias.corridas(c)
    clfs = ["fkdc", "kdc_ard", "superlearner", "tabpfn"]
    medianas = (
        corridas[corridas.clf.isin(clfs) & corridas.variante.isin(crudas_tesis)]
        .groupby(["variante", "clf"])
        .agg(mediana=("r2", "median"), n=("r2", "size"))
        .reset_index()
    )
    medianas.to_csv(dir_data / "mapa-tabpfn.csv", index=False)
    tabla = medianas.pivot(index="variante", columns="clf", values="mediana").reindex(
        crudas_tesis
    )
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ys = np.arange(len(crudas_tesis))[::-1]
    for y, (_, fila) in zip(ys, tabla.iterrows(), strict=True):
        presentes = fila.dropna()
        ax.hlines(y, presentes.min(), presentes.max(), color=regla, lw=2, zorder=1)
    for k, clf in enumerate(clfs):
        if clf not in tabla:
            continue
        ax.scatter(
            tabla[clf].clip(lower=-0.05),
            ys + (k - 1.5) * 0.06,
            s=44,
            marker=marcadores[clf],
            color=colores[clf],
            edgecolor="white",
            lw=0.8,
            zorder=3,
            label=etiquetas[clf],
        )
    ax.set_yticks(ys, crudas_tesis, fontfamily="monospace", fontsize=10)
    ax.set_xlabel("R² crudo, mediana sobre 100 semillas")
    ax.set_xlim(-0.06, 1.02)
    ax.grid(axis="x", color=regla, lw=0.6)
    ax.legend(frameon=False, loc="lower left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(dir_img / "mapa-tabpfn.svg")
    plt.close(fig)
    return tabla


def oraculo():
    """Oráculo (LR en coordenadas intrínsecas) frente a fkdc y TabPFN, 25 semillas."""
    r = pd.read_csv(dir_fundacionales / "resumen-oraculo.csv").set_index("variante")
    t = pd.read_csv(dir_fundacionales / "resumen-tabpfn.csv")
    t = t[t.clf.eq("tabpfn")].set_index("dataset")
    elegidos = [
        "lunas_lo", "espirales_lo", "helices_0", "hueveras_0", "pionono_0",
        "helices_12", "hueveras_12",
    ]  # fmt: skip
    oraculos = [c for c in r.columns if c.endswith("_oraculo_r2_mediana")]
    tabla = pd.DataFrame(
        {
            "dataset": elegidos,
            "fkdc": [r.loc[v, "fkdc_r2_mediana"] for v in elegidos],
            "tabpfn": [t.loc[v, "r2_mediana"] for v in elegidos],
            "lr_oraculo": [r.loc[v, "lr_oraculo_r2_mediana"] for v in elegidos],
            "mejor_oraculo": [r.loc[v, oraculos].max() for v in elegidos],
        }
    ).round(3)
    tabla.to_csv(dir_data / "oraculo.csv", index=False)
    return tabla


def anchos_ard():
    """Cociente entre la menor ventana de ruido y la mayor de señal en kdc_ard."""
    anchos = pd.read_csv(dir_fundacionales / "anchos-kdc_ard.csv")
    senal = ["h1", "h2", "h3"]  # las variantes `_12` tienen D = 3 + 12 de ruido
    ruido = [f"h{k}" for k in range(4, 16)]
    anchos["cociente"] = anchos[ruido].min(axis=1) / anchos[senal].max(axis=1)
    tabla = (
        anchos.groupby("variante")
        .agg(
            cociente_mediano=("cociente", "median"),
            separa=("cociente", lambda c: float((c > 1).mean())),
        )
        .round(2)
        .reset_index()
    )
    tabla.to_csv(dir_data / "anchos-kdc_ard.csv", index=False)
    return tabla


def main():
    dir_img.mkdir(parents=True, exist_ok=True)
    dir_data.mkdir(parents=True, exist_ok=True)
    logger.info("Veredictos:\n%s", veredictos().to_string(index=False))
    pares = pares_wilcoxon()
    fkdc_kdc_crudo_templado(pares)
    logger.info("Grupo líder:\n%s", grupo_lider(pares).to_string(index=False))
    logger.info("Mapa TabPFN:\n%s", mapa_tabpfn().round(3).to_string())
    logger.info("Oráculo:\n%s", oraculo().to_string(index=False))
    logger.info("Anchos kdc_ard: %s", anchos_ard().columns.tolist())


if __name__ == "__main__":
    main()
