import json
import logging
import pickle
import warnings
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from sklearn.base import BaseEstimator
from sklearn.datasets import make_circles
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KernelDensity
from sklearn.utils import Bunch

from fkdc import config, dir_cache, dir_raiz
from fkdc.datasets import ConjuntoDatos, datasets_reales, datasets_sinteticos
from fkdc.tarea import Tarea

# TODO: Reentrenar con versión consistente para evitar este problema
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

logger = logging.getLogger(__name__)
dir_ejecucion = config.dir_ejecucion
dir_datasets = dir_ejecucion / "../datasets"
dir_imagenes = dir_raiz / "docs" / "img"
dir_datos = dir_raiz / "docs" / "data"

datasets = [*datasets_sinteticos, *datasets_reales]
clfs = list(config.clasificadores.keys())
infos = None
info_basica = None

# Paleta: clfs pareados comparten color base, variantes usan sombreado.
# Familias: dc (fkdc/kdc), kn (fkn/kn), logr (lr/slr), gbt, gnb, svc
_familias_clf = {
    "dc": ["fkdc", "kdc"],
    "kn": ["fkn", "kn"],
    "logr": ["lr", "slr"],
    "gbt": ["gbt"],
    "gnb": ["gnb"],
    "svc": ["svc"],
}
_colores_familia = dict(
    zip(_familias_clf, sns.color_palette("Pastel2", 6), strict=True)
)
paleta_predeterminada = {
    clf: _colores_familia[familia]
    for familia, miembros in _familias_clf.items()
    for clf in miembros
}
_clfs_sombreados = {"kdc", "kn", "slr"}


def guardar_fig(fig: Figure, ruta: str | Path, **kw_guardar):
    """Guarda la figura, registra la ruta y cierra."""
    kw_guardar.setdefault("bbox_inches", "tight")
    fig.savefig(ruta, **kw_guardar)
    logger.info(f"Escribió {ruta}")
    plt.close(fig)


def cargar_infos(dir_o_rutas: list[Path] | Path):
    """Carga diccionarios de info desde pickles."""
    if isinstance(dir_o_rutas, Path):
        assert dir_o_rutas.is_dir()
        rutas = dir_o_rutas.glob("*.pkl")
    elif isinstance(dir_o_rutas, list):
        rutas = dir_o_rutas
    else:
        raise ValueError(
            "`dir_o_rutas` debe ser un directorio con pickles de info "
            "o una lista de rutas a pickles de info"
        )
    return {tuple(fn.stem.split("-")): pickle.load(open(fn, "rb")) for fn in rutas}


def procesar_info_basica(
    infos: dict, semilla_principal: int = config.semilla_principal
):
    """Extrae campos básicos (accuracy, r2, logvero) de los infos."""
    campos_basicos = ["accuracy", "r2", "logvero"]
    infos_basicos = {}
    for k, v in infos.items():
        clf = k[2]
        infos_basicos[k] = {k: v for k, v in v[clf].items() if k in campos_basicos}

    info_basica = pd.DataFrame.from_records(
        list(infos_basicos.values()),
        index=pd.MultiIndex.from_tuples(
            infos_basicos.keys(),
            names=["dataset", "semilla_ds", "clf", "semilla_corrida", "score"],
        ),
    ).reset_index()
    if semilla_principal:  # Valida que las semillas se correspondan
        assert all(
            (info_basica.semilla_ds == "None")
            | (info_basica.semilla_corrida == str(semilla_principal))
        )
    info_basica["semilla"] = np.where(
        info_basica.semilla_ds == "None",
        info_basica.semilla_corrida,
        info_basica.semilla_ds,
    ).astype(int)
    return info_basica.drop(columns=["semilla_ds", "semilla_corrida"])


def _resolver_info_basica(info: pd.DataFrame | None):
    """Resuelve la info básica: la calcula si no fue proporcionada."""
    if info is None:
        logger.debug("No se pasó info, usando info_basica por defecto")
        global infos, info_basica
        if infos is None:
            logger.warning("Infos no parseados aún; parseando...")
            infos = cargar_infos(config.dir_ejecucion)
        if info_basica is None:
            logger.warning("info_basica no extraída aún, extrayendo...")
            info_basica = procesar_info_basica(infos, config.semilla_principal)
        info: pd.DataFrame = info_basica
    return info


def obtener_destacados(
    dataset: str, por: str = "r2", info: None | pd.DataFrame = None
) -> Bunch:
    """Obtiene el mejor clasificador y los excluidos para un dataset."""
    info = _resolver_info_basica(info)
    este_dataset = info["dataset"].eq(dataset)
    resumen = (
        info[este_dataset]
        .groupby("clf")[["accuracy", "r2"]]
        .median()
        .sort_values(by=por, ascending=False)
    )
    mejor = resumen.idxmax()[por]
    corridas_mejor = info[este_dataset & info["clf"].eq(mejor)][por]
    primer_cuartil = np.percentile(corridas_mejor, 25)
    min_mejor = corridas_mejor.min()
    logger.debug(f"mejor: {mejor} (por {por}); p25: {primer_cuartil}; min: {min_mejor}")
    malos = resumen[resumen[por].lt(primer_cuartil)].index.tolist()
    excluidos = resumen[resumen[por].lt(min_mejor)].index.tolist()
    return Bunch(
        dataset=dataset,
        summary=resumen,
        best=mejor,
        bad=malos,
        excluded=excluidos,
    )


def diagrama_caja(
    dataset: str,
    metrica: str = "r2",
    info: pd.DataFrame | None = None,
    ax=None,
    paleta: dict | None = None,
    excluir_clfs: list[str] | None = None,
):
    """Diagrama de caja (boxplot) de una métrica por clasificador."""
    paleta = paleta or paleta_predeterminada
    info = _resolver_info_basica(info)

    if ax is None:
        ax = plt.gca()
    datos = info[info.dataset.eq(dataset)].sort_values("clf").dropna(subset=metrica)
    if excluir_clfs:
        datos = datos[~datos.clf.isin(excluir_clfs)]
    sns.boxplot(
        datos, hue="clf", y=metrica, gap=0.2, ax=ax, palette=paleta, saturation=1.0
    )
    aplicar_sombreado(ax)
    ax.axhline(
        datos.groupby("clf")[metrica].median().max(),
        linestyle="dotted",
        color="gray",
    )


def aplicar_sombreado(ax, clfs_sombreados=None):
    """Aplica sombreado (hatching) a las cajas de clasificadores variantes."""
    clfs_sombreados = clfs_sombreados or _clfs_sombreados
    leyenda = ax.get_legend()
    if leyenda is None:
        return
    etiquetas = [t.get_text() for t in leyenda.get_texts()]
    handles = leyenda.legend_handles
    parches_caja = [p for p in ax.patches if isinstance(p, PathPatch)]
    for i, etiqueta in enumerate(etiquetas):
        if etiqueta in clfs_sombreados:
            if i < len(parches_caja):
                parches_caja[i].set_hatch("///")
            handles[i].set_hatch("///")


def frontera_decision(
    dataset: str,
    semilla: int,
    clf: str,
    ax: Axes | None = None,
    cmap: str | Colormap | None = None,
):
    """Grafica la frontera de decisión de un clasificador sobre un dataset 2D."""
    ax = ax or plt.gca()
    cmap = cmap or colormaps["coolwarm"]

    if isinstance(cmap, str):
        cmap = colormaps[cmap]
    if clf == "svc":
        metodo_respuesta, puntuacion = "predict", "accuracy"
    elif clf in clfs:
        metodo_respuesta, puntuacion = "predict_proba", "neg_log_loss"
    else:
        raise ValueError(f"Clasificador desconocido: {clf}")
    if dataset in datasets_sinteticos:
        ruta_ds = dir_datasets / f"{dataset}-{semilla}.pkl"
        clave_grafico = (
            dataset,
            str(semilla),
            clf,
            str(config.semilla_principal),
            puntuacion,
        )
    elif dataset in datasets_reales:
        ruta_ds = dir_datasets / f"{dataset}.pkl"
        clave_grafico = (
            dataset,
            str(None),
            clf,
            str(semilla),
            puntuacion,
        )
    else:
        raise ValueError(f"Dataset desconocido: {dataset}-{semilla}")
    with open(ruta_ds, "rb") as fp:
        ds: ConjuntoDatos = pickle.load(fp)
    tarea = Tarea(ds, {}, semilla=semilla, split_evaluacion=config.split_evaluacion)
    X = tarea.X_eval
    y = tarea.y_eval
    k = ds.k
    X0, X1 = X[:, 0], X[:, 1]
    with open(
        config.dir_ejecucion / f"{'-'.join(map(str, clave_grafico))}.pkl", "rb"
    ) as fpath:
        info: dict = pickle.load(fpath)
        mejor_est: BaseEstimator = info[clf]["busqueda"].best_estimator_
    DecisionBoundaryDisplay.from_estimator(
        mejor_est,
        X,
        eps=0.05,
        response_method=metodo_respuesta if k == 2 else "predict",
        cmap=cmap,
        alpha=0.8,
        ax=ax,
        xlabel="x",
        ylabel="y",
    )
    ax.scatter(X0, X1, c=y.astype(float), cmap=cmap, s=20, edgecolors="gray")
    ax.set_xticks(())
    ax.set_yticks(())
    acc = info[clf]["accuracy"] * 100
    r2 = info[clf].get("r2", 0)
    ax.set_title(f"{clf} ({acc:.2f}% acc., {r2:.3f} $R^2$)")
    return info, ds, tarea


def contorno_perdida(
    dataset: str,
    semilla: int,
    clf: str,
    x: str,
    y: str,
    otros_params: dict | None = None,
    cmap: str | Colormap = "viridis",
):
    """Grafica el contorno de la función de pérdida en el espacio de hiperparámetros."""
    cmap = cmap or "viridis"
    if isinstance(cmap, str):
        cmap = colormaps[cmap]
    puntuacion = "accuracy" if clf == "svc" else "neg_log_loss"
    if clf not in clfs:
        raise ValueError(f"Clasificador desconocido: {clf}")
    if dataset in datasets_sinteticos:
        clave_grafico = (
            dataset,
            str(semilla),
            clf,
            str(config.semilla_principal),
            puntuacion,
        )
    elif dataset in datasets_reales:
        clave_grafico = (dataset, str(None), clf, str(semilla), puntuacion)
    else:
        raise ValueError(f"Dataset desconocido: {dataset}, {semilla}")
    ruta = config.dir_ejecucion / f"{'-'.join(map(str, clave_grafico))}.pkl"
    with open(ruta, "rb") as fp:
        info: dict = pickle.load(fp)
    resultados_cv = pd.DataFrame(info[clf]["busqueda"].cv_results_)

    datos = resultados_cv.copy()
    otros_params = otros_params or {}
    if otros_params:
        for p, valor in otros_params.items():
            datos = datos[datos[f"param_{p}"] == valor]
    datos = datos.set_index([f"param_{y}", f"param_{x}"]).mean_test_score.unstack()
    X = datos.columns.values
    Y = datos.index.values
    Z = datos.values

    fig, ax = plt.subplots(layout="constrained")
    CS = ax.contourf(X, Y, Z, 15, cmap=cmap)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    mejor_X = X[Z.argmax(axis=1)]
    ax.scatter(mejor_X, Y, marker="x", color="red")
    idx_izq = max(np.where(X == min(mejor_X))[0][0] - 1, 0)
    idx_der = min(np.where(X == max(mejor_X))[0][0] + 1, len(X) - 1)
    ax.set_xlim(X[idx_izq], X[idx_der])
    barra_color = fig.colorbar(CS)
    barra_color.ax.set_ylabel("Score")
    return fig, ax


def parametros_comparados(
    dataset: str,
    clf_base: str = "kdc",
    infos: dict | None = None,
    bi: pd.DataFrame | None = None,
):
    """Construye y guarda CSV de parámetros comparados para un dataset."""
    infos = infos or globals().get("infos") or cargar_infos(config.dir_ejecucion)
    bi = _resolver_info_basica(bi)
    param_base = "bandwidth" if clf_base == "kdc" else "n_neighbors"

    infos_relevantes = {
        k: pd.DataFrame(v[k[2]]["busqueda"].cv_results_)
        for k, v in infos.items()
        if k[0] == dataset and k[2].endswith(clf_base)
    }
    df = pd.concat(
        infos_relevantes.values(),
        keys=infos_relevantes.keys(),
        names=("dataset", "seed", "clf", "main_seed", "scoring", "run"),
    )

    mejores_estimadores = {
        k: v[k[2]]["busqueda"].best_estimator_
        for k, v in infos.items()
        if k[0] == dataset and k[2].endswith(clf_base)
    }
    mejores_params = [
        {
            "clf": k[2],
            "semilla": int(k[1 if dataset in datasets_sinteticos else 3]),
            "alpha": v.get_params().get("alpha", 1),
            param_base: v.get_params()[param_base],
        }
        for k, v in mejores_estimadores.items()
    ]
    mejores_params = pd.DataFrame.from_records(mejores_params).set_index(
        ["semilla", "clf"]
    )

    puntajes = (
        bi[bi.dataset.eq(dataset) & bi.clf.str.endswith(clf_base)]
        .set_index(["semilla", "clf"])
        .r2
    )
    mejores_params["r2"] = puntajes
    mejores_params = mejores_params.reset_index()

    params_comparados = (
        mejores_params.pivot(
            columns="clf",
            index="semilla",
            values=["r2", param_base, "alpha"],
        )
        .drop(columns=("alpha", clf_base))
        .assign(delta_r2=lambda df_: df_.r2[f"f{clf_base}"] - df_.r2[clf_base])
        .sort_values("delta_r2", ascending=False)
        .reorder_levels([1, 0], axis=1)
        .sort_index(axis=1)
    )

    max_alpha_max_puntaje_test = (
        df.xs(f"f{clf_base}", level="clf")
        .query("rank_test_score == 1")
        .param_alpha.groupby("seed")
        .agg("unique")
    )
    max_alpha_max_puntaje_test.index = max_alpha_max_puntaje_test.index.astype(int)
    params_comparados["max_score_alpha_test"] = max_alpha_max_puntaje_test

    ruta = dir_datos / f"{dataset}-parametros_comparados-{clf_base}.csv"
    params_comparados.round(3).to_csv(ruta)
    logger.info(f"Escribió {ruta}")
    return params_comparados


def graficar_fkn_kn_score_vs_n_vecinos(dataset, semilla, ax, infos=None):
    """Grafica mean_test_score vs n_neighbors para fkn y kn."""
    infos = infos or globals().get("infos") or cargar_infos(config.dir_ejecucion)
    puntuacion = "neg_log_loss"
    info_fkn = infos[
        (dataset, str(semilla), "fkn", str(config.semilla_principal), puntuacion)
    ]
    puntaje_fkn = (
        pd.DataFrame(info_fkn["fkn"].busqueda.cv_results_)
        .groupby("param_n_neighbors")
        .mean_test_score.max()
    ).rename("fkn")
    info_kn = infos[
        (dataset, str(semilla), "kn", str(config.semilla_principal), puntuacion)
    ]
    puntaje_kn = (
        pd.DataFrame(info_kn["kn"].busqueda.cv_results_)
        .groupby("param_n_neighbors")
        .mean_test_score.max()
    ).rename("kn")
    pd.concat([puntaje_kn, puntaje_fkn], axis=1).plot(ax=ax)
    ax.set_xscale("log")
    return ax


# ---------------------------------------------------------------------------
# __main__: genera todas las figuras y archivos de datos para la tesis
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.hashsalt"] = "fkdc"

    # --- Cargar datos (con caché) ---
    ruta_cache = dir_cache / "infos_bi.pkl"
    ruta_cache.parent.mkdir(exist_ok=True)
    if ruta_cache.exists():
        logger.info(f"Cargando infos+bi desde caché: {ruta_cache}")
        with open(ruta_cache, "rb") as fp:
            infos, info_basica = pickle.load(fp)
    else:
        logger.info("Cargando infos desde disco (primera ejecución, se cachea)...")
        infos = cargar_infos(dir_ejecucion)
        info_basica = procesar_info_basica(infos, config.semilla_principal)
        with open(ruta_cache, "wb") as fp:
            pickle.dump((infos, info_basica), fp)
        logger.info(f"Cacheado en {ruta_cache}")

    for d in [dir_datos, dir_imagenes]:
        d.mkdir(exist_ok=True)

    semillas = config._obtener_semillas()
    semilla_graficos = semillas[0]
    bi = info_basica  # alias corto

    # Datasets curados usados en la tesis
    todos_datasets = [
        "lunas_lo",
        "circulos_lo",
        "espirales_lo",
        "lunas_hi",
        "circulos_hi",
        "espirales_hi",
        "eslabones_0",
        "helices_0",
        "pionono_0",
        "hueveras_0",
    ]

    # =====================================================================
    # §2 Preliminares: figuras independientes
    # =====================================================================

    # dos-circulos jointplot
    ds_circulos = ConjuntoDatos.de_fabrica(
        make_circles, n_samples=800, noise=0.05, random_state=semilla_graficos
    )
    g = sns.jointplot(
        x=ds_circulos.X[:, 0],
        y=ds_circulos.X[:, 1],
        hue=ds_circulos.y,
        legend=False,
    )
    g.ax_joint.set_xlim(-1.5, 1.5)
    g.ax_joint.set_ylim(-1.5, 1.5)
    guardar_fig(g.figure, dir_imagenes / "dos-circulos-jointplot.svg")

    # maldición de la dimensionalidad
    hs = [0.25, 0.5, 0.9, 0.95]
    rango_ds = np.arange(1, 51, 1)
    df_maldicion = pd.DataFrame(
        [(h, d, h**d) for h in hs for d in rango_ds], columns=["h", "d", "h**d"]
    )
    fig, ax = plt.subplots(figsize=(12, 4), layout="tight")
    df_maldicion.set_index(["d", "h"]).unstack()["h**d"].plot(ax=ax)
    guardar_fig(fig, dir_imagenes / "curse-dim.svg")

    # comparación de kernels (gaussiano vs tophat) para seminario-modesto
    rng = np.random.default_rng(42)
    xs = np.sort(rng.standard_normal(200)).reshape(-1, 1)
    grilla = np.arange(-5, 5, 0.01).reshape(-1, 1)
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True, layout="tight")
    for kernel, ax in zip(["gaussian", "tophat"], axs, strict=False):
        ax.plot(
            grilla,
            sp.stats.norm().pdf(grilla),
            alpha=0.5,
            color="gray",
            linestyle="dashed",
        )
        for bw in [0.1, 0.3, 1, 3]:
            kde = KernelDensity(kernel=kernel, bandwidth=bw).fit(xs)
            dens = np.exp(kde.score_samples(grilla))
            ax.plot(grilla, dens, label=f"h = {bw}", alpha=0.5)
        ax.set_title(f"Kernel = {kernel}")
        ax.legend(loc="upper right")
    guardar_fig(fig, dir_imagenes / "unif-gaus-kern.svg")

    # =====================================================================
    # Gráficos de dispersión (adaptativos por dimensionalidad)
    # =====================================================================
    for dataset in todos_datasets:
        ruta_ds = dir_datasets / f"{dataset}-{semilla_graficos}.pkl"
        with open(ruta_ds, "rb") as fp:
            ds = pickle.load(fp)
        if ds.p == 3:
            # Datasets 3D: dispersar_3d como dispersión por defecto
            fig, ax = plt.subplots(layout="tight", subplot_kw={"projection": "3d"})
            ds.dispersar_3d(ax=ax)
        else:
            # Datasets 2D: dispersión estándar
            fig, ax = plt.subplots(layout="tight")
            ds.dispersar(ax=ax)
        guardar_fig(fig, dir_imagenes / f"{dataset}-scatter.svg")

    # hélices pairplot
    with open(dir_datasets / f"helices_0-{semilla_graficos}.pkl", "rb") as fp:
        ds_helices = pickle.load(fp)
    grafico = ds_helices.grafico_pares(
        dims=[2, 1, 0], height=2, plot_kws={"alpha": 0.5, "s": 5}, corner=True
    )
    guardar_fig(grafico.figure, dir_imagenes / "helices-pairplot.svg")

    # =====================================================================
    # Destacados JSON + Diagramas de caja
    # =====================================================================
    destacar_por = "r2"
    for dataset in todos_datasets:
        hl = obtener_destacados(dataset, por=destacar_por, info=bi)
        excluidos = hl.excluded

        # Guardar destacados JSON
        hl_json = dict(hl)
        hl_json["summary"] = hl.summary.round(4).to_csv()
        nombre_archivo = f"{dataset}-{destacar_por}-highlights.json"
        with open(dir_datos / nombre_archivo, "w") as fp:
            json.dump(hl_json, fp, indent=4)
        logger.info(f"Escribió {dir_datos / nombre_archivo}")

        # Diagramas de caja (R² y accuracy), excluyendo clasificadores no competitivos
        for metrica in ["r2", "accuracy"]:
            fig, ax = plt.subplots(layout="tight")
            diagrama_caja(dataset, metrica, info=bi, ax=ax, excluir_clfs=excluidos)
            guardar_fig(fig, dir_imagenes / f"{dataset}-{metrica}-boxplot.svg")

    # =====================================================================
    # CSVs "mejor-clf-por-dataset" (agregado sobre TODOS los datasets en bi)
    # =====================================================================
    for metrica in ["r2", "accuracy"]:
        mejor = (
            bi.dropna(subset=metrica)
            .groupby(["dataset", "clf"])[metrica]
            .median()
            .sort_values()
            .reset_index("clf")
            .groupby("dataset")
            .last()
            .clf.reset_index()
            .groupby("clf")
            .agg([len, ", ".join])
        )
        mejor.columns = ["cant", "datasets"]
        nombre_archivo = f"mejor-clf-por-dataset-segun-{metrica}-mediano.csv"
        mejor.sort_values("cant", ascending=False).to_csv(dir_datos / nombre_archivo)
        logger.info(f"Escribió {dir_datos / nombre_archivo}")

    # =====================================================================
    # Fronteras de decisión (pares curados)
    # =====================================================================
    todos_clfs = (
        "kdc",
        "fkdc",
        "svc",
        "kn",
        "fkn",
        "gbt",
        "slr",
        "lr",
        "gnb",
    )
    clfs_destacados = ("fkdc", "gbt", "svc")
    curvas_destacadas = ("lunas", "circulos", "espirales")
    pares_frontera = (
        [("espirales_lo", clf) for clf in todos_clfs]
        + [("lunas_lo", "lr")]
        + [
            (f"{curva}_hi", clf)
            for curva in curvas_destacadas
            for clf in clfs_destacados
        ]
    )
    for dataset, clf in pares_frontera:
        fig, ax = plt.subplots(layout="tight")
        frontera_decision(dataset, semilla_graficos, clf, ax=ax)
        guardar_fig(fig, dir_imagenes / f"{dataset}-{clf}-decision_boundary.svg")

    # =====================================================================
    # Gráficos de dispersión R²: fkdc vs kdc, fkn vs kn (curvas 2D)
    # =====================================================================
    for dataset, sufijo in product(
        ("lunas_lo", "circulos_lo", "espirales_lo"), ("kn", "kdc")
    ):
        col_x, col_y = sufijo, f"f{sufijo}"
        datos = (
            bi[bi.dataset.eq(dataset) & bi.clf.str.endswith(sufijo)]
            .set_index(["semilla", "clf"])["r2"]
            .unstack()
        )
        fig, ax = plt.subplots(layout="tight")
        datos.plot(kind="scatter", y=col_y, x=col_x, ax=ax)
        rango = datos.max().max() - datos.min().min()
        x_izq = datos.min()[col_x] - 0.1 * rango
        ax.set_xlim(x_izq)
        ax.set_ylim(datos.min()[col_y] - 0.1 * rango)
        ax.axline((x_izq, x_izq), slope=1, color="gray", linestyle="dotted")
        ax.set_title(f"$R^2$ por semilla para {col_y} y {col_x} en `{dataset}`")
        guardar_fig(fig, dir_imagenes / f"{dataset}-{col_x}-{col_y}-r2-scatter.svg")

    # Dispersión R²: fkdc vs kdc para helices_0
    datos_disp = (
        bi[bi.dataset.eq("helices_0") & bi.clf.isin(["fkdc", "kdc"])]
        .set_index(["semilla", "clf"])["r2"]
        .unstack()
    )
    fig, ax = plt.subplots(layout="tight")
    datos_disp.plot(kind="scatter", y="fkdc", x="kdc", ax=ax)
    rango = datos_disp.max().max() - datos_disp.min().min()
    x_izq = datos_disp.min()["kdc"] - 0.1 * rango
    ax.set_xlim(x_izq)
    ax.set_ylim(datos_disp.min()["fkdc"] - 0.1 * rango)
    ax.axline((x_izq, x_izq), slope=1, color="gray", linestyle="dotted")
    ax.set_title("$R^2$ por semilla para fkdc y kdc en `helices_0`")
    guardar_fig(fig, dir_imagenes / "helices_0-r2-fkdc-vs-kdc.svg")

    # Dispersión R²: fkn vs kn para helices_0 y eslabones_0 (seminario-modesto)
    for dataset, nombre_archivo in [
        ("helices_0", "r2-fkn-kn-helices_0.svg"),
        ("eslabones_0", "eslabones_0-r2-fkn-vs-kn.svg"),
    ]:
        datos_disp = (
            bi[bi.dataset.eq(dataset) & bi.clf.isin(["fkn", "kn"])]
            .set_index(["semilla", "clf"])["r2"]
            .unstack()
        )
        fig, ax = plt.subplots(layout="tight")
        datos_disp.plot(kind="scatter", y="fkn", x="kn", ax=ax)
        rango = datos_disp.max().max() - datos_disp.min().min()
        x_izq = datos_disp.min()["kn"] - 0.1 * rango
        ax.set_xlim(x_izq)
        ax.set_ylim(datos_disp.min()["fkn"] - 0.1 * rango)
        ax.axline((x_izq, x_izq), slope=1, color="gray", linestyle="dotted")
        ax.set_title(f"$R^2$ por semilla para FKN y KN en `{dataset}`")
        guardar_fig(fig, dir_imagenes / nombre_archivo)

    # =====================================================================
    # Superficies de contorno de pérdida (curadas)
    # =====================================================================
    clf_lc, x_lc, y_lc = "fkdc", "bandwidth", "alpha"
    semillas_2d = [7354, 8527, 1188]
    curvas_2d = ["lunas", "circulos", "espirales"]
    for curva, semilla in product(curvas_2d, semillas_2d):
        dataset = f"{curva}_lo"
        fig, ax = contorno_perdida(dataset, semilla, clf_lc, x_lc, y_lc)
        ax.set_xscale("log")
        nombre_archivo = f"{dataset}-{semilla}-{clf_lc}-{x_lc}-{y_lc}-loss_contour.svg"
        guardar_fig(fig, dir_imagenes / nombre_archivo)

    # Contornos de pérdida: helices_0 (tesis), espirales_lo (seminario)
    for dataset, semilla in [("helices_0", 1188), ("espirales_lo", 1434)]:
        fig, ax = contorno_perdida(dataset, semilla, clf_lc, x_lc, y_lc)
        ax.set_xscale("log")
        nombre_archivo = f"{dataset}-{semilla}-{clf_lc}-{x_lc}-{y_lc}-loss_contour.svg"
        guardar_fig(fig, dir_imagenes / nombre_archivo)

    # =====================================================================
    # lunas_lo: mejores_params + score vs bandwidth
    # =====================================================================
    dataset = "lunas_lo"
    clf_base_lu, param_base_lu = "kdc", "bandwidth"
    mejores_estimadores_lu = {
        k: v[k[2]]["busqueda"].best_estimator_
        for k, v in infos.items()
        if k[0] == dataset and k[2].endswith(clf_base_lu)
    }
    mejores_params_lu = pd.DataFrame.from_records(
        [
            {
                "clf": k[2],
                "semilla": int(k[1 if dataset in datasets_sinteticos else 3]),
                "alpha": v.get_params().get("alpha", 1),
                param_base_lu: v.get_params()[param_base_lu],
            }
            for k, v in mejores_estimadores_lu.items()
        ]
    ).set_index(["semilla", "clf"])
    puntajes_lu = (
        bi[bi.dataset.eq(dataset) & bi.clf.str.endswith(clf_base_lu)]
        .set_index(["semilla", "clf"])
        .r2
    )
    mejores_params_lu["r2"] = puntajes_lu
    mejores_params_lu = mejores_params_lu.reset_index()

    # CSV mejores_params (conteos de valores)
    (
        mejores_params_lu[["clf", "alpha", "bandwidth"]]
        .value_counts()
        .sort_index()
        .reset_index()
        .round(4)
        .to_csv(dir_datos / f"{dataset}-best_params.csv", index=False)
    )
    logger.info(f"Escribió {dir_datos / f'{dataset}-best_params.csv'}")

    # CSV mejores_params_test
    infos_relevantes_lu = {
        k: pd.DataFrame(v[k[2]]["busqueda"].cv_results_)
        for k, v in infos.items()
        if k[0] == dataset and k[2].endswith(clf_base_lu)
    }
    df_lunas = pd.concat(
        infos_relevantes_lu.values(),
        keys=infos_relevantes_lu.keys(),
        names=("dataset", "seed", "clf", "main_seed", "scoring", "run"),
    )
    df_lunas.xs("fkdc", level="clf").query("rank_test_score == 1")[
        ["param_alpha", "param_bandwidth"]
    ].round(4).value_counts().sort_index().reset_index().rename(
        columns=lambda s: s.replace("param_", "")
    ).to_csv(dir_datos / f"{dataset}-best_test_params.csv", index=False)
    logger.info(f"Escribió {dir_datos / f'{dataset}-best_test_params.csv'}")

    # Dispersión score vs bandwidth
    fig, ax = plt.subplots(layout="tight")
    sns.scatterplot(data=mejores_params_lu, x="bandwidth", y="r2", hue="clf", ax=ax)
    ax.set_xscale("log")
    ax.set_title(f"$R^2$ vs bandwidth para kdc y fkdc en `{dataset}`")
    guardar_fig(fig, dir_imagenes / f"{dataset}-[f]kdc-score-vs-bandwidth.svg")

    # delta R² vs delta h
    cp = mejores_params_lu.pivot(
        columns="clf", index="semilla", values=["r2", "bandwidth"]
    )
    cp["delta_r2"] = cp["r2"]["fkdc"] - cp["r2"]["kdc"]
    cp["delta_h"] = cp["bandwidth"]["fkdc"] - cp["bandwidth"]["kdc"]
    fig, ax = plt.subplots(layout="tight")
    ax.scatter(cp["delta_h"], cp["delta_r2"])
    ax.axhline(0, color="gray", linestyle="dotted", alpha=0.5)
    ax.axvline(0, color="gray", linestyle="dotted", alpha=0.5)
    ax.set_xlabel("$\\Delta h$ (fkdc − kdc)")
    ax.set_ylabel("$\\Delta R^2$ (fkdc − kdc)")
    ax.set_title(f"$\\Delta R^2$ vs $\\Delta h$ en `{dataset}`")
    guardar_fig(fig, dir_imagenes / f"{dataset}-[f]kdc-delta_r2-vs-delta_h.svg")

    # =====================================================================
    # Caída de R² (lo vs hi)
    # =====================================================================
    bi2d = bi[bi.dataset.str.endswith(("_lo", "_hi"))].copy()
    bi2d[["figura", "ruido"]] = bi2d.dataset.str.split("_", expand=True)
    caidas = (
        bi2d.groupby(["figura", "ruido", "clf"])[["r2", "accuracy"]]
        .mean()
        .unstack("ruido")
    )
    # Excluir clasificadores que no se diferencian de cero en ningún nivel de ruido
    for figura in bi2d.figura.unique():
        caidas_fig = caidas.xs(figura)["r2"][["lo", "hi"]]
        # Mantener solo clfs con R² significativo en al menos un nivel de ruido
        lo_ok = caidas_fig["lo"].abs() > 0.05
        hi_ok = caidas_fig["hi"].abs() > 0.05
        significativos = caidas_fig[lo_ok | hi_ok]
        fig, ax = plt.subplots(layout="tight")
        significativos.sort_values("hi", ascending=False).plot(kind="bar", ax=ax)
        guardar_fig(fig, dir_imagenes / f"{figura}-caida_r2.svg")

    # =====================================================================
    # helices_0: diagrama de caja R² ampliado (solo clasificadores kernel)
    # =====================================================================
    fig, ax = plt.subplots(layout="tight")
    datos_caja = bi[
        bi.dataset.eq("helices_0") & bi.clf.isin(["fkdc", "kdc", "kn", "fkn"])
    ].sort_values("clf")
    sns.boxplot(
        datos_caja,
        hue="clf",
        y="r2",
        gap=0.2,
        ax=ax,
        palette=paleta_predeterminada,
        saturation=1.0,
    )
    aplicar_sombreado(ax)
    ax.axhline(
        datos_caja.groupby("clf")["r2"].median().max(),
        linestyle="dotted",
        color="gray",
    )
    y_inf = np.percentile(datos_caja["r2"].dropna(), 10)
    ax.set_ylim(y_inf, None)
    guardar_fig(fig, dir_imagenes / "helices_0-boxplot-r2-zoomed.svg")

    # =====================================================================
    # eslabones_0: parámetros para una semilla específica
    # =====================================================================
    dataset_esl = "eslabones_0"
    semilla_esl = 2411
    mejores_estimadores_esl = {
        k: v[k[2]]["busqueda"].best_estimator_
        for k, v in infos.items()
        if k[0] == dataset_esl and k[2].endswith("kdc")
    }
    mejores_params_esl = pd.DataFrame.from_records(
        [
            {
                "clf": k[2],
                "semilla": int(k[1 if dataset_esl in datasets_sinteticos else 3]),
                "alpha": v.get_params().get("alpha", 1),
                "bandwidth": v.get_params()["bandwidth"],
            }
            for k, v in mejores_estimadores_esl.items()
        ]
    ).set_index(["semilla", "clf"])
    puntajes_esl = (
        bi[bi.dataset.eq(dataset_esl) & bi.clf.str.endswith("kdc")]
        .set_index(["semilla", "clf"])
        .r2
    )
    mejores_params_esl["r2"] = puntajes_esl
    ruta_esl = dir_datos / f"{dataset_esl}-params-{semilla_esl}.csv"
    mejores_params_esl.loc[pd.IndexSlice[semilla_esl, :]].round(4).to_csv(ruta_esl)
    logger.info(f"Escribió {ruta_esl}")

    # =====================================================================
    # CSVs de parámetros comparados
    # =====================================================================
    parametros_comparados("helices_0", "kdc", infos=infos, bi=bi)
    parametros_comparados("hueveras_0", "kdc", infos=infos, bi=bi)
    parametros_comparados("hueveras_0", "kn", infos=infos, bi=bi)

    # =====================================================================
    # Seminario-modesto: score vs n_neighbors
    # =====================================================================
    for dataset in ["helices_0", "eslabones_0"]:
        fig, ax = plt.subplots(figsize=(10, 5), layout="tight")
        graficar_fkn_kn_score_vs_n_vecinos(dataset, 2411, ax, infos=infos)
        guardar_fig(fig, dir_imagenes / f"{dataset}-fkn_kn-mean_test_score.svg")

    logger.info("Listo.")
