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

from fkdc import cache_dir, config, root_dir
from fkdc.datasets import Dataset, found_datasets, synth_datasets
from fkdc.tarea import Tarea

# TODO: Retrain with consistent version to avoid issue
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

logger = logging.getLogger(__name__)
run_dir = config.run_dir
datasets_dir = run_dir / "../datasets"
img_dir = root_dir / "docs" / "img"
data_dir = root_dir / "docs" / "data"

datasets = [*synth_datasets, *found_datasets]
clfs = list(config.clasificadores.keys())
infos = None
basic_info = None

# Palette: paired classifiers share the same base color, variants use hatching.
# Families: dc (fkdc/kdc), kn (fkn/kn), logr (lr/slr), gbt, gnb, svc
_clf_families = {
    "dc": ["fkdc", "kdc"],
    "kn": ["fkn", "kn"],
    "logr": ["lr", "slr"],
    "gbt": ["gbt"],
    "gnb": ["gnb"],
    "svc": ["svc"],
}
_family_colors = dict(zip(_clf_families, sns.color_palette("Pastel2", 6), strict=True))
default_palette = {
    clf: _family_colors[family]
    for family, members in _clf_families.items()
    for clf in members
}
_hatched_clfs = {"kdc", "kn", "slr"}


def save_fig(fig: Figure, fpath: str | Path, **savefig_kw):
    """Save figure, log, and close."""
    savefig_kw.setdefault("bbox_inches", "tight")
    fig.savefig(fpath, **savefig_kw)
    logger.info(f"Wrote {fpath}")
    plt.close(fig)


def load_infos(dir_or_paths: list[Path] | Path):
    if isinstance(dir_or_paths, Path):
        assert dir_or_paths.is_dir()
        paths = dir_or_paths.glob("*.pkl")
    elif isinstance(dir_or_paths, list):
        paths = dir_or_paths
    else:
        raise ValueError(
            "`dir_or_paths` debe ser un directorio con pickles de info "
            "o una lista de rutas a pickles de info"
        )
    return {tuple(fn.stem.split("-")): pickle.load(open(fn, "rb")) for fn in paths}


def parse_basic_info(infos: dict, main_seed: int = config.main_seed):
    basic_fields = ["accuracy", "r2", "logvero"]
    basic_infos = {}
    for k, v in infos.items():
        clf = k[2]
        basic_infos[k] = {k: v for k, v in v[clf].items() if k in basic_fields}

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


def _render_basic_info(info: pd.DataFrame | None):
    if info is None:
        logger.debug("No info passed, defaulting to basic_info")
        global infos, basic_info
        if infos is None:
            logger.warning("Infos not parsed yet; parsing...")
            infos = load_infos(config.run_dir)
        if basic_info is None:
            logger.warning("basic_info not extracted yet, extracting...")
            basic_info = parse_basic_info(infos, config.main_seed)
        info: pd.DataFrame = basic_info
    return info


def get_highlights(
    dataset: str, by: str = "r2", info: None | pd.DataFrame = None
) -> Bunch:
    info = _render_basic_info(info)
    this_dataset = info["dataset"].eq(dataset)
    summary = (
        info[this_dataset]
        .groupby("clf")[["accuracy", "r2"]]
        .median()
        .sort_values(by=by, ascending=False)
    )
    best = summary.idxmax()[by]
    best_runs = info[this_dataset & info["clf"].eq(best)][by]
    first_quartile = np.percentile(best_runs, 25)
    best_min = best_runs.min()
    logger.debug(f"best: {best} (by {by}); p25: {first_quartile}; min: {best_min}")
    bad = summary[summary[by].lt(first_quartile)].index.tolist()
    excluded = summary[summary[by].lt(best_min)].index.tolist()
    return Bunch(
        dataset=dataset,
        summary=summary,
        best=best,
        bad=bad,
        excluded=excluded,
    )


def boxplot(
    dataset: str,
    metric: str = "r2",
    info: pd.DataFrame | None = None,
    ax=None,
    palette: dict | None = None,
    exclude_clfs: list[str] | None = None,
):
    palette = palette or default_palette
    info = _render_basic_info(info)

    if ax is None:
        ax = plt.gca()
    data = info[info.dataset.eq(dataset)].sort_values("clf").dropna(subset=metric)
    if exclude_clfs:
        data = data[~data.clf.isin(exclude_clfs)]
    sns.boxplot(
        data, hue="clf", y=metric, gap=0.2, ax=ax, palette=palette, saturation=1.0
    )
    apply_hatching(ax)
    ax.axhline(
        data.groupby("clf")[metric].median().max(), linestyle="dotted", color="gray"
    )


def apply_hatching(ax, hatched_clfs=None):
    """Apply hatching to boxplot boxes for variant classifiers."""
    hatched_clfs = hatched_clfs or _hatched_clfs
    legend = ax.get_legend()
    if legend is None:
        return
    labels = [t.get_text() for t in legend.get_texts()]
    handles = legend.legend_handles
    box_patches = [p for p in ax.patches if isinstance(p, PathPatch)]
    for i, label in enumerate(labels):
        if label in hatched_clfs:
            if i < len(box_patches):
                box_patches[i].set_hatch("///")
            handles[i].set_hatch("///")


def decision_boundary(
    dataset: str,
    seed: int,
    clf: str,
    ax: Axes | None = None,
    cmap: str | Colormap | None = None,
):
    ax = ax or plt.gca()
    cmap = cmap or colormaps["coolwarm"]

    if isinstance(cmap, str):
        cmap = colormaps[cmap]
    if clf == "svc":
        response_method, scoring = "predict", "accuracy"
    elif clf in clfs:
        response_method, scoring = "predict_proba", "neg_log_loss"
    else:
        raise ValueError(f"Clf not known: {clf}")
    if dataset in synth_datasets:
        ds_path = datasets_dir / f"{dataset}-{seed}.pkl"
        plotting_key = (dataset, str(seed), clf, str(config.main_seed), scoring)
    elif dataset in found_datasets:
        ds_path = datasets_dir / f"{dataset}.pkl"
        plotting_key = (dataset, str(None), clf, str(seed), scoring)
    else:
        raise ValueError(f"Dataset not known: {dataset}-{seed}")
    with open(ds_path, "rb") as fp:
        ds: Dataset = pickle.load(fp)
    tarea = Tarea(ds, {}, seed=seed, split_evaluacion=config.split_evaluacion)
    X = tarea.X_eval
    y = tarea.y_eval
    k = ds.k
    X0, X1 = X[:, 0], X[:, 1]
    with open(
        config.run_dir / f"{'-'.join(map(str, plotting_key))}.pkl", "rb"
    ) as fpath:
        info: dict = pickle.load(fpath)
        best_est: BaseEstimator = info[clf]["busqueda"].best_estimator_
    DecisionBoundaryDisplay.from_estimator(
        best_est,
        X,
        eps=0.05,
        response_method=response_method if k == 2 else "predict",
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


def loss_contour(
    dataset: str,
    seed: int,
    clf: str,
    x: str,
    y: str,
    other_params: dict | None = None,
    cmap: str | Colormap = "viridis",
):
    cmap = cmap or "viridis"
    if isinstance(cmap, str):
        cmap = colormaps[cmap]
    scoring = "accuracy" if clf == "svc" else "neg_log_loss"
    if clf not in clfs:
        raise ValueError(f"Clf not known: {clf}")
    if dataset in synth_datasets:
        plotting_key = (dataset, str(seed), clf, str(config.main_seed), scoring)
    elif dataset in found_datasets:
        plotting_key = (dataset, str(None), clf, str(seed), scoring)
    else:
        raise ValueError(f"Dataset not known: {dataset}, {seed}")
    fpath = config.run_dir / f"{'-'.join(map(str, plotting_key))}.pkl"
    with open(fpath, "rb") as fp:
        info: dict = pickle.load(fp)
    cv_results = pd.DataFrame(info[clf]["busqueda"].cv_results_)

    data = cv_results.copy()
    other_params = other_params or {}
    if other_params:
        for p, value in other_params.items():
            data = data[data[f"param_{p}"] == value]
    data = data.set_index([f"param_{y}", f"param_{x}"]).mean_test_score.unstack()
    X = data.columns.values
    Y = data.index.values
    Z = data.values

    fig, ax = plt.subplots(layout="constrained")
    CS = ax.contourf(X, Y, Z, 15, cmap=cmap)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    best_X = X[Z.argmax(axis=1)]
    ax.scatter(best_X, Y, marker="x", color="red")
    left_index = max(np.where(X == min(best_X))[0][0] - 1, 0)
    right_index = min(np.where(X == max(best_X))[0][0] + 1, len(X) - 1)
    ax.set_xlim(X[left_index], X[right_index])
    cbar = fig.colorbar(CS)
    cbar.ax.set_ylabel("Score")
    return fig, ax


def parametros_comparados(
    dataset: str,
    base_clf: str = "kdc",
    infos: dict | None = None,
    bi: pd.DataFrame | None = None,
):
    """Build and save parametros_comparados CSV for a dataset."""
    infos = infos or globals().get("infos") or load_infos(config.run_dir)
    bi = _render_basic_info(bi)
    base_param = "bandwidth" if base_clf == "kdc" else "n_neighbors"

    infos_relevantes = {
        k: pd.DataFrame(v[k[2]]["busqueda"].cv_results_)
        for k, v in infos.items()
        if k[0] == dataset and k[2].endswith(base_clf)
    }
    df = pd.concat(
        infos_relevantes.values(),
        keys=infos_relevantes.keys(),
        names=("dataset", "seed", "clf", "main_seed", "scoring", "run"),
    )

    best_estimators = {
        k: v[k[2]]["busqueda"].best_estimator_
        for k, v in infos.items()
        if k[0] == dataset and k[2].endswith(base_clf)
    }
    best_params = [
        {
            "clf": k[2],
            "semilla": int(k[1 if dataset in synth_datasets else 3]),
            "alpha": v.get_params().get("alpha", 1),
            base_param: v.get_params()[base_param],
        }
        for k, v in best_estimators.items()
    ]
    best_params = pd.DataFrame.from_records(best_params).set_index(["semilla", "clf"])

    scores = (
        bi[bi.dataset.eq(dataset) & bi.clf.str.endswith(base_clf)]
        .set_index(["semilla", "clf"])
        .r2
    )
    best_params["r2"] = scores
    best_params = best_params.reset_index()

    compared_params = (
        best_params.pivot(
            columns="clf", index="semilla", values=["r2", base_param, "alpha"]
        )
        .drop(columns=("alpha", base_clf))
        .assign(delta_r2=lambda df_: df_.r2[f"f{base_clf}"] - df_.r2[base_clf])
        .sort_values("delta_r2", ascending=False)
        .reorder_levels([1, 0], axis=1)
        .sort_index(axis=1)
    )

    max_alpha_max_test_score = (
        df.xs(f"f{base_clf}", level="clf")
        .query("rank_test_score == 1")
        .param_alpha.groupby("seed")
        .agg("unique")
    )
    max_alpha_max_test_score.index = max_alpha_max_test_score.index.astype(int)
    compared_params["max_score_alpha_test"] = max_alpha_max_test_score

    fpath = data_dir / f"{dataset}-parametros_comparados-{base_clf}.csv"
    compared_params.round(3).to_csv(fpath)
    logger.info(f"Wrote {fpath}")
    return compared_params


def plot_fkn_kn_score_vs_n_neighbors(dataset, seed, ax, infos=None):
    """Plot mean_test_score vs n_neighbors for fkn and kn."""
    infos = infos or globals().get("infos") or load_infos(config.run_dir)
    scoring = "neg_log_loss"
    info_fkn = infos[(dataset, str(seed), "fkn", str(config.main_seed), scoring)]
    score_fkn = (
        pd.DataFrame(info_fkn["fkn"].busqueda.cv_results_)
        .groupby("param_n_neighbors")
        .mean_test_score.max()
    ).rename("fkn")
    info_kn = infos[(dataset, str(seed), "kn", str(config.main_seed), scoring)]
    score_kn = (
        pd.DataFrame(info_kn["kn"].busqueda.cv_results_)
        .groupby("param_n_neighbors")
        .mean_test_score.max()
    ).rename("kn")
    pd.concat([score_kn, score_fkn], axis=1).plot(ax=ax)
    ax.set_xscale("log")
    return ax


# ---------------------------------------------------------------------------
# __main__: generate all figures and data files for the thesis
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

    # --- Load data (with cache) ---
    cache_path = cache_dir / "infos_bi.pkl"
    cache_path.parent.mkdir(exist_ok=True)
    if cache_path.exists():
        logger.info(f"Loading cached infos+bi from {cache_path}")
        with open(cache_path, "rb") as fp:
            infos, basic_info = pickle.load(fp)
    else:
        logger.info("Loading infos from disk (first run, will cache)...")
        infos = load_infos(run_dir)
        basic_info = parse_basic_info(infos, config.main_seed)
        with open(cache_path, "wb") as fp:
            pickle.dump((infos, basic_info), fp)
        logger.info(f"Cached to {cache_path}")

    for d in [data_dir, img_dir]:
        d.mkdir(exist_ok=True)

    run_seeds = config._get_run_seeds()
    plotting_seed = run_seeds[0]
    bi = basic_info  # short alias

    # Curated datasets used in the thesis
    all_datasets = [
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
    # §2 Preliminares: standalone figures
    # =====================================================================

    # dos-circulos jointplot
    ds_circles = Dataset.de_fabrica(
        make_circles, n_samples=800, noise=0.05, random_state=plotting_seed
    )
    g = sns.jointplot(
        x=ds_circles.X[:, 0], y=ds_circles.X[:, 1], hue=ds_circles.y, legend=False
    )
    g.ax_joint.set_xlim(-1.5, 1.5)
    g.ax_joint.set_ylim(-1.5, 1.5)
    save_fig(g.figure, img_dir / "dos-circulos-jointplot.svg")

    # curse of dimensionality
    hs = [0.25, 0.5, 0.9, 0.95]
    ds_range = np.arange(1, 51, 1)
    df_curse = pd.DataFrame(
        [(h, d, h**d) for h in hs for d in ds_range], columns=["h", "d", "h**d"]
    )
    fig, ax = plt.subplots(figsize=(12, 4), layout="tight")
    df_curse.set_index(["d", "h"]).unstack()["h**d"].plot(ax=ax)
    save_fig(fig, img_dir / "curse-dim.svg")

    # kernel comparison (gaussian vs tophat) for seminario-modesto
    rng = np.random.default_rng(42)
    xs = np.sort(rng.standard_normal(200)).reshape(-1, 1)
    grid = np.arange(-5, 5, 0.01).reshape(-1, 1)
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True, layout="tight")
    for kernel, ax in zip(["gaussian", "tophat"], axs, strict=False):
        ax.plot(
            grid, sp.stats.norm().pdf(grid), alpha=0.5, color="gray", linestyle="dashed"
        )
        for bw in [0.1, 0.3, 1, 3]:
            kde = KernelDensity(kernel=kernel, bandwidth=bw).fit(xs)
            dens = np.exp(kde.score_samples(grid))
            ax.plot(grid, dens, label=f"h = {bw}", alpha=0.5)
        ax.set_title(f"Kernel = {kernel}")
        ax.legend(loc="upper right")
    save_fig(fig, img_dir / "unif-gaus-kern.svg")

    # =====================================================================
    # Scatter plots (adaptive by dimensionality)
    # =====================================================================
    for dataset in all_datasets:
        ds_path = datasets_dir / f"{dataset}-{plotting_seed}.pkl"
        with open(ds_path, "rb") as fp:
            ds = pickle.load(fp)
        if ds.p == 3:
            # 3D datasets: scatter_3d as the default scatter
            fig, ax = plt.subplots(layout="tight", subplot_kw={"projection": "3d"})
            ds.scatter_3d(ax=ax)
        else:
            # 2D datasets: standard scatter
            fig, ax = plt.subplots(layout="tight")
            ds.scatter(ax=ax)
        save_fig(fig, img_dir / f"{dataset}-scatter.svg")

    # helices pairplot
    with open(datasets_dir / f"helices_0-{plotting_seed}.pkl", "rb") as fp:
        ds_helices = pickle.load(fp)
    graph = ds_helices.pairplot(
        dims=[2, 1, 0], height=2, plot_kws={"alpha": 0.5, "s": 5}, corner=True
    )
    save_fig(graph.figure, img_dir / "helices-pairplot.svg")

    # =====================================================================
    # Highlights JSON + Boxplots
    # =====================================================================
    highlights_by = "r2"
    for dataset in all_datasets:
        hl = get_highlights(dataset, by=highlights_by, info=bi)
        excluded = hl.excluded

        # Save highlights JSON
        hl_json = dict(hl)
        hl_json["summary"] = hl.summary.round(4).to_csv()
        fname = f"{dataset}-{highlights_by}-highlights.json"
        with open(data_dir / fname, "w") as fp:
            json.dump(hl_json, fp, indent=4)
        logger.info(f"Wrote {data_dir / fname}")

        # Boxplots (R² and accuracy), excluding non-competitive classifiers
        for metric in ["r2", "accuracy"]:
            fig, ax = plt.subplots(layout="tight")
            boxplot(dataset, metric, info=bi, ax=ax, exclude_clfs=excluded)
            save_fig(fig, img_dir / f"{dataset}-{metric}-boxplot.svg")

    # =====================================================================
    # "mejor-clf-por-dataset" CSVs (aggregate over ALL datasets in bi)
    # =====================================================================
    for metric in ["r2", "accuracy"]:
        best = (
            bi.dropna(subset=metric)
            .groupby(["dataset", "clf"])[metric]
            .median()
            .sort_values()
            .reset_index("clf")
            .groupby("dataset")
            .last()
            .clf.reset_index()
            .groupby("clf")
            .agg([len, ", ".join])
        )
        best.columns = ["cant", "datasets"]
        fname = f"mejor-clf-por-dataset-segun-{metric}-mediano.csv"
        best.sort_values("cant", ascending=False).to_csv(data_dir / fname)
        logger.info(f"Wrote {data_dir / fname}")

    # =====================================================================
    # Decision boundaries (curated pairs only)
    # =====================================================================
    all_clfs = ("kdc", "fkdc", "svc", "kn", "fkn", "gbt", "slr", "lr", "gnb")
    hi_clfs = ("fkdc", "gbt", "svc")
    hi_curves = ("lunas", "circulos", "espirales")
    db_pairs = (
        [("espirales_lo", clf) for clf in all_clfs]
        + [("lunas_lo", "lr")]
        + [(f"{curve}_hi", clf) for curve in hi_curves for clf in hi_clfs]
    )
    for dataset, clf in db_pairs:
        fig, ax = plt.subplots(layout="tight")
        decision_boundary(dataset, plotting_seed, clf, ax=ax)
        save_fig(fig, img_dir / f"{dataset}-{clf}-decision_boundary.svg")

    # =====================================================================
    # R² scatter plots: fkdc vs kdc, fkn vs kn (2D curves)
    # =====================================================================
    for dataset, sufijo in product(
        ("lunas_lo", "circulos_lo", "espirales_lo"), ("kn", "kdc")
    ):
        x_col, y_col = sufijo, f"f{sufijo}"
        data = (
            bi[bi.dataset.eq(dataset) & bi.clf.str.endswith(sufijo)]
            .set_index(["semilla", "clf"])["r2"]
            .unstack()
        )
        fig, ax = plt.subplots(layout="tight")
        data.plot(kind="scatter", y=y_col, x=x_col, ax=ax)
        range_ = data.max().max() - data.min().min()
        x_left = data.min()[x_col] - 0.1 * range_
        ax.set_xlim(x_left)
        ax.set_ylim(data.min()[y_col] - 0.1 * range_)
        ax.axline((x_left, x_left), slope=1, color="gray", linestyle="dotted")
        ax.set_title(f"$R^2$ por semilla para {y_col} y {x_col} en `{dataset}`")
        save_fig(fig, img_dir / f"{dataset}-{x_col}-{y_col}-r2-scatter.svg")

    # R² scatter: fkdc vs kdc for helices_0
    data_scatter = (
        bi[bi.dataset.eq("helices_0") & bi.clf.isin(["fkdc", "kdc"])]
        .set_index(["semilla", "clf"])["r2"]
        .unstack()
    )
    fig, ax = plt.subplots(layout="tight")
    data_scatter.plot(kind="scatter", y="fkdc", x="kdc", ax=ax)
    range_ = data_scatter.max().max() - data_scatter.min().min()
    x_left = data_scatter.min()["kdc"] - 0.1 * range_
    ax.set_xlim(x_left)
    ax.set_ylim(data_scatter.min()["fkdc"] - 0.1 * range_)
    ax.axline((x_left, x_left), slope=1, color="gray", linestyle="dotted")
    ax.set_title("$R^2$ por semilla para fkdc y kdc en `helices_0`")
    save_fig(fig, img_dir / "helices_0-r2-fkdc-vs-kdc.svg")

    # R² scatter: fkn vs kn for helices_0 and eslabones_0 (seminario-modesto)
    for dataset, fname in [
        ("helices_0", "r2-fkn-kn-helices_0.svg"),
        ("eslabones_0", "eslabones_0-r2-fkn-vs-kn.svg"),
    ]:
        data_scatter = (
            bi[bi.dataset.eq(dataset) & bi.clf.isin(["fkn", "kn"])]
            .set_index(["semilla", "clf"])["r2"]
            .unstack()
        )
        fig, ax = plt.subplots(layout="tight")
        data_scatter.plot(kind="scatter", y="fkn", x="kn", ax=ax)
        range_ = data_scatter.max().max() - data_scatter.min().min()
        x_left = data_scatter.min()["kn"] - 0.1 * range_
        ax.set_xlim(x_left)
        ax.set_ylim(data_scatter.min()["fkn"] - 0.1 * range_)
        ax.axline((x_left, x_left), slope=1, color="gray", linestyle="dotted")
        ax.set_title(f"$R^2$ por semilla para FKN y KN en `{dataset}`")
        save_fig(fig, img_dir / fname)

    # =====================================================================
    # Loss contour surfaces (curated)
    # =====================================================================
    clf_lc, x_lc, y_lc = "fkdc", "bandwidth", "alpha"
    semillas_2d = [7354, 8527, 1188]
    curvas_2d = ["lunas", "circulos", "espirales"]
    for curve, seed in product(curvas_2d, semillas_2d):
        dataset = f"{curve}_lo"
        fig, ax = loss_contour(dataset, seed, clf_lc, x_lc, y_lc)
        ax.set_xscale("log")
        fname = f"{dataset}-{seed}-{clf_lc}-{x_lc}-{y_lc}-loss_contour.svg"
        save_fig(fig, img_dir / fname)

    # Standalone loss contours (helices_0 for tesis, espirales_lo for seminario)
    for dataset, seed in [("helices_0", 1188), ("espirales_lo", 1434)]:
        fig, ax = loss_contour(dataset, seed, clf_lc, x_lc, y_lc)
        ax.set_xscale("log")
        fname = f"{dataset}-{seed}-{clf_lc}-{x_lc}-{y_lc}-loss_contour.svg"
        save_fig(fig, img_dir / fname)

    # =====================================================================
    # lunas_lo: best_params + score vs bandwidth analysis
    # =====================================================================
    dataset = "lunas_lo"
    base_clf_lu, base_param_lu = "kdc", "bandwidth"
    best_estimators_lu = {
        k: v[k[2]]["busqueda"].best_estimator_
        for k, v in infos.items()
        if k[0] == dataset and k[2].endswith(base_clf_lu)
    }
    best_params_lu = pd.DataFrame.from_records(
        [
            {
                "clf": k[2],
                "semilla": int(k[1 if dataset in synth_datasets else 3]),
                "alpha": v.get_params().get("alpha", 1),
                base_param_lu: v.get_params()[base_param_lu],
            }
            for k, v in best_estimators_lu.items()
        ]
    ).set_index(["semilla", "clf"])
    scores_lu = (
        bi[bi.dataset.eq(dataset) & bi.clf.str.endswith(base_clf_lu)]
        .set_index(["semilla", "clf"])
        .r2
    )
    best_params_lu["r2"] = scores_lu
    best_params_lu = best_params_lu.reset_index()

    # best_params CSV (value counts)
    (
        best_params_lu[["clf", "alpha", "bandwidth"]]
        .value_counts()
        .sort_index()
        .reset_index()
        .round(4)
        .to_csv(data_dir / f"{dataset}-best_params.csv", index=False)
    )
    logger.info(f"Wrote {data_dir / f'{dataset}-best_params.csv'}")

    # best_test_params CSV
    infos_relevantes_lu = {
        k: pd.DataFrame(v[k[2]]["busqueda"].cv_results_)
        for k, v in infos.items()
        if k[0] == dataset and k[2].endswith(base_clf_lu)
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
    ).to_csv(data_dir / f"{dataset}-best_test_params.csv", index=False)
    logger.info(f"Wrote {data_dir / f'{dataset}-best_test_params.csv'}")

    # score vs bandwidth scatter
    fig, ax = plt.subplots(layout="tight")
    sns.scatterplot(data=best_params_lu, x="bandwidth", y="r2", hue="clf", ax=ax)
    ax.set_xscale("log")
    ax.set_title(f"$R^2$ vs bandwidth para kdc y fkdc en `{dataset}`")
    save_fig(fig, img_dir / f"{dataset}-[f]kdc-score-vs-bandwidth.svg")

    # delta R² vs delta h
    cp = best_params_lu.pivot(
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
    save_fig(fig, img_dir / f"{dataset}-[f]kdc-delta_r2-vs-delta_h.svg")

    # =====================================================================
    # Caída de R² (lo vs hi)
    # =====================================================================
    bi2d = bi[bi.dataset.str.endswith(("_lo", "_hi"))].copy()
    bi2d[["figura", "ruido"]] = bi2d.dataset.str.split("_", expand=True)
    drops = (
        bi2d.groupby(["figura", "ruido", "clf"])[["r2", "accuracy"]]
        .mean()
        .unstack("ruido")
    )
    # Exclude classifiers that don't differentiate from zero in either lo or hi
    for figura in bi2d.figura.unique():
        fig_drops = drops.xs(figura)["r2"][["lo", "hi"]]
        # Keep only clfs with meaningful R² in at least one noise level
        lo_ok = fig_drops["lo"].abs() > 0.05
        hi_ok = fig_drops["hi"].abs() > 0.05
        meaningful = fig_drops[lo_ok | hi_ok]
        fig, ax = plt.subplots(layout="tight")
        meaningful.sort_values("hi", ascending=False).plot(kind="bar", ax=ax)
        save_fig(fig, img_dir / f"{figura}-caida_r2.svg")

    # =====================================================================
    # helices_0: boxplot R² zoomed (kernel classifiers only)
    # =====================================================================
    fig, ax = plt.subplots(layout="tight")
    data_box = bi[
        bi.dataset.eq("helices_0") & bi.clf.isin(["fkdc", "kdc", "kn", "fkn"])
    ].sort_values("clf")
    sns.boxplot(
        data_box,
        hue="clf",
        y="r2",
        gap=0.2,
        ax=ax,
        palette=default_palette,
        saturation=1.0,
    )
    apply_hatching(ax)
    ax.axhline(
        data_box.groupby("clf")["r2"].median().max(), linestyle="dotted", color="gray"
    )
    ybot = np.percentile(data_box["r2"].dropna(), 10)
    ax.set_ylim(ybot, None)
    save_fig(fig, img_dir / "helices_0-boxplot-r2-zoomed.svg")

    # =====================================================================
    # eslabones_0: params for a specific seed
    # =====================================================================
    dataset_esl = "eslabones_0"
    semilla_esl = 2411
    best_estimators_esl = {
        k: v[k[2]]["busqueda"].best_estimator_
        for k, v in infos.items()
        if k[0] == dataset_esl and k[2].endswith("kdc")
    }
    best_params_esl = pd.DataFrame.from_records(
        [
            {
                "clf": k[2],
                "semilla": int(k[1 if dataset_esl in synth_datasets else 3]),
                "alpha": v.get_params().get("alpha", 1),
                "bandwidth": v.get_params()["bandwidth"],
            }
            for k, v in best_estimators_esl.items()
        ]
    ).set_index(["semilla", "clf"])
    scores_esl = (
        bi[bi.dataset.eq(dataset_esl) & bi.clf.str.endswith("kdc")]
        .set_index(["semilla", "clf"])
        .r2
    )
    best_params_esl["r2"] = scores_esl
    fpath_esl = data_dir / f"{dataset_esl}-params-{semilla_esl}.csv"
    best_params_esl.loc[pd.IndexSlice[semilla_esl, :]].round(4).to_csv(fpath_esl)
    logger.info(f"Wrote {fpath_esl}")

    # =====================================================================
    # Parametros comparados CSVs
    # =====================================================================
    parametros_comparados("helices_0", "kdc", infos=infos, bi=bi)
    parametros_comparados("hueveras_0", "kdc", infos=infos, bi=bi)
    parametros_comparados("hueveras_0", "kn", infos=infos, bi=bi)

    # =====================================================================
    # Seminario-modesto: score vs n_neighbors
    # =====================================================================
    for dataset in ["helices_0", "eslabones_0"]:
        fig, ax = plt.subplots(figsize=(10, 5), layout="tight")
        plot_fkn_kn_score_vs_n_neighbors(dataset, 2411, ax, infos=infos)
        save_fig(fig, img_dir / f"{dataset}-fkn_kn-mean_test_score.svg")

    logger.info("Done.")
