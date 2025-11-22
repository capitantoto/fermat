import json
import logging
import pickle
import warnings
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.pyplot import close
from sklearn.base import BaseEstimator
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.utils import Bunch

from fkdc import config, root_dir
from fkdc.datasets import Dataset, found_datasets, synth_datasets
from fkdc.tarea import Tarea

# TODO: Retrain with consistent version to avoid issue
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

logger = logging.getLogger(__name__)
datasets = [*synth_datasets, *found_datasets]
clfs = list(config.clasificadores.keys())
infos = None
basic_info = None
default_palette = dict(zip(clfs, sns.color_palette("Set3")))


def load_infos(dir_or_paths: list[Path] | Path):
    if isinstance(dir_or_paths, Path):
        assert dir_or_paths.is_dir()
        paths = dir_or_paths.glob("*.pkl")
    elif isinstance(dir_or_paths, list):
        paths = dir_or_paths
    else:
        raise ValueError(
            "`dir_or_paths` debe ser un directorio con pickles de info o una lista de rutas a pickles de info"
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
    logger.debug(f"best: {best} (by {by}); p25: {first_quartile}")
    bad = summary[summary[by].lt(first_quartile)].index.tolist()
    return Bunch(
        dataset=dataset,
        summary=summary,
        best=best,
        bad=bad,
        # best_runs=best_runs,
    )


def boxplot(
    dataset: str,
    metric: str = "r2",
    info: pd.DataFrame | None = None,
    ax=None,
    palette: dict | None = None,
):
    palette = palette or default_palette
    info = _render_basic_info(info)

    if ax is None:
        ax = plt.gca()
    data = info[info.dataset.eq(dataset)].sort_values("clf").dropna(subset=metric)
    sns.boxplot(data, hue="clf", y=metric, gap=0.2, ax=ax, palette=palette)
    ax.set_title(dataset)
    ax.axhline(
        data.groupby("clf")[metric].median().max(), linestyle="dotted", color="gray"
    )


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
        fpath = datasets_dir / f"{dataset}-{seed}.pkl"
        plotting_key = (dataset, str(seed), clf, str(config.main_seed), scoring)
    elif dataset in found_datasets:
        fpath = datasets_dir / f"{dataset}.pkl"
        plotting_key = (dataset, str(None), clf, str(seed), scoring)
    else:
        raise ValueError(f"Dataset not known: {dataset}-{seed}")
    with open(data_dir / fpath, "rb") as fp:
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
    # plot_r2: bool=False  #TODO: implementar?
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
    # zmin, zmax = Z.min(), Z.max()
    CS = ax.contourf(X, Y, Z, 15, cmap=cmap)
    ax.set_title(f"{dataset} {seed} {clf}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    best_X = X[Z.argmax(axis=1)]
    ax.scatter(best_X, Y, marker="x", color="red")
    left_index = max(np.where(X == min(best_X))[0][0] - 1, 0)
    right_index = min(np.where(X == max(best_X))[0][0] + 1, len(X) - 1)
    logger.info([left_index, X[left_index], right_index, X[right_index]])
    ax.set_xlim(X[left_index], X[right_index])
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(CS)
    cbar.ax.set_ylabel("Score")
    return fig, ax


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
        handlers=[logging.StreamHandler()],
    )

    run_dir = config.run_dir
    logger.info(run_dir)
    infos = load_infos(run_dir)
    basic_info = parse_basic_info(infos, config.main_seed)
    datasets_dir = run_dir / "../datasets"
    img_dir = root_dir / "docs/img"
    data_dir = root_dir / "docs/data"
    for directory in [data_dir, img_dir, run_dir, datasets_dir]:
        directory.mkdir(exist_ok=True)
    run_seeds = config._get_run_seeds()
    logger.info(run_seeds)
    plotting_seed = run_seeds[0]
    # TODO: run anteojos como 25-sample? poder se puede, pero no es que no está estudiado...
    # datasets' scatterplots for fixed seed
    for dataset in datasets:
        if dataset in synth_datasets:
            fpath = datasets_dir / f"{dataset}-{plotting_seed}.pkl"
        elif dataset in found_datasets:
            fpath = datasets_dir / f"{dataset}.pkl"
        else:
            raise ValueError(f"Dataset not known: {dataset}, {plotting_seed}")
        with open(fpath, "rb") as fp:
            ds = pickle.load(fp)
        fig, ax = plt.subplots(layout="tight")
        ds.scatter(ax=ax)
        ax.set_title(dataset)
        fpath = img_dir / f"{dataset}-scatter.svg"
        logger.debug(fpath)
        fig.savefig(fpath)
        close(fig)
        plt.close()
    # "Highlights" by R^2
    highlights_by = "r2"
    for dataset in datasets:
        hl = get_highlights(dataset, by=highlights_by, info=basic_info)
        hl["summary"] = hl["summary"].round(4).to_csv()
        fname = f"{dataset}-{highlights_by}-highlights.json"
        fpath = data_dir / fname
        logger.debug(fpath)
        with open(fpath, "w") as fp:
            json.dump(hl, fp, indent=4)
    # Boxplots by R^2 + Accuracy
    for dataset, metric in product(datasets, ["r2", "accuracy"]):
        logger.debug(f"{dataset}-{metric}")
        fig, ax = plt.subplots(layout="tight")
        boxplot(dataset, metric, info=basic_info, ax=ax, palette=default_palette)
        ax.set_title(f"{dataset} -  {metric} por clf (boxplot)")
        fig.savefig(img_dir / f"{dataset}-{metric}-boxplot.svg")
        close(fig)
        plt.close()

    # decision boundaries for D2 datasets
    datasets_D2 = [
        "circulos_lo",
        "circulos_hi",
        "lunas_lo",
        "lunas_hi",
        "espirales_lo",
        "espirales_hi",
        "anteojos",
    ]
    for dataset, clf in product(datasets_D2, clfs):
        logger.debug([dataset, clf])
        fig, ax = plt.subplots(layout="tight")
        decision_boundary(dataset, plotting_seed, clf, ax=ax)
        fpath = img_dir / f"{dataset}-{clf}-decision_boundary.svg"
        logger.info(fpath)
        fig.savefig(fpath)
        close(fig)
        plt.close()
    # loss contours for (fkdc, fkn) in D2 datasets
    for dataset, seed in product(datasets_D2, run_seeds):
        contours = (("fkdc", "bandwidth", "alpha"), ("fkn", "n_neighbors", "alpha"))
        for clf, x, y in contours:
            logger.info([dataset, seed])
            clf, x, y = "fkdc", "bandwidth", "alpha"
            fig, ax, *_ = loss_contour(dataset, seed, clf, x, y)
            ax.set_xscale("log")
            fpath = img_dir / f"{dataset}-{seed}-{clf}-{x}-{y}-loss_contour.svg"
            logger.info(fpath)
            fig.savefig(fpath)
            close(fig)
            plt.close()
    # Custom Plots
