import os
import pandas as pd
import numpy as np

#%%Helpers
def _safe_scalar(x):
    """Convert numpy/python scalar to CSV-safe value."""
    if x is None:
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (int, float, str, bool)):
        return x
    try:
        return float(x)
    except Exception:
        return str(x)


def individual_to_row(ind, metrics_names=None, key=None):
    """Convert a DEAP individual into a flat CSV-safe row."""
    metrics_names = metrics_names or []

    row = {
        "key": key,
        "individual": str(ind),
        "fitness": (
            _safe_scalar(ind.fitness.values[0])
            if hasattr(ind, "fitness") and ind.fitness.valid
            else None
        ),
        "params": _safe_scalar(getattr(ind, "params", None)),
    }

    for metric_name in metrics_names:
        row[metric_name] = _safe_scalar(getattr(ind, metric_name, None))

    return row


def row_to_individual(row, creator, pset, stgp_module, metrics_names=None):
    """
    Recover a DEAP individual from a saved CSV row.

    Requires:
    - same pset definition
    - same primitive/terminal names
    - same creator.Individual
    - same stgp module used originally
    """
    metrics_names = metrics_names or []

    expr_str = row["individual"]

    tree = stgp_module.PrimitiveTree.from_string(expr_str, pset)
    ind = creator.Individual(tree)

    if pd.notna(row.get("fitness", None)):
        ind.fitness.values = (float(row["fitness"]),)

    if pd.notna(row.get("params", None)):
        ind.params = int(float(row["params"]))

    for metric_name in metrics_names:
        if metric_name in row and pd.notna(row[metric_name]):
            setattr(ind, metric_name, float(row[metric_name]))

    return ind

#%%Saving functions
def save_population_csv(population, filepath, metrics_names=None):
    rows = [
        individual_to_row(ind, metrics_names=metrics_names)
        for ind in population
    ]

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    return filepath


def save_cache_csv(cache, filepath, metrics_names=None):
    rows = [
        individual_to_row(ind, metrics_names=metrics_names, key=key)
        for key, ind in cache.items()
    ]

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    return filepath


def save_best_csv(best, filepath, metrics_names=None):
    row = individual_to_row(best, metrics_names=metrics_names)

    df = pd.DataFrame([row])
    df.to_csv(filepath, index=False)
    return filepath

#%% Recovering functions
def load_population_csv(filepath, creator, pset, stgp_module, metrics_names=None):
    df = pd.read_csv(filepath)

    population = [
        row_to_individual(row, creator, pset, stgp_module, metrics_names)
        for _, row in df.iterrows()
    ]

    return population


def load_cache_csv(filepath, creator, pset, stgp_module, metrics_names=None):
    df = pd.read_csv(filepath)

    cache = {}

    for _, row in df.iterrows():
        ind = row_to_individual(row, creator, pset, stgp_module, metrics_names)
        key = row["key"]

        if pd.isna(key):
            key = str(ind)

        cache[key] = ind

    return cache


def load_best_csv(filepath, creator, pset, stgp_module, metrics_names=None):
    df = pd.read_csv(filepath)

    if len(df) == 0:
        return None

    return row_to_individual(
        df.iloc[0],
        creator,
        pset,
        stgp_module,
        metrics_names
    )