"""Data audit for Rawaj Pay transfer extracts.
You write this file. Keep the signature exactly as it is: the grader calls it by name.
Allowed dependencies: numpy, pandas, scikit-learn (and the standard library).
"""
from __future__ import annotations
import pandas as pd
from sklearn.metrics import roc_auc_score

# check missing values
def is_missing(value) -> bool:
    """Return True for (NaN, None, empty strings, and whitespace-only strings)."""
    return pd.isna(value) or (isinstance(value, str) and value.strip() == "")

# check leakage
def check_leak(series, y) -> bool:
    """Check whether a series strongly predicts the target."""
    if len(y) < 20 or y.nunique() != 2:
        return False

    try:
        auc = roc_auc_score(y, series)
        return auc >= 0.98 or auc <= 0.02
    except ValueError:
        return False

# find may leak cols
def find_leakage_suspects(
    df, target, id_col, time_col, labelled_mask, missing_masks) -> list:
    """Find columns that may leak the target."""
    y = df.loc[labelled_mask, target]

    if len(y) < 20 or y.nunique() != 2:
        return []

    suspects = []
    for col in df.columns:
        if col in (target, id_col, time_col):
            continue

        series = df[col]
        missing_mask = missing_masks[col]

        # Treat all missing values as one distinct state.
        distinct_count = (
            series.loc[~missing_mask].nunique()
            + int(missing_mask.any())
        )

        if distinct_count <= 1:
            continue

        # Check missingness leakage.
        indicator = missing_mask.loc[labelled_mask].astype(int)

        if indicator.nunique() > 1 and check_leak(indicator, y):
            suspects.append(col)
            continue

        # Check numeric or Boolean value leakage.
        if (
            pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
        ):
            usable = labelled_mask & ~missing_mask

            if usable.sum() >= 20:
                x = series.loc[usable]
                y_sub = df.loc[usable, target]

                if x.nunique() > 1 and check_leak(x, y_sub):
                    suspects.append(col)

    return sorted(suspects)



def audit(df: pd.DataFrame, *, target: str, id_col: str, time_col: str) -> dict:
    """Audit a transfer extract before anyone trains a model on it.
    Raises KeyError if `target`, `id_col` or `time_col` is not a column of `df`.
    Returns a dict with exactly these keys:

    - "n_rows" (int): number of rows.
    - "missing" ({column: float}): fraction missing, for every column. Missing means
      NaN/None, or a string that is empty or only whitespace.
    - "duplicate_ids" (list): sorted non-missing `id_col` values that appear more than once.
    - "duplicate_rows" (int): rows that exactly repeat an earlier row.
    - "constant_columns" (list): sorted columns other than `target` with at most one
      distinct non-missing value.
    - "unlabelled_rows" (int): rows where `target` is missing.
    - "positive_rate" (float): share of labelled rows with target == 1; 0.0 if none is labelled.
    - "bad_timestamps" (int): non-missing values in `time_col` that do not parse as dates.
    - "time_range" (list): [min, max] of the parsed timestamps as ISO strings, or [None, None].
    - "leakage_suspects" (list): sorted columns (not target, id_col or time_col) that look
      like leaks. See README.md for the exact rule.
    """
    # 1/ validate req input cols
    for col in (target, id_col, time_col):
        if col not in df.columns:
            raise KeyError(col)
    
    n_rows = len(df)

    # Calculate missing values.
    missing_masks = {
        col: df[col].map(is_missing)
        for col in df.columns
    }

    missing = {
        col: float(mask.mean()) if n_rows else 0.0
        for col, mask in missing_masks.items()
    }

    # Find duplicate IDs.
    valid_ids = df.loc[~missing_masks[id_col], id_col]
    duplicates = valid_ids[valid_ids.duplicated(keep=False)]
    duplicate_ids = sorted(duplicates.unique().tolist())

    # Count duplicate rows.
    duplicate_rows = int(df.duplicated().sum())

    # Find constant columns, excluding target.
    constant_columns = []

    for col in df.columns:
        if col == target:
            continue

        values = df.loc[~missing_masks[col], col]

        if values.nunique() <= 1:
            constant_columns.append(col)

    constant_columns.sort()

    # Check labels.
    target_missing = missing_masks[target]
    labelled_mask = ~target_missing
    labelled = df.loc[labelled_mask, target]

    unlabelled_rows = int(target_missing.sum())
    positive_rate = (float((labelled == 1).mean()) if len(labelled) else 0.0)

    # Check timestamps.
    times = df.loc[~missing_masks[time_col], time_col]
    parsed_times = pd.to_datetime(times, errors="coerce")

    bad_timestamps = int(parsed_times.isna().sum())
    valid_times = parsed_times.dropna()

    time_range = (
        [valid_times.min().isoformat(), valid_times.max().isoformat()]
        if not valid_times.empty
        else [None, None]
    )

    # Find possible data leakage.
    leakage_suspects = find_leakage_suspects(
        df, target, id_col, time_col, labelled_mask, missing_masks
    )

    return {
        "n_rows": int(n_rows),
        "missing": missing,
        "duplicate_ids": duplicate_ids,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "unlabelled_rows": unlabelled_rows,
        "positive_rate": positive_rate,
        "bad_timestamps": bad_timestamps,
        "time_range": time_range,
        "leakage_suspects": leakage_suspects,
    }
