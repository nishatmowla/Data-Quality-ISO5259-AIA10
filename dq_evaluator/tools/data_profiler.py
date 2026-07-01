"""Statistical profiling of a dataset to feed into evaluation agents."""

import json
from pathlib import Path

import pandas as pd
import numpy as np


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix == ".parquet":
        return pd.read_parquet(path)
    elif suffix == ".json" or suffix == ".jsonl":
        return pd.read_json(path, lines=(suffix == ".jsonl"))
    elif suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def profile_dataset(df: pd.DataFrame, sample_rows: int = 5) -> dict:
    """Return a serialisable profile summary of the dataset."""
    n_rows, n_cols = df.shape
    profile: dict = {
        "shape": {"rows": n_rows, "columns": n_cols},
        "columns": {},
        "sample": df.head(sample_rows).to_dict(orient="records"),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_rate": float(df.duplicated().mean()),
    }

    for col in df.columns:
        series = df[col]
        col_info: dict = {
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "null_rate": float(series.isna().mean()),
            "unique_count": int(series.nunique()),
        }

        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            col_info.update({
                "min": _safe_float(desc.get("min")),
                "max": _safe_float(desc.get("max")),
                "mean": _safe_float(desc.get("mean")),
                "std": _safe_float(desc.get("std")),
                "q25": _safe_float(desc.get("25%")),
                "median": _safe_float(desc.get("50%")),
                "q75": _safe_float(desc.get("75%")),
                "outlier_count": _count_outliers(series),
            })
        else:
            top_values = series.value_counts().head(10).to_dict()
            col_info["top_values"] = {str(k): int(v) for k, v in top_values.items()}

        profile["columns"][col] = col_info

    # Label / target column detection heuristic
    label_candidates = [c for c in df.columns if c.lower() in ("label", "target", "class", "attack", "y")]
    if label_candidates:
        lc = label_candidates[0]
        vc = df[lc].value_counts()
        profile["label_column"] = lc
        profile["label_distribution"] = {str(k): int(v) for k, v in vc.items()}
        profile["class_balance"] = float(vc.min() / vc.max()) if vc.max() > 0 else None

    return profile


def profile_to_text(profile: dict) -> str:
    """Convert profile dict to a compact text summary for LLM context."""
    lines = [
        f"Dataset: {profile['shape']['rows']} rows x {profile['shape']['columns']} columns",
        f"Duplicate rows: {profile['duplicate_rows']} ({profile['duplicate_rate']:.1%})",
    ]
    if "label_column" in profile:
        lines.append(f"Label column: '{profile['label_column']}'")
        lines.append(f"Label distribution: {profile['label_distribution']}")
        if profile.get("class_balance") is not None:
            lines.append(f"Class balance (min/max): {profile['class_balance']:.3f}")

    lines.append("\nColumn profiles:")
    for col, info in profile["columns"].items():
        null_pct = f"{info['null_rate']:.1%} null"
        dtype = info["dtype"]
        unique = info["unique_count"]
        if "mean" in info:
            lines.append(f"  {col} [{dtype}]: {null_pct}, range [{info['min']}, {info['max']}], "
                         f"mean={info['mean']:.3g}, outliers={info['outlier_count']}, unique={unique}")
        else:
            top = list(info.get("top_values", {}).keys())[:5]
            lines.append(f"  {col} [{dtype}]: {null_pct}, unique={unique}, top={top}")

    lines.append(f"\nSample rows (first {len(profile['sample'])}):")
    for row in profile["sample"]:
        lines.append(f"  {row}")

    return "\n".join(lines)


def _safe_float(val) -> float | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return float(val)


def _count_outliers(series: pd.Series) -> int:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
    return int(mask.sum())
