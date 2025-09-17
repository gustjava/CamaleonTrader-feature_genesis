"""Data loading helpers for backtesting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:  # Optional GPU dependencies
    import cudf  # type: ignore
except Exception:  # pragma: no cover - cudf is optional
    cudf = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas expected to exist
    pd = None  # type: ignore


def has_gpu_dataframe() -> bool:
    """Return True when cudf is available in the environment."""
    return cudf is not None


def load_parquet_dataset(path: Any) -> Any:
    """Load a parquet dataset using pandas by default."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Parquet dataset not found: {dataset_path}")

    if pd is None:  # pragma: no cover - defensive
        raise RuntimeError("pandas is required to load parquet files on CPU")

    try:
        return pd.read_parquet(dataset_path)
    except Exception as exc:  # pragma: no cover - surface explicit message
        raise RuntimeError(f"Failed to read parquet dataset: {dataset_path}") from exc


def ensure_cudf(df: Any):
    """Convert a pandas DataFrame to cudf when GPU mode is requested."""
    if cudf is None:
        raise RuntimeError("cudf is not available in this environment")
    if isinstance(df, cudf.DataFrame):  # type: ignore[attr-defined]
        return df
    if pd is not None and isinstance(df, pd.DataFrame):  # type: ignore[attr-defined]
        return cudf.from_pandas(df)
    raise TypeError(f"Unsupported dataframe type for cudf conversion: {type(df)!r}")


def to_pandas(df: Any):
    """Convert an arbitrary dataframe to pandas."""
    if pd is None:
        raise RuntimeError("pandas is required for CPU operations")
    if cudf is not None and isinstance(df, cudf.DataFrame):  # type: ignore[attr-defined]
        return df.to_pandas()
    if isinstance(df, pd.DataFrame):  # type: ignore[attr-defined]
        return df
    raise TypeError(f"Unsupported dataframe type for pandas conversion: {type(df)!r}")
