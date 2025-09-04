import cudf
import numpy as np
from typing import Dict, List, Optional


def _normalize_holidays(holidays: Optional[List[str]]) -> Optional[cudf.Series]:
    if not holidays:
        return None
    try:
        s = cudf.to_datetime(cudf.Series(holidays), format="%Y-%m-%d", errors="coerce")
        # normalize to midnight
        return s.dt.normalize()
    except Exception:
        return None


def build_driver_masks(
    ts: cudf.Series,
    sessions_cfg: Dict,
    driver_keys: List[str],
) -> Dict[str, cudf.Series]:
    """Build boolean open-session masks per driver for UTC timestamps.

    sessions_cfg structure (per driver_key):
      - weekmask: list of weekdays (0=Mon..6=Sun)
      - open_windows: dict or list mapping weekday->list of [start_minute,end_minute)
      - holidays: optional list[str YYYY-MM-DD]
    """
    out: Dict[str, cudf.Series] = {}
    if ts is None or sessions_cfg is None:
        return out
    try:
        wday = ts.dt.weekday
        minute = ts.dt.hour * 60 + ts.dt.minute
        ts_norm = ts.dt.normalize()
    except Exception:
        return out

    for key in driver_keys:
        cfg = sessions_cfg.get(key, {}) if isinstance(sessions_cfg, dict) else {}
        weekmask = set(cfg.get('weekmask', [0, 1, 2, 3, 4]))
        open_windows = cfg.get('open_windows', {}) or {}
        # open_windows can be dict of weekday->list or a list indexed by weekday
        def _windows_for_d(d: int) -> List[List[int]]:
            if isinstance(open_windows, dict):
                return open_windows.get(str(d), []) or open_windows.get(d, []) or []
            if isinstance(open_windows, list) and 0 <= d < len(open_windows):
                return open_windows[d] or []
            return []

        # Start with weekday mask
        try:
            m = wday.isin(cudf.Series(list(weekmask)))
        except Exception:
            m = cudf.Series(np.ones(len(ts), dtype=bool))

        # Build time window mask by OR over configured intervals for each weekday
        # Compute a mask that is True if minute in any window for its weekday
        try:
            # Initialize all False, then fill per weekday
            time_ok = cudf.Series(np.zeros(len(ts), dtype=bool))
            for d in range(7):
                wins = _windows_for_d(d)
                if not wins:
                    continue
                sel_d = (wday == d)
                if sel_d.any():
                    min_d = minute.where(sel_d, -1)
                    ok_d = cudf.Series(np.zeros(len(ts), dtype=bool))
                    for w in wins:
                        try:
                            start, end = int(w[0]), int(w[1])
                        except Exception:
                            continue
                        ok_w = (min_d >= start) & (min_d < end)
                        ok_d = ok_d | ok_w
                    time_ok = time_ok | ok_d
        except Exception:
            time_ok = cudf.Series(np.ones(len(ts), dtype=bool))

        mask = m & time_ok

        # Apply holidays if present
        holidays = _normalize_holidays(cfg.get('holidays')) if isinstance(cfg, dict) else None
        if holidays is not None and len(holidays) > 0:
            try:
                mask = mask & (~ts_norm.isin(holidays))
            except Exception:
                pass

        out[key] = mask

    return out


def driver_for_feature(name: str, feature_prefix_map: Dict[str, str]) -> Optional[str]:
    """Infer driver key from a feature name using substring mapping.

    feature_prefix_map: dict where key is substring to search, value is driver key.
    Returns the first driver whose substring appears in the feature name (case-insensitive),
    preferring longer substrings first.
    """
    if not feature_prefix_map:
        return None
    nm = name.lower()
    # sort substrings by length desc to match most specific first
    items = sorted(feature_prefix_map.items(), key=lambda kv: len(str(kv[0])), reverse=True)
    for sub, drv in items:
        try:
            if str(sub).lower() in nm:
                return str(drv)
        except Exception:
            continue
    return None


def build_feature_masks_data_driven(
    gdf,
    feats: List[str],
    window_rows: int,
    min_valid: int,
) -> Dict[str, cudf.Series]:
    """Build per-feature open-session masks based on data coverage (NaN gaps).

    For each feature f, marks True where rolling count of non-NaN values over
    a window of size `window_rows` is >= `min_valid`. This approximates session
    openness without a fixed schedule and captures early closes/weekends.
    """
    out: Dict[str, cudf.Series] = {}
    try:
        w = max(1, int(window_rows))
        mv = max(1, int(min_valid))
    except Exception:
        w, mv = 60, 45
    for f in feats:
        try:
            s = gdf[f]
            cnt = s.rolling(window=w, min_periods=1).count()
            out[f] = (cnt >= mv)
        except Exception:
            # If rolling unsupported on this column, default to all True
            out[f] = cudf.Series(np.ones(len(gdf), dtype=bool))
    return out
