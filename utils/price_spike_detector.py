# utils/price_spike_detector.py
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class SpikeDetectorConfig:
    time_col: str = "datetime"
    price_col: str = "eprice_15min"
    # rolling window for baseline (e.g. "1D", "2D", "48H")
    window: str = None
    # minimum number of points required in window
    min_window_points: int = None
    # Require larger deviation from median baseline (large deviations → catches only large anomalies):
    z_threshold: float = None
    # required relative jump vs previous point (e.g. 1.2 / 100 = +120%)
    pct_threshold: float = None
    # ignore spikes when absolute price is tiny (< 5 snt)
    abs_min_price: float = None
    


class PriceSpikeDetector:
    """
    Rolling, robust price spike detector for 15-min eprice series.

    - Uses rolling median + MAD to compute a robust z-score.
    - Also checks percentage jump vs previous point.
    - Works on any DataFrame with [time_col, price_col].
    """

    def __init__(self, config: SpikeDetectorConfig | None = None, logger=None):
        self.config = config or SpikeDetectorConfig()
        self._log = logger if logger is not None else print

    def detect_spikes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a copy of df with additional columns:

            'baseline_median'
            'baseline_mad'
            'robust_z'
            'pct_jump'
            'is_spike'  (bool)

        Assumes time_col is sortable datetime (tz-aware or naive).
        """
        c = self.config

        if df is None or df.empty:
            self._log("[SpikeDetector] Input DataFrame is empty.")
            return df.copy()

        if c.time_col not in df.columns or c.price_col not in df.columns:
            raise ValueError(
                f"[SpikeDetector] DataFrame must contain '{c.time_col}' and '{c.price_col}'."
            )

        # Work on a copy, sorted by time
        work = df[[c.time_col, c.price_col]].copy()
        work = work.sort_values(c.time_col).reset_index(drop=True)

        # Ensure datetime dtype
        if not pd.api.types.is_datetime64_any_dtype(work[c.time_col]):
            work[c.time_col] = pd.to_datetime(work[c.time_col], utc=True, errors="coerce")
        work = work.dropna(subset=[c.time_col])

        # Set as index for rolling window
        work = work.set_index(c.time_col)

        # Rolling median as baseline
        rolling = work[c.price_col].rolling(c.window, min_periods=c.min_window_points)

        baseline_median = rolling.median()

        # MAD: median(|x - median|)
        abs_dev = (work[c.price_col] - baseline_median).abs()
        baseline_mad = abs_dev.rolling(c.window, min_periods=c.min_window_points).median()

        # Scale MAD to be comparable to std-dev-like measure
        # (for normal distrib, sigma ≈ 1.4826 * MAD)
        mad_scaled = baseline_mad * 1.4826

        # Robust z-score
        with np.errstate(divide="ignore", invalid="ignore"):
            robust_z = (work[c.price_col] - baseline_median) / mad_scaled
            robust_z = robust_z.replace([np.inf, -np.inf], np.nan)

        # Relative jump vs previous point
        prev_price = work[c.price_col].shift(1)
        pct_jump = (work[c.price_col] - prev_price) / prev_price
        pct_jump = pct_jump.replace([np.inf, -np.inf], np.nan)

        # Spike condition
        price = work[c.price_col]

        # Base spike mask: "statistically odd OR big jump" AND above abs_min
        is_spike = (
            (price >= c.abs_min_price) &
            (
                (robust_z >= c.z_threshold) |
                (pct_jump >= c.pct_threshold)
            )
        )

        out = work.copy()
        out["baseline_median"] = baseline_median
        out["baseline_mad"] = baseline_mad
        out["robust_z"] = robust_z
        out["pct_jump"] = pct_jump
        out["is_spike"] = is_spike.fillna(False)

        # --- Severity classification: mild vs strong ---
        # "strong" = spike where z or pct_jump is significantly above the UI thresholds
        # (2× the threshold is used as a simple heuristic)
        n = len(out)
        severity = np.full(n, "none", dtype=object)

        # Make sure we have numpy arrays for comparisons; NaNs → False
        z_vals = out["robust_z"].to_numpy()
        pct_vals = out["pct_jump"].to_numpy()
        spike_vals = out["is_spike"].to_numpy()

        strong_mask = np.zeros(n, dtype=bool)

        if c.z_threshold is not None:
            strong_mask |= (z_vals >= (2.0 * float(c.z_threshold)))
        if c.pct_threshold is not None:
            strong_mask |= (pct_vals >= (2.0 * float(c.pct_threshold)))

        strong_mask &= spike_vals  # must also be a spike

        mild_mask = spike_vals & ~strong_mask

        severity[mild_mask] = "mild"
        severity[strong_mask] = "strong"
        out["severity"] = severity

        # restore index as column
        out = out.reset_index()

        n_total = int(out["is_spike"].sum())
        n_strong = int((out["severity"] == "strong").sum())
        n_mild = int((out["severity"] == "mild").sum())

        self._log(
            f"[SpikeDetector] Detected {n_total} spikes over {len(out)} points "
            f"(mild={n_mild}, strong={n_strong}, "
            f"window={c.window}, z_thr={c.z_threshold}, pct_thr={c.pct_threshold})."
        )

        return out