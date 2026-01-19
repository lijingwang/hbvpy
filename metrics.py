# metrics.py
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def _to_numpy(x) -> np.ndarray:
    """Convert array like to 1D float array."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr


def mse(obs, sim) -> float:
    """Mean squared error."""
    o = _to_numpy(obs)
    s = _to_numpy(sim)
    return float(np.mean((o - s) ** 2))


def rmse(obs, sim) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mse(obs, sim)))


def mae(obs, sim) -> float:
    """Mean absolute error."""
    o = _to_numpy(obs)
    s = _to_numpy(sim)
    return float(np.mean(np.abs(o - s)))


def nse(obs, sim) -> float:
    """
    Nash Sutcliffe Efficiency.
    NSE = 1 - sum((obs - sim)^2) / sum((obs - mean(obs))^2)
    """
    o = _to_numpy(obs)
    s = _to_numpy(sim)
    denom = np.sum((o - np.mean(o)) ** 2)
    if denom == 0.0:
        return np.nan
    return float(1.0 - np.sum((o - s) ** 2) / denom)


def kge(obs, sim, version: str = "2009") -> float:
    """
    Kling Gupta Efficiency.

    KGE (Gupta et al., 2009):
      KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
      r     = corr(sim, obs)
      alpha = std(sim) / std(obs)
      beta  = mean(sim) / mean(obs)

    """
    o = _to_numpy(obs)
    s = _to_numpy(sim)

    if o.size < 2:
        return np.nan

    o_mean = np.mean(o)
    s_mean = np.mean(s)
    o_std = np.std(o, ddof=1)
    s_std = np.std(s, ddof=1)

    if o_std == 0.0:
        return np.nan

    r = np.corrcoef(s, o)[0, 1]
    beta = s_mean / o_mean if o_mean != 0.0 else np.nan

    alpha = s_std / o_std
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def align_on_time(
    modeled: pd.DataFrame,
    observed: pd.DataFrame,
    modeled_time_col: str = "Time",
    modeled_value_col: str = "Q_m3s",
    observed_time_col: str = "Time",
    observed_value_col: str = "Q (m3/s)",
    how: str = "inner",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Align modeled and observed discharge by timestamp.

    Returns
    - obs_values (np.ndarray)
    - sim_values (np.ndarray)
    - merged DataFrame with columns: Time, obs, sim
    """
    m = modeled[[modeled_time_col, modeled_value_col]].copy()
    o = observed[[observed_time_col, observed_value_col]].copy()

    m[modeled_time_col] = pd.to_datetime(m[modeled_time_col])
    o[observed_time_col] = pd.to_datetime(o[observed_time_col])

    m = m.rename(columns={modeled_time_col: "Time", modeled_value_col: "sim"})
    o = o.rename(columns={observed_time_col: "Time", observed_value_col: "obs"})

    merged = pd.merge(o, m, on="Time", how=how).sort_values("Time").reset_index(drop=True)

    # Drop NaNs pairwise
    merged = merged.dropna(subset=["obs", "sim"]).reset_index(drop=True)

    obs_vals = merged["obs"].to_numpy(dtype=float)
    sim_vals = merged["sim"].to_numpy(dtype=float)
    return obs_vals, sim_vals, merged


def discharge_metrics(
    modeled: pd.DataFrame,
    observed: pd.DataFrame,
    modeled_value_col: str = "Q_m3s",
    observed_value_col: str = "Q (m3/s)",
    modeled_time_col: str = "Time",
    observed_time_col: str = "Time",
    kge_version: str = "2009",
) -> Dict[str, float]:
    """
    Compute MSE, RMSE, MAE, NSE, KGE after aligning on Time.
    """
    obs_vals, sim_vals, _ = align_on_time(
        modeled=modeled,
        observed=observed,
        modeled_time_col=modeled_time_col,
        modeled_value_col=modeled_value_col,
        observed_time_col=observed_time_col,
        observed_value_col=observed_value_col,
    )

    return {
        "MSE": mse(obs_vals, sim_vals),
        "RMSE": rmse(obs_vals, sim_vals),
        "MAE": mae(obs_vals, sim_vals),
        "NSE": nse(obs_vals, sim_vals),
        "KGE": kge(obs_vals, sim_vals, version=kge_version),
    }
