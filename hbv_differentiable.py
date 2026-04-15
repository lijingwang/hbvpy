import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Union

# Author: Lijing Wang (lijing.wang@uconn.edu)
#
# Differentiable PyTorch port of hbv.py
#
# Every HBV equation is rewritten as a PyTorch tensor operation so that
# autograd can compute gradients of any output (e.g. Q_m3s) with respect
# to any parameter.  This enables:
#   1. Gradient-based calibration for a single basin (see calibrate())
#   2. End-to-end dPL training where an upstream NN predicts parameters
#      and gradients flow through HBV back to the NN weights
#
# Key translation rules from the original numpy version:
#   if/else branches   →  is_cold mask  (multiply by 0 or 1)
#   max(0, x)          →  torch.clamp(x, min=0.0)
#   min(a, b)          →  torch.minimum(a, b)
#   max(a, b)          →  torch.maximum(a, b)
#
# Modified from HRL (2026). HBV-EDU Hydrologic Model
# (https://www.mathworks.com/matlabcentral/fileexchange/41395-hbv-edu-hydrologic-model)
#
# Citation:
# AghaKouchak A., Habib E., 2010, Application of a Conceptual Hydrologic
# Model in Teaching Hydrologic Processes, International Journal of
# Engineering Education, 26(4), 963-973.

PARAM_NAMES = ["d", "fc", "beta", "cpar", "k0", "lthr", "k1", "k2", "kp", "pwp"]


# ======================================================================
# Fast path: pre-compute forcing + torch.compile on the state loop
# ======================================================================

def prepare_forcing(
    forcing: pd.DataFrame,
    pet_monthly: pd.DataFrame,
    device: str = "cpu",
) -> dict:
    """
    Convert forcing DataFrame to tensors ONCE before training.

    Call this outside the epoch loop.  The returned dict is passed
    directly to hbv_run_fast() every epoch — no repeated pandas → tensor
    conversion.

    Returns
    -------
    dict with keys:
        prec, temp      : (n_days,) float32 tensors on device
        pe_lut          : (12,) monthly PET values
        T_avg_lut       : (12,) monthly mean temperature
        month_idx       : (n_days,) int64 month indices 0..11
        times           : DatetimeIndex (for results DataFrame)
    """
    forcing = forcing.copy()
    forcing["Time"] = pd.to_datetime(forcing["Time"])
    forcing = forcing.sort_values("Time").reset_index(drop=True)

    if "month" in pet_monthly.columns:
        pet = pet_monthly.copy().set_index("month")
    else:
        pet = pet_monthly.copy()
        pet.index = pet.index.astype(int)

    T_avg_np   = np.array([pet.loc[m, "T_avg_month"] for m in range(1, 13)], dtype=float)
    PEm_day_np = np.array([pet.loc[m, "PEm_day"]     for m in range(1, 13)], dtype=float)

    return {
        "prec"     : torch.tensor(forcing["Precipitation"].to_numpy(dtype=float),
                                  dtype=torch.float32, device=device),
        "temp"     : torch.tensor(forcing["Temperature"].to_numpy(dtype=float),
                                  dtype=torch.float32, device=device),
        "T_avg_lut": torch.tensor(T_avg_np,   dtype=torch.float32, device=device),
        "pe_lut"   : torch.tensor(PEm_day_np, dtype=torch.float32, device=device),
        "month_idx": torch.tensor(
                         forcing["Time"].dt.month.to_numpy(dtype=int) - 1,
                         dtype=torch.long, device=device),
        "times"    : forcing["Time"],
        "n_days"   : len(forcing),
    }


def _hbv_state_loop(
    prec: torch.Tensor,
    temp: torch.Tensor,
    pe_all: torch.Tensor,
    is_cold: torch.Tensor,
    melt_all: torch.Tensor,
    params: torch.Tensor,
    snow0: torch.Tensor,
    soil0: torch.Tensor,
    s1_0: torch.Tensor,
    s2_0: torch.Tensor,
    area_scale: float,
) -> torch.Tensor:
    """
    Pure-tensor HBV state loop.  No pandas, no list appends, no stack.
    Decorated with torch.compile to remove Python interpreter overhead.

    All recurrent states are tracked as scalar tensors — autograd builds
    the computation graph through each timestep exactly as before.
    """
    n = prec.shape[0]
    d, fc, beta, _, k0, lthr, k1, k2, kp, pwp = [params[i] for i in range(10)]

    snow_t = snow0
    soil_t = soil0
    s1_t   = s1_0
    s2_t   = s2_0

    Q_out = torch.zeros(n, dtype=prec.dtype, device=prec.device)

    for t in range(1, n):
        # Snow routine
        actual_melt = torch.minimum(snow_t, melt_all[t])
        snow_t = (
            is_cold[t] * (snow_t + prec[t])
            + (1.0 - is_cold[t]) * torch.clamp(snow_t - melt_all[t], min=0.0)
        )
        liq_t = (1.0 - is_cold[t]) * (prec[t] + actual_melt)

        # ET and soil
        ea_t   = pe_all[t] * torch.clamp(soil_t / (pwp + 1e-6), min=0.0, max=1.0)
        dq_t   = liq_t * torch.clamp(soil_t / (fc + 1e-6), min=0.0, max=1.0) ** beta
        soil_t = torch.clamp(soil_t + liq_t - dq_t - ea_t, min=0.0)

        # Reservoirs
        qf    = torch.clamp(s1_t - lthr, min=0.0) * k0
        s1_t  = torch.clamp(s1_t + dq_t - qf - s1_t * k1 - s1_t * kp, min=0.0)
        s2_t  = torch.clamp(s2_t + s1_t * kp - s2_t * k2, min=0.0)

        Q_out[t] = torch.clamp(qf + s1_t * k1 + s2_t * k2, min=0.0)

    return Q_out * area_scale


# Compile the loop — removes Python interpreter overhead from every iteration.
# First call pays a one-time compilation cost (~5s); all subsequent calls are fast.
# Falls back to eager mode gracefully if compile is unavailable.
try:
    _hbv_state_loop_compiled = torch.compile(_hbv_state_loop)
except Exception:
    _hbv_state_loop_compiled = _hbv_state_loop


def hbv_run_fast(
    forcing_tensors: dict,
    params: Union[torch.Tensor, np.ndarray, list],
    area_km2: float = 410.0,
    Tsnow_thresh: float = 0.0,
    init_state: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """
    Fast differentiable HBV forward pass.

    Requires forcing to be pre-converted with prepare_forcing() — call
    that once before training, then call hbv_run_fast() every epoch.

    Two speedups over hbv_run_differentiable():
      1. No pandas → tensor conversion (done once in prepare_forcing)
      2. Forcing-derived quantities (PET, snow mask) pre-computed as
         full tensors before the state loop — not recomputed each step
      3. torch.compile removes Python interpreter overhead from the loop

    Parameters
    ----------
    forcing_tensors : dict
        Output of prepare_forcing().
    params : tensor or array, shape (10,)
        [d, fc, beta, cpar, k0, lthr, k1, k2, kp, pwp]
        Pass with requires_grad=True for gradient-based use.
    area_km2 : float
    Tsnow_thresh : float
    init_state : dict or None

    Returns
    -------
    Q_m3s : torch.Tensor, shape (n_days,)
        Gradients flow through this tensor.
    """
    prec      = forcing_tensors["prec"]
    temp      = forcing_tensors["temp"]
    T_avg_lut = forcing_tensors["T_avg_lut"]
    pe_lut    = forcing_tensors["pe_lut"]
    month_idx = forcing_tensors["month_idx"]
    device    = prec.device

    if not isinstance(params, torch.Tensor):
        params = torch.tensor(params, dtype=torch.float32, device=device)
    else:
        params = params.to(device=device, dtype=torch.float32)

    cpar   = params[3]
    Tsnow  = torch.tensor(Tsnow_thresh, dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # Pre-compute ALL forcing-derived quantities as full tensors.
    # These have no state dependency → computed once, outside the loop.
    # ------------------------------------------------------------------
    pe_all   = torch.clamp(
                   (1.0 + cpar * (temp - T_avg_lut[month_idx])) * pe_lut[month_idx],
                   min=0.0)                                          # (n_days,)
    is_cold  = (temp < Tsnow).to(dtype=torch.float32)               # (n_days,) binary
    melt_all = torch.clamp(params[0] * (temp - Tsnow), min=0.0)     # (n_days,)

    init = init_state or {}
    snow0 = torch.tensor(float(init.get("snow", 0.0)), dtype=torch.float32, device=device)
    soil0 = torch.tensor(float(init.get("soil", 0.0)), dtype=torch.float32, device=device)
    s1_0  = torch.tensor(float(init.get("s1",   0.0)), dtype=torch.float32, device=device)
    s2_0  = torch.tensor(float(init.get("s2",   0.0)), dtype=torch.float32, device=device)

    area_scale = float(area_km2 * 1000.0 / 86400.0)

    return _hbv_state_loop_compiled(
        prec, temp, pe_all, is_cold, melt_all,
        params, snow0, soil0, s1_0, s2_0, area_scale,
    )

# Physically reasonable default bounds for constrained calibration
PARAM_BOUNDS = {
    #        lower   upper
    "d"    : (0.1,   10.0),
    "fc"   : (10.0,  600.0),
    "beta" : (0.5,   6.0),
    "cpar" : (0.0,   1.0),
    "k0"   : (0.01,  0.99),
    "lthr" : (0.0,   100.0),
    "k1"   : (0.01,  0.99),
    "k2"   : (0.001, 0.5),
    "kp"   : (0.001, 0.5),
    "pwp"  : (0.0,   200.0),
}


def hbv_run_differentiable(
    forcing: pd.DataFrame,
    pet_monthly: pd.DataFrame,
    params: Union[torch.Tensor, np.ndarray, list],
    area_km2: float = 410.0,
    Tsnow_thresh: float = 0.0,
    init_state: Optional[Dict[str, float]] = None,
    device: str = "cpu",
) -> Tuple[torch.Tensor, pd.DataFrame]:
    """
    Differentiable PyTorch HBV model.

    All internal computations use PyTorch tensors so that gradients flow
    from Q_m3s back through every HBV equation to the parameters.

    Parameters
    ----------
    forcing : DataFrame
        Required columns: Time, Precipitation (mm/day), Temperature (degC)
    pet_monthly : DataFrame
        Required columns: month (1..12), T_avg_month (degC), PEm_day (mm/day)
    params : torch.Tensor or array-like, shape (10,)
        [d, fc, beta, cpar, k0, lthr, k1, k2, kp, pwp]
        Pass as torch.Tensor(requires_grad=True) for gradient-based use.
    area_km2 : float
        Catchment area (km^2) for mm/day -> m^3/s conversion.
    Tsnow_thresh : float
        Temperature threshold (degC) for snow/rain split.
    init_state : dict or None
        Initial snow, soil, s1, s2 (mm). Defaults to zeros.
    device : str
        "cpu" or "cuda"

    Returns
    -------
    Q_m3s : torch.Tensor, shape (n_days,)
        Simulated discharge (m^3/s). Gradients flow through this tensor.
    results_df : pd.DataFrame
        Time series of all states and fluxes (detached numpy values,
        matches the column layout of hbv.py).
    """
    # ------------------------------------------------------------------
    # Prepare forcing
    # ------------------------------------------------------------------
    forcing = forcing.copy()
    forcing["Time"] = pd.to_datetime(forcing["Time"])
    forcing = forcing.sort_values("Time").reset_index(drop=True)
    n_days = len(forcing)

    if n_days < 2:
        raise ValueError("forcing must contain at least 2 time steps")

    month_0 = forcing["Time"].dt.month.to_numpy(dtype=int) - 1  # 0..11

    prec = torch.tensor(
        forcing["Precipitation"].to_numpy(dtype=float),
        dtype=torch.float32, device=device
    )
    temp = torch.tensor(
        forcing["Temperature"].to_numpy(dtype=float),
        dtype=torch.float32, device=device
    )

    # ------------------------------------------------------------------
    # PET monthly lookup  (non-trainable data tensors)
    # ------------------------------------------------------------------
    if "month" in pet_monthly.columns:
        pet = pet_monthly.copy()
        pet["month"] = pet["month"].astype(int)
        pet = pet.set_index("month")
    else:
        pet = pet_monthly.copy()
        pet.index = pet.index.astype(int)

    try:
        T_avg_np   = np.array([pet.loc[m, "T_avg_month"] for m in range(1, 13)], dtype=float)
        PEm_day_np = np.array([pet.loc[m, "PEm_day"]     for m in range(1, 13)], dtype=float)
    except KeyError as e:
        raise ValueError(
            "pet_monthly must include months 1..12 and columns T_avg_month and PEm_day"
        ) from e

    T_avg_lut   = torch.tensor(T_avg_np,   dtype=torch.float32, device=device)
    PEm_day_lut = torch.tensor(PEm_day_np, dtype=torch.float32, device=device)

    Tsnow = torch.tensor(Tsnow_thresh, dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # Parameters — accept tensor (with grad) or plain array
    # ------------------------------------------------------------------
    if not isinstance(params, torch.Tensor):
        params = torch.tensor(params, dtype=torch.float32, device=device)
    else:
        params = params.to(device=device, dtype=torch.float32)

    d    = params[0]   # degree day melt factor  (mm/degC/day)
    fc   = params[1]   # field capacity           (mm)
    beta = params[2]   # runoff nonlinearity
    cpar = params[3]   # PET temperature correction factor
    k0   = params[4]   # quickflow coefficient    (1/day)
    lthr = params[5]   # upper storage threshold  (mm)
    k1   = params[6]   # interflow coefficient    (1/day)
    k2   = params[7]   # baseflow coefficient     (1/day)
    kp   = params[8]   # percolation coefficient  (1/day)
    pwp  = params[9]   # permanent wilting point  (mm)

    # ------------------------------------------------------------------
    # Initial states
    # ------------------------------------------------------------------
    init = init_state or {}
    snow_t = torch.tensor(float(init.get("snow", 0.0)), dtype=torch.float32, device=device)
    soil_t = torch.tensor(float(init.get("soil", 0.0)), dtype=torch.float32, device=device)
    s1_t   = torch.tensor(float(init.get("s1",   0.0)), dtype=torch.float32, device=device)
    s2_t   = torch.tensor(float(init.get("s2",   0.0)), dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # Time loop  — every operation is differentiable
    # ------------------------------------------------------------------
    zero = torch.zeros((), dtype=torch.float32, device=device)

    snow_out    = [snow_t]
    liq_out     = [zero.clone()]
    pe_out      = [zero.clone()]
    ea_out      = [zero.clone()]
    soil_out    = [soil_t]
    dq_out      = [zero.clone()]
    s1_out      = [s1_t]
    s2_out      = [s2_t]
    q_mmday_out = [zero.clone()]

    for t in range(1, n_days):
        m = month_0[t]  # integer index 0..11, used only for lookup — not optimised

        # ---- PET (temperature corrected) --------------------------------
        # pe[t] = (1 + cpar*(T - T_avg_month)) * PEm_day_month
        pe_t = (1.0 + cpar * (temp[t] - T_avg_lut[m])) * PEm_day_lut[m]
        pe_t = torch.clamp(pe_t, min=0.0)          # PET >= 0

        # ---- Snow routine -----------------------------------------------
        # Original uses if/else on temperature.
        # Replaced by a binary mask (is_cold) so the graph stays connected.
        #
        #   is_cold = 1  when temp < Tsnow  →  snow accumulates, no melt
        #   is_cold = 0  when temp >= Tsnow →  melt occurs, liquid water produced
        is_cold = (temp[t] < Tsnow).float()                          # 0 or 1, no gradient needed here

        melt        = torch.clamp(d * (temp[t] - Tsnow), min=0.0)   # melt only when warm
        actual_melt = torch.minimum(snow_t, melt)                    # cannot melt more than snowpack

        snow_new = (
            is_cold       * (snow_t + prec[t])                       # cold: accumulate
            + (1.0 - is_cold) * torch.clamp(snow_t - melt, min=0.0) # warm: deplete
        )
        liq_t = (1.0 - is_cold) * (prec[t] + actual_melt)           # liquid water reaching soil

        # ---- Actual ET --------------------------------------------------
        # Original: full ET above pwp, linear reduction below
        # Equivalent: ET_factor = clamp(soil / pwp, 0, 1)
        # 1e-6 guard prevents division by zero when pwp -> 0
        et_factor = torch.clamp(soil_t / (pwp + 1e-6), min=0.0, max=1.0)
        ea_t      = pe_t * et_factor

        # ---- Effective precipitation (soil wetness nonlinearity) --------
        # dq = liq * (soil/fc)^beta
        rel  = torch.clamp(soil_t / (fc + 1e-6), min=0.0, max=1.0)
        dq_t = liq_t * (rel ** beta)

        # ---- Soil water balance ----------------------------------------
        soil_new = torch.clamp(soil_t + liq_t - dq_t - ea_t, min=0.0)

        # ---- Upper reservoir s1 ----------------------------------------
        # quickflow only when s1 exceeds threshold lthr
        quickflow = torch.clamp(s1_t - lthr, min=0.0) * k0
        s1_new    = torch.clamp(
            s1_t + dq_t - quickflow - s1_t * k1 - s1_t * kp,
            min=0.0
        )

        # ---- Lower reservoir s2 ----------------------------------------
        s2_new = torch.clamp(s2_t + s1_t * kp - s2_t * k2, min=0.0)

        # ---- Total runoff (mm/day) --------------------------------------
        # Matches original: quickflow uses s1[t-1], interflow uses s1[t], baseflow uses s2[t]
        q_t = torch.clamp(
            torch.clamp(s1_t - lthr, min=0.0) * k0   # quickflow  (s1 before update)
            + s1_new * k1                              # interflow  (s1 after update)
            + s2_new * k2,                             # baseflow   (s2 after update)
            min=0.0
        )

        # ---- Advance states --------------------------------------------
        snow_t = snow_new
        soil_t = soil_new
        s1_t   = s1_new
        s2_t   = s2_new

        snow_out.append(snow_t)
        liq_out.append(liq_t)
        pe_out.append(pe_t)
        ea_out.append(ea_t)
        soil_out.append(soil_t)
        dq_out.append(dq_t)
        s1_out.append(s1_t)
        s2_out.append(s2_t)
        q_mmday_out.append(q_t)

    # ------------------------------------------------------------------
    # Stack outputs
    # ------------------------------------------------------------------
    q_mmday_tensor = torch.stack(q_mmday_out)                        # (n_days,)
    Q_m3s = q_mmday_tensor * (area_km2 * 1000.0 / 86400.0)          # mm/day -> m^3/s

    # Build results DataFrame with detached values (same columns as hbv.py)
    def _np(lst: List[torch.Tensor]) -> np.ndarray:
        return torch.stack(lst).detach().cpu().numpy()

    results_df = pd.DataFrame({
        "Time"     : forcing["Time"].values,
        "snow"     : _np(snow_out),
        "liq_water": _np(liq_out),
        "pe"       : _np(pe_out),
        "ea"       : _np(ea_out),
        "soil"     : _np(soil_out),
        "dq"       : _np(dq_out),
        "s1"       : _np(s1_out),
        "s2"       : _np(s2_out),
        "q_mmday"  : _np(q_mmday_out),
        "Q_m3s"    : Q_m3s.detach().cpu().numpy(),
    })

    return Q_m3s, results_df


def calibrate(
    forcing: pd.DataFrame,
    pet_monthly: pd.DataFrame,
    Q_obs: Union[torch.Tensor, np.ndarray, pd.Series],
    params_init: Optional[Union[np.ndarray, list]] = None,
    area_km2: float = 410.0,
    Tsnow_thresh: float = 0.0,
    init_state: Optional[Dict[str, float]] = None,
    n_epochs: int = 500,
    lr: float = 0.01,
    constrain_params: bool = True,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[torch.Tensor, List[float]]:
    """
    Gradient-based single-basin calibration of differentiable HBV.

    Minimises MSE(Q_sim, Q_obs) using Adam.  NaN values in Q_obs are
    automatically masked out.

    Parameters
    ----------
    Q_obs : tensor or array, shape (n_days,)
        Observed discharge (m^3/s).
    params_init : array or None
        Initial [d, fc, beta, cpar, k0, lthr, k1, k2, kp, pwp].
        Defaults to physically reasonable mid-range values.
    n_epochs : int
        Number of gradient descent iterations.
    lr : float
        Adam learning rate.
    constrain_params : bool
        If True, clamp parameters to PARAM_BOUNDS after each step.
    verbose : bool
        Print NSE every 50 epochs.

    Returns
    -------
    params : torch.Tensor, shape (10,)
        Calibrated parameters (detached). Index matches PARAM_NAMES.
    loss_history : list of float
        MSE loss recorded after each epoch.

    Example
    -------
    >>> params, history = calibrate(forcing, pet_monthly, Q_obs, n_epochs=300)
    >>> print(dict(zip(PARAM_NAMES, params.numpy())))
    """
    if params_init is None:
        # Mid-range defaults
        params_init = [2.0, 150.0, 2.0, 0.1, 0.3, 10.0, 0.1, 0.05, 0.05, 50.0]

    params = torch.tensor(
        params_init, dtype=torch.float32, device=device, requires_grad=True
    )

    if not isinstance(Q_obs, torch.Tensor):
        Q_obs = torch.tensor(np.asarray(Q_obs, dtype=float), dtype=torch.float32, device=device)

    valid = ~torch.isnan(Q_obs)  # boolean mask for valid (non-NaN) observations

    # Pre-compute parameter bounds as tensors for clamping
    lb = torch.tensor([PARAM_BOUNDS[n][0] for n in PARAM_NAMES], dtype=torch.float32, device=device)
    ub = torch.tensor([PARAM_BOUNDS[n][1] for n in PARAM_NAMES], dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam([params], lr=lr)
    loss_history: List[float] = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        Q_sim, _ = hbv_run_differentiable(
            forcing=forcing,
            pet_monthly=pet_monthly,
            params=params,
            area_km2=area_km2,
            Tsnow_thresh=Tsnow_thresh,
            init_state=init_state,
            device=device,
        )

        # MSE on valid timesteps
        loss = torch.mean((Q_sim[valid] - Q_obs[valid]) ** 2)
        loss.backward()
        optimizer.step()

        # Optional: project parameters back into physical bounds
        if constrain_params:
            with torch.no_grad():
                params.clamp_(lb, ub)

        loss_val = loss.item()
        loss_history.append(loss_val)

        if verbose and (epoch + 1) % 50 == 0:
            with torch.no_grad():
                obs_np  = Q_obs[valid].cpu().numpy()
                sim_np  = Q_sim[valid].detach().cpu().numpy()
                denom   = np.sum((obs_np - obs_np.mean()) ** 2)
                nse_val = 1.0 - np.sum((obs_np - sim_np) ** 2) / denom if denom > 0 else float("nan")
            print(f"Epoch {epoch + 1:4d}/{n_epochs}  MSE={loss_val:.6f}  NSE={nse_val:.4f}")

    return params.detach(), loss_history
