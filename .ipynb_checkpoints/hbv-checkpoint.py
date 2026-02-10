import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple

# Author: Lijing Wang (lijing.wang@uconn.edu)

#    Modified from HRL (2026). HBV-EDU Hydrologic Model 
#    (https://www.mathworks.com/matlabcentral/fileexchange/41395-hbv-edu-hydrologic-model), MATLAB Central File Exchange. Retrieved January 19, 2026. 
#    and from https://github.com/johnrobertcraven/hbv_hydromodel.git

#    Citation:
#    AghaKouchak A., Habib E., 2010, Application of a Conceptual Hydrologic
#    Model in Teaching Hydrologic Processes, International Journal of Engineering Education, 26(4), 963-973. 

def hbv_run(
    forcing: pd.DataFrame,
    pet_monthly: pd.DataFrame,
    params: np.ndarray,
    area_km2: float = 410.0,
    Tsnow_thresh: float = 0.0,
    init_state: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    HBV type model run with snow, soil moisture, and two reservoirs.

    Inputs
    forcing : DataFrame
        Required columns:
          - Time (datetime)
          - Precipitation (mm/day)
          - Temperature (degC)

    pet_monthly : DataFrame
        Required columns:
          - month (1..12) or index convertible to 1..12
          - T_avg_month: Average temperature for each month (degC)
          - PEm_day: Reference PET per day for each month (mm/day)
        Optional:
          - PEm_month: PET monthly (mm/month)

    params : array like, length 10
        Model parameter vector:
        [d, fc, beta, cpar, k0, lthr, k1, k2, kp, pwp], where
        d    : degree day melt factor (mm/degC/day)
        fc   : field capacity (mm)
        beta : runoff nonlinearity exponent
        cpar : PET temperature correction factor
        k0   : quickflow coefficient for s1 above threshold lthr (1/day)
        lthr : threshold in upper storage for quickflow activation (mm)
        k1   : linear outflow coefficient from s1 (1/day)
        k2   : linear outflow coefficient from s2 (1/day)
        kp   : percolation coefficient from s1 to s2 (1/day)
        pwp  : permanent wilting point (mm)

    area_km2 : float
        Catchment area (km^2) used to convert runoff depth (mm/day) to discharge (m^3/s).

    Tsnow_thresh : float
        Temperature threshold (degC) for snow vs rain.

    init_state : dict or None
        Optional initial conditions at Time[0]:
          - snow (mm), soil (mm), s1 (mm), s2 (mm)
        If None, defaults to zeros.

    Returns
    results_df : DataFrame
        Time series of all states and fluxes:
          snow, liq_water, pe, ea, soil, dq, s1, s2, q_mmday, Q_m3s

    aux : dict
        Parameter vector, month indices, PET lookup arrays, and constants used.
    """
    required_cols = {"Time", "Precipitation", "Temperature"}
    missing = required_cols - set(forcing.columns)
    if missing:
        raise ValueError(f"forcing is missing required columns: {sorted(missing)}")

    forcing = forcing.copy()
    forcing["Time"] = pd.to_datetime(forcing["Time"])
    forcing = forcing.sort_values("Time").reset_index(drop=True)

    n_days = len(forcing)
    if n_days < 2:
        raise ValueError("forcing must contain at least 2 time steps")

    # Python uses 0 based indexing. Months from datetime are 1..12, so we convert to 0..11.
    month_0 = forcing["Time"].dt.month.to_numpy(dtype=int) - 1  # 0..11

    # PET lookup arrays. The first line in the monthly PET file should correspond to January.
    if "month" in pet_monthly.columns:
        pet = pet_monthly.copy()
        pet["month"] = pet["month"].astype(int)
        pet = pet.set_index("month")
    else:
        pet = pet_monthly.copy()
        pet.index = pet.index.astype(int)

    try:
        T_avg = np.array([pet.loc[m, "T_avg_month"] for m in range(1, 13)], dtype=float)  # Jan..Dec
        PEm_day = np.array([pet.loc[m, "PEm_day"] for m in range(1, 13)], dtype=float)    # Jan..Dec
    except KeyError as e:
        raise ValueError("pet_monthly must include months 1..12 and columns T_avg_month and PEm_day") from e

    # Forcings
    prec = forcing["Precipitation"].to_numpy(dtype=float)  # mm/day
    temp = forcing["Temperature"].to_numpy(dtype=float)    # degC

    # ----------
    # State (mm) and flux (mm/day) variables
    # ----------
    snow = np.zeros(n_days, dtype=float)       # snow water equivalent storage (mm)
    liq_water = np.zeros(n_days, dtype=float)  # liquid water, rain + melt available to soil/runoff (mm/day)
    pe = np.zeros(n_days, dtype=float)         # potential evapotranspiration (mm/day)
    ea = np.zeros(n_days, dtype=float)         # actual evapotranspiration (mm/day)
    soil = np.zeros(n_days, dtype=float)       # soil moisture storage (mm)
    dq = np.zeros(n_days, dtype=float)         # effective precipitation to reservoirs (mm/day)
    s1 = np.zeros(n_days, dtype=float)         # upper reservoir storage (mm)
    s2 = np.zeros(n_days, dtype=float)         # lower reservoir storage (mm)
    q_mmday = np.zeros(n_days, dtype=float)    # runoff depth equivalent (mm/day)
    Q_m3s = np.zeros(n_days, dtype=float)      # discharge (m^3/s)

    # Initial conditions (Time[0])
    init = init_state or {}
    snow[0] = float(init.get("snow", 0.0))
    soil[0] = float(init.get("soil", 0.0))
    s1[0] = float(init.get("s1", 0.0))
    s2[0] = float(init.get("s2", 0.0))

    # Parameters
    d, fc, beta, cpar, k0, lthr, k1, k2, kp, pwp = [float(x) for x in params]

    for t in range(1, n_days):
        m = month_0[t]  # 0..11

        # Temperature corrected PET using monthly climatology
        pe[t] = (1.0 + cpar * (temp[t] - T_avg[m])) * PEm_day[m]

        # Snow and rain partition and degree day melt
        if temp[t] < Tsnow_thresh:
            snow[t] = snow[t - 1] + prec[t]
            liq_water[t] = 0.0
        else:
            melt = d * (temp[t] - Tsnow_thresh)
            snow[t] = max(snow[t - 1] - melt, 0.0)
            liq_water[t] = prec[t] + min(snow[t - 1], melt)

        # Actual ET limited by soil moisture relative to pwp
        if soil[t - 1] > pwp:
            ea[t] = pe[t]
        else:
            ea[t] = pe[t] * (soil[t - 1] / pwp) if pwp > 0 else 0.0

        # Effective precipitation depends on relative soil wetness
        rel = soil[t - 1] / fc if fc > 0 else 0.0
        dq[t] = liq_water[t] * (rel ** beta)

        # Soil water balance
        soil[t] = soil[t - 1] + liq_water[t] - dq[t] - ea[t]

        # Upper reservoir storage s1 [mm]
        s1[t] = (
            s1[t - 1]                          # storage carried over from previous day
            + dq[t]                            # inflow from soil as effective precipitation
            - max(0.0, s1[t - 1] - lthr) * k0  # fast runoff (quickflow) when storage exceeds threshold lthr
            - s1[t - 1] * k1                   # linear outflow (interflow) from upper reservoir
            - s1[t - 1] * kp                   # percolation loss from upper to lower reservoir
        )

        # Lower reservoir storage s2 [mm]
        s2[t] = (
            s2[t - 1]                          # previous groundwater storage
            + s1[t - 1] * kp                   # recharge from upper reservoir via percolation
            - s2[t - 1] * k2                   # linear baseflow discharge
        )

        # Total runoff generation q [mm/day]
        q_mmday[t] = (
            max(0.0, s1[t - 1] - lthr) * k0    # fast runoff component from upper reservoir
            + s1[t] * k1                       # slower interflow from upper reservoir
            + s2[t] * k2                       # baseflow from lower reservoir
        )
        
        # Convert runoff depth (mm/day) to discharge (m^3/s) using catchment area (km^2)
        Q_m3s[t] = (q_mmday[t] * area_km2 * 1000.0) / 86400.0

    results_df = pd.DataFrame(
        {
            "Time": forcing["Time"],
            "snow": snow,
            "liq_water": liq_water,
            "pe": pe,
            "ea": ea,
            "soil": soil,
            "dq": dq,
            "s1": s1,
            "s2": s2,
            "q_mmday": q_mmday,
            "Q_m3s": Q_m3s,
        }
    )

    aux = {
        "params": np.array(params, dtype=float),
        "month_index_0": month_0,
        "area_km2": float(area_km2),
        "Tsnow_thresh": float(Tsnow_thresh),
        "pet_T_avg_month_0": T_avg,
        "pet_PEm_day_0": PEm_day,
    }

    return results_df, aux
