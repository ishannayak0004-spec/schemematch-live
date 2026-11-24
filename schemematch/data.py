"""Read de-identified survey features and audit missing records."""
import numpy as np
import pandas as pd

IHDS = ["annual_income_inr", "household_size", "asset_index"]
ACCESS = ["annual_income_inr", "household_size", "unbanked_adult", "no_lpg", "owns_farmland"]
ALIASES = {"income": IHDS[0], "npersons": IHDS[1], "assets": IHDS[2]}


def prepare(frame):
    frame = frame.rename(columns=lambda name: str(name).strip().lower())
    frame = frame.rename(columns=ALIASES)
    if frame.columns.duplicated().any():
        raise ValueError("Each feature must have exactly one column.")
    for features in (ACCESS, IHDS):
        if set(features).issubset(frame.columns):
            break
    else:
        raise ValueError("Use the IHDS or outreach columns listed in the data guide.")
    selected = frame.loc[:, features]
    values = selected.apply(pd.to_numeric, errors="coerce")
    if (selected.notna() & values.isna()).any().any():
        raise ValueError("Features must be numeric. Recode documented missing values before import.")
    data = values.dropna().reset_index(drop=True)
    if not np.isfinite(data.to_numpy()).all() or (data.abs() > 1e12).any().any():
        raise ValueError("Check the range of the numeric fields.")
    sizes = data.household_size
    if ((sizes < 1) | (sizes % 1 != 0)).any():
        raise ValueError("Household size must be a positive integer.")
    if features == ACCESS and not data[ACCESS[2:]].isin([0, 1]).all().all():
        raise ValueError("Outreach indicators accept only 0 and 1.")
    if features == IHDS and (data.asset_index < 0).any():
        raise ValueError("Asset index must be nonnegative.")
    return data, {"received": len(frame), "complete": len(data), "excluded": len(frame) - len(data)}
