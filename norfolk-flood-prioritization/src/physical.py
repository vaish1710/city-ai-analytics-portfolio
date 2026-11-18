"""Physically based expected annual loss (EAL) when no parcel-level loss labels exist.

Method: for each parcel, estimate flood stage at three return periods from the FEMA base
flood elevation (BFE), convert stage to depth above the first floor, apply a depth-damage
curve, and integrate damage over annual exceedance probability.

PLACEHOLDERS to replace before real use (each is a documented assumption, not a finding):
  * stage offsets for the 10-yr and 500-yr events relative to the BFE
  * the depth-damage curve (swap in USACE / HAZUS curves for your building types)
  * first-floor height by foundation type
Better still: replace stage inputs with Norfolk's scenario depth grids if obtainable.
"""
import numpy as np
import pandas as pd

# depth above first floor (ft) -> fraction of building value damaged (illustrative shape)
DEPTH = np.array([-2.0, -1.0, 0.0, 1.0, 3.0, 6.0, 10.0])
DAMAGE = np.array([0.00, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65])
FIRST_FLOOR_HEIGHT = {"slab": 0.7, "crawlspace": 2.0, "basement": 0.0}
DEFAULT_FFH = 1.5
STAGE_OFFSETS = {"p10": -2.0, "p100": 0.0, "p500": 1.5}  # relative to BFE, ft
P = {"p10": 0.10, "p100": 0.01, "p500": 0.002}


def _damage(stage, ffe):
    return np.interp(stage - ffe, DEPTH, DAMAGE, left=0.0, right=DAMAGE[-1])


def estimate_eal(df: pd.DataFrame) -> pd.Series:
    ffh = df["foundation"].map(FIRST_FLOOR_HEIGHT).fillna(DEFAULT_FFH) if "foundation" in df else DEFAULT_FFH
    ffe = df["elevation_ft"] + ffh
    bfe = df["bfe_ft"] if "bfe_ft" in df else pd.Series(np.nan, index=df.index)
    zone = df["fema_zone"].astype(str)
    # Parcels in AE/VE/AO/A zones without a published BFE: assume BFE = ground + 1 ft (flagged by caller).
    sfha = zone.str.upper().str.match(r"^(A|V)")
    bfe = bfe.where(bfe.notna(), np.where(sfha, df["elevation_ft"] + 1.0, np.nan))
    d10 = _damage(bfe + STAGE_OFFSETS["p10"], ffe)
    d100 = _damage(bfe + STAGE_OFFSETS["p100"], ffe)
    d500 = _damage(bfe + STAGE_OFFSETS["p500"], ffe)
    frac = (0.5 * d10 * P["p10"]                              # frequent-event tail
            + 0.5 * (d10 + d100) * (P["p10"] - P["p100"])
            + 0.5 * (d100 + d500) * (P["p100"] - P["p500"])
            + d500 * P["p500"])
    eal = frac * df["building_value"]
    return eal.where(bfe.notna(), 0.0).fillna(0.0)
