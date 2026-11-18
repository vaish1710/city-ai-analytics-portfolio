"""Generate SYNTHETIC parcel data shaped like what you would assemble from real sources.

Nothing here is real Norfolk data. Neighborhood names are used only as labels so the
demo reads naturally. Replace this module's output with the real join described in the
README ("Swapping in real data") before drawing any conclusion.
"""
import numpy as np
import pandas as pd

NEIGHBORHOODS = {
    # name: (mean elevation ft, mean tract income $, exposure multiplier)
    "Willoughby": (4.5, 62000, 1.3),
    "Ocean View": (5.5, 58000, 1.2),
    "Ingleside": (6.5, 72000, 1.0),
    "Berkley": (5.0, 41000, 1.3),
    "Campostella": (5.5, 39000, 1.2),
    "Chesterfield Heights": (6.0, 47000, 1.1),
    "Larchmont": (9.0, 98000, 0.7),
    "Colonial Place": (7.0, 90000, 0.9),
}


def make_parcels(n: int = 4000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(NEIGHBORHOODS)
    hood = rng.choice(names, size=n, p=[0.14, 0.14, 0.12, 0.13, 0.11, 0.1, 0.13, 0.13])
    mean_elev = np.array([NEIGHBORHOODS[h][0] for h in hood])
    mean_inc = np.array([NEIGHBORHOODS[h][1] for h in hood])
    expo = np.array([NEIGHBORHOODS[h][2] for h in hood])

    elevation = np.clip(rng.normal(mean_elev, 1.2), 1.0, 14.0)  # ft NAVD88
    dist_water = np.clip(rng.gamma(2.0, 180, n) / expo, 15, 2500)  # metres
    sqft = np.clip(rng.normal(1500, 450, n), 600, 4000)
    year_built = rng.integers(1925, 2022, n)
    foundation = rng.choice(["slab", "crawlspace", "basement"], size=n, p=[0.35, 0.55, 0.10])
    income = np.clip(rng.normal(mean_inc, 9000), 22000, 160000)
    owner_occ = rng.random(n) < np.clip(income / 130000 + 0.25, 0.2, 0.9)
    building_value = np.clip(sqft * rng.normal(130, 25, n) * (0.6 + income / 150000), 60000, 900000)

    # Hidden ground truth used only to create labels for the demo.
    design_flood_elev = 8.5  # ft: stand-in for base flood elevation + tidal scenario
    depth = np.clip(design_flood_elev - elevation, 0, None) * np.exp(-dist_water / 1500)
    base_prob = np.clip(0.04 + 0.22 * (depth > 0) + 0.04 * depth, 0, 0.9)  # annual flood probability
    dmg_frac = np.clip(0.08 + 0.12 * depth, 0, 0.7) * np.where(foundation == "basement", 1.35, 1.0)
    true_annual_loss = base_prob * dmg_frac * building_value

    # Observable features derived from the hidden truth, with noise.
    reports_311 = rng.poisson(np.clip(base_prob * 6, 0.05, None))
    prior_claims = rng.poisson(np.clip(base_prob * 1.2, 0.02, None))
    fema_zone = np.where(depth > 1.0, "AE", np.where(depth > 0, "X-shaded", "X"))
    observed_loss = np.clip(true_annual_loss * rng.lognormal(0, 0.35, n) + rng.normal(0, 150, n), 0, None)

    return pd.DataFrame({
        "parcel_id": [f"P{i:05d}" for i in range(n)],
        "neighborhood": hood,
        "elevation_ft": elevation.round(2),
        "dist_to_water_m": dist_water.round(0),
        "fema_zone": fema_zone,
        "building_value": building_value.round(0),
        "sqft": sqft.round(0),
        "year_built": year_built,
        "foundation": foundation,
        "flood_reports_311_5y": reports_311,
        "prior_claims_5y": prior_claims,
        "tract_median_income": income.round(0),
        "owner_occupied": owner_occ,
        "observed_annual_loss": observed_loss.round(0),
        "_true_annual_loss": true_annual_loss.round(0),  # never used as a model feature
    })


if __name__ == "__main__":
    df = make_parcels()
    df.to_csv("data/parcels_synthetic.csv", index=False)
    print(f"wrote data/parcels_synthetic.csv  rows={len(df)}")
