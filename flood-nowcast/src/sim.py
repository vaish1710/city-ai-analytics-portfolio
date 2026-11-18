"""SYNTHETIC hourly tide + wind-driven surge, for testing and for demonstrating the method offline.

Nothing here is real Money Point data. The point is a series where the non-tidal part (surge) is partly
predictable from wind, pressure and its own recent history, so we can check the pipeline learns what it should.
"""
import numpy as np
import pandas as pd

# (period hours, amplitude ft): rough orders of magnitude only
CONSTITUENTS = [(12.4206, 1.30), (12.0, 0.25), (12.6583, 0.30), (23.9345, 0.20), (25.8193, 0.15)]


def make_series(days=1095, seed=7, start="2023-01-01"):
    rng = np.random.default_rng(seed)
    n = days * 24
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    h = np.arange(n, dtype=float)
    phases = rng.uniform(0, 2 * np.pi, len(CONSTITUENTS))
    tide = 1.6 + sum(a * np.cos(2 * np.pi * h / p + ph) for (p, a), ph in zip(CONSTITUENTS, phases))

    # Onshore wind (m/s): slow AR(1) background plus storm pulses (nor'easter-like, 1-3 days).
    u = np.zeros(n)
    for i in range(1, n):
        u[i] = 0.97 * u[i - 1] + rng.normal(0, 0.8)
    storm = np.zeros(n)
    for c in rng.choice(n, size=max(1, days // 25), replace=False):
        width = rng.uniform(10, 24)
        storm += rng.normal(10, 3) * np.exp(-0.5 * ((h - c) / width) ** 2)
    wind = u + storm
    pressure = 1015 - 0.7 * storm + np.cumsum(rng.normal(0, 0.15, n)) * 0 + rng.normal(0, 1.5, n)

    # Surge responds to wind stress and low pressure with a lag (exponential smoothing, ~6 h).
    forcing = 0.006 * np.sign(wind) * wind ** 2 + 0.0328 * (1013 - pressure)
    surge = np.zeros(n)
    for i in range(1, n):
        surge[i] = surge[i - 1] + (forcing[i] - surge[i - 1]) / 6.0
    surge += rng.normal(0, 0.03, n)

    return pd.DataFrame({
        "observed": tide + surge + rng.normal(0, 0.02, n),
        "predicted": tide,           # astronomical prediction: what NOAA publishes ahead of time
        "wind_speed": np.abs(wind),  # speed only (direction omitted)
        "pressure": pressure,
        "_true_surge": surge,        # for tests only, never a feature
    }, index=idx)
