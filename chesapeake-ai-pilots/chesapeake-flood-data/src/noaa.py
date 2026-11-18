"""NOAA CO-OPS data client (water level, tide predictions, wind, air pressure).

Facts taken from NOAA's API page: base URL below; `datum` is required for water_level and predictions;
6-minute data is limited to 1 month per request, predictions to 1 year; errors arrive as {"error": {"message": ...}}.
NOT verified from here: the exact JSON field names of wind/air_pressure rows ("s" speed, "g" gust, "v" value are used,
as I recall them), and whether a given station has wind or pressure sensors. The parser is tolerant: missing met data
becomes NaN, and the model is trained without those features.
"""
import time
from datetime import timedelta

import pandas as pd
import requests

BASE = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
APP = "flood-nowcast-portfolio"  # NOAA asks clients to identify the application; no personal details used
CHUNK_DAYS = {"water_level": 28, "wind": 28, "air_pressure": 28, "predictions": 360}  # under the stated limits


class NoaaError(RuntimeError):
    pass


def _utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _windows(begin: pd.Timestamp, end: pd.Timestamp, days: int):
    s = begin
    while s < end:
        e = min(s + timedelta(days=days), end)
        yield s, e
        s = e


def _parse(js: dict, product: str) -> pd.DataFrame:
    if "error" in js:
        msg = str(js["error"].get("message", js["error"]))
        if "no data" in msg.lower():
            return pd.DataFrame()
        raise NoaaError(msg)
    rows = js.get("predictions") if product == "predictions" else js.get("data")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    out = pd.DataFrame(index=pd.to_datetime(df["t"], utc=True))
    col = "s" if product == "wind" else "v"
    out["value"] = pd.to_numeric(df[col], errors="coerce").values
    return out


def fetch(station, product, begin, end, datum="MLLW", base_url=BASE, session=None, interval=None, retries=3):
    """Return a UTC-indexed DataFrame with one column, 'value' (feet, knots or millibars). Empty if no data."""
    s = session or requests.Session()
    begin, end = _utc(begin), _utc(end)
    parts = []
    for ws, we in _windows(begin, end, CHUNK_DAYS[product]):
        params = {"product": product, "application": APP, "station": station,
                  "begin_date": ws.strftime("%Y%m%d %H:%M"), "end_date": we.strftime("%Y%m%d %H:%M"),
                  "time_zone": "gmt", "units": "english", "format": "json"}
        if product in ("water_level", "predictions"):
            params["datum"] = datum
        if interval:
            params["interval"] = interval
        for i in range(retries):
            try:
                r = s.get(base_url, params=params, timeout=60)
                r.raise_for_status()
                parts.append(_parse(r.json(), product))
                break
            except NoaaError:
                raise
            except Exception:
                if i == retries - 1:
                    raise
                time.sleep(0.5 * 2 ** i)
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=["value"], index=pd.DatetimeIndex([], tz="UTC"))
    df = pd.concat(parts)
    return df[~df.index.duplicated(keep="first")].sort_index()


def build_dataset(station, begin, end, datum="MLLW", with_met=True, **kw) -> pd.DataFrame:
    """Hourly frame: observed, predicted, and (if the station has them) wind_speed, pressure.

    Observed/met are the 6-minute values at minute :00, so they line up with hourly predictions.
    """
    obs = fetch(station, "water_level", begin, end, datum, **kw)
    pred = fetch(station, "predictions", begin, end, datum, interval="h", **kw)
    out = pd.DataFrame({"observed": obs["value"][obs.index.minute == 0],
                        "predicted": pred["value"]}).sort_index()
    if with_met:
        for name, product in (("wind_speed", "wind"), ("pressure", "air_pressure")):
            try:
                m = fetch(station, product, begin, end, **kw)
                out[name] = m["value"][m.index.minute == 0] if not m.empty else float("nan")
            except NoaaError:
                out[name] = float("nan")
    return out.asfreq("h")
