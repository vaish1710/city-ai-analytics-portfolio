"""National Weather Service API client (api.weather.gov): forecast grid (rain amounts, chance of rain) and station observations.

WHAT IS FROM MEMORY, NOT VERIFIED AGAINST THE LIVE API (the build environment cannot reach it): the response shapes below.
  /points/{lat},{lon}             -> properties.forecastGridData, properties.observationStations
  forecastGridData                -> properties.quantitativePrecipitation / probabilityOfPrecipitation
                                     {uom, values:[{validTime:"2026-01-01T00:00:00+00:00/PT6H", value}]}, properties.updateTime
  observationStations             -> features[].properties.stationIdentifier
  /stations/{id}/observations     -> features[].properties {timestamp, precipitationLastHour:{value, unitCode}}
NWS asks every client to send an identifying User-Agent; `session(contact)` requires you to supply yours.
The parsers raise clear errors on unexpected units or shapes, so a mismatch shows up on the first live run instead of as wrong numbers.
"""
import re
import time

import pandas as pd
import requests

BASE = "https://api.weather.gov"
DUR = re.compile(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?)?$")


class NwsError(RuntimeError):
    pass


def parse_duration_hours(s: str) -> float:
    m = DUR.match(s or "")
    if not m or not any(m.groups()):
        raise NwsError(f"unrecognized ISO-8601 duration: {s!r}")
    d, h, mi = (int(x or 0) for x in m.groups())
    return d * 24 + h + mi / 60


def parse_valid_time(s: str):
    try:
        start, dur = s.split("/")
    except ValueError:
        raise NwsError(f"unrecognized validTime: {s!r}")
    t = pd.Timestamp(start)
    return (t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")), parse_duration_hours(dur)


def session(contact: str) -> requests.Session:
    if not contact or "@" not in contact and "http" not in contact:
        raise ValueError("NWS asks for an identifying contact (an email or a URL) in the User-Agent; pass --contact")
    s = requests.Session()
    s.headers.update({"User-Agent": f"(chesapeake-flood-data, {contact})", "Accept": "application/geo+json"})
    return s


def get_json(s, url, retries=3, **params):
    for i in range(retries):
        r = s.get(url, params=params or None, timeout=60)
        if r.status_code in (429, 500, 502, 503, 504) and i < retries - 1:
            time.sleep(0.5 * 2 ** i)
            continue
        if r.status_code == 404:
            raise NwsError(f"404 for {url}")
        r.raise_for_status()
        return r.json()


def locate(s, lat, lon, base=BASE):
    p = get_json(s, f"{base}/points/{lat:.4f},{lon:.4f}")["properties"]
    return {"grid_url": p["forecastGridData"], "stations_url": p["observationStations"], "wfo": p.get("gridId")}


def grid_series(js, name, uom_expected):
    try:
        block = js["properties"][name]
    except KeyError:
        raise NwsError(f"forecast grid has no {name!r}")
    if uom_expected and block.get("uom") != uom_expected:
        raise NwsError(f"{name}: expected unit {uom_expected}, got {block.get('uom')}")
    rows = []
    for v in block["values"]:
        if v["value"] is None:
            continue
        start, hours = parse_valid_time(v["validTime"])
        rows.append((start, hours, float(v["value"])))
    return rows


def updated_at(js):
    t = js["properties"].get("updateTime")
    if not t:
        return None
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def station_ids(s, stations_url):
    return [f["properties"]["stationIdentifier"] for f in get_json(s, stations_url)["features"]]


def observations(s, station, start, end, base=BASE):
    """(timestamp, mm of rain in the last hour) for observations that report it. Many stations report it rarely or never."""
    js = get_json(s, f"{base}/stations/{station}/observations", start=pd.Timestamp(start).isoformat(), end=pd.Timestamp(end).isoformat())
    out = []
    for f in js["features"]:
        p = f["properties"]
        pr = p.get("precipitationLastHour") or {}
        if pr.get("value") is None:
            continue
        if pr.get("unitCode") != "wmoUnit:mm":
            raise NwsError(f"precipitationLastHour unit {pr.get('unitCode')!r} is not mm")
        t = pd.Timestamp(p["timestamp"])
        out.append(((t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")), float(pr["value"])))
    return out
