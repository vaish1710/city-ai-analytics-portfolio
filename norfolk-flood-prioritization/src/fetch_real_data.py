"""Download REAL Norfolk data and assemble the parcel table used by run.py.

Run this on your own computer (the cloud workspace used to write it cannot reach these hosts,
so this script has NOT been executed against the live servers; start with `discover`).

Step 1  python src/fetch_real_data.py discover
        Lists layers and field names on Norfolk's ArcGIS server. The parcel layer holds only
        IDs and geometry; assessed value / square feet / year built live in the Property
        Information layers, so you must pick that layer and its field names.
Step 2  python src/fetch_real_data.py build \
            --property-layer https://gisshare.norfolk.gov/pubserver/rest/services/OpenData/Property_Information_Group_OpenData/MapServer/<ID> \
            --prop-key <FIELD matching parcel TAX_ACCT or GPIN> \
            --map building_value=<FIELD>,sqft=<FIELD>,year_built=<FIELD>[,foundation=<FIELD>] \
            --limit 3000
Step 3  python src/run.py --data data/parcels_real.csv --budget 25000000

Requires: pip install requests pandas geopandas shapely pyproj
"""
import argparse
import io
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

BASE = "https://gisshare.norfolk.gov/pubserver/rest/services"
PARCELS = f"{BASE}/OpenData/Parcels/MapServer/1"          # fields: TAX_ACCT, GPIN, geometry
FIRM = f"{BASE}/OpenData/Environmental/MapServer/4"       # FEMA flood zones (polygon)
PROP_ROOT = f"{BASE}/OpenData/Property_Information_Group_OpenData/MapServer"
EPQS = "https://epqs.nationalmap.gov/v1/json"             # USGS elevation point query service
ACS = "https://api.census.gov/data/2023/acs/acs5"        # state 51, county 710 = Norfolk city
TRACTS = "https://www2.census.gov/geo/tiger/TIGER2023/TRACT/tl_2023_51_tract.zip"
S = requests.Session()


def get_json(url, **params):
    params.setdefault("f", "pjson")
    r = S.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def discover():
    for name, root in [("Parcels", f"{BASE}/OpenData/Parcels/MapServer"),
                       ("Environmental", f"{BASE}/OpenData/Environmental/MapServer"),
                       ("Property info", PROP_ROOT)]:
        print(f"\n=== {name}: {root}")
        for lyr in get_json(root).get("layers", []):
            info = get_json(f"{root}/{lyr['id']}")
            fields = ", ".join(f["name"] for f in info.get("fields", []))
            print(f"  [{lyr['id']}] {lyr['name']} ({info.get('geometryType', 'table')})\n      {fields}")


def query_all(layer_url, geometry=True, out_fields="*"):
    """Page through a layer by object id (server caps each response at 2000 records)."""
    import geopandas as gpd
    ids = get_json(f"{layer_url}/query", where="1=1", returnIdsOnly="true")["objectIds"]
    frames = []
    for i in range(0, len(ids), 1000):
        chunk = ids[i:i + 1000]
        r = S.post(f"{layer_url}/query", data={
            "objectIds": ",".join(map(str, chunk)), "outFields": out_fields, "f": "geojson",
            "returnGeometry": str(geometry).lower(), "outSR": 4326}, timeout=120)
        r.raise_for_status()
        frames.append(gpd.GeoDataFrame.from_features(r.json()["features"], crs="EPSG:4326"))
    return pd.concat(frames, ignore_index=True)


def elevation(lon_lat):
    lon, lat = lon_lat
    for _ in range(3):
        try:
            v = S.get(EPQS, params={"x": lon, "y": lat, "wkid": 4326, "units": "Feet", "includeDate": "false"},
                      timeout=30).json().get("value")
            return float(v) if v is not None else None
        except Exception:
            continue
    return None


def pick(df, *names):
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def build(a):
    import geopandas as gpd
    print("parcels..."); parcels = query_all(PARCELS)
    print("flood zones..."); firm = query_all(FIRM)
    zcol = pick(firm, "FLD_ZONE"); sub = pick(firm, "ZONE_SUBTY"); bcol = pick(firm, "STATIC_BFE")
    if not zcol:
        sys.exit(f"FIRM layer has no FLD_ZONE field; columns: {list(firm.columns)}")
    firm["fema_zone"] = firm[zcol].astype(str)
    if sub:  # distinguish shaded X (0.2% chance) from unshaded X
        shaded = firm[sub].astype(str).str.contains("0.2", na=False)
        firm.loc[(firm["fema_zone"] == "X") & shaded, "fema_zone"] = "X-shaded"
    firm["bfe_ft"] = pd.to_numeric(firm[bcol], errors="coerce").replace(-9999, float("nan")) if bcol else float("nan")

    parcels["geometry_poly"] = parcels.geometry
    parcels["geometry"] = parcels.geometry.to_crs(32618).centroid.to_crs(4326)  # UTM 18N for centroids
    j = gpd.sjoin(parcels, firm[["fema_zone", "bfe_ft", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    j = j[j["fema_zone"].isin(["AE", "VE", "AO", "A", "AH", "X-shaded"])]  # only parcels with flood exposure
    print(f"{len(j)} parcels in flood hazard zones")

    print("property attributes..."); prop = query_all(a.property_layer, geometry=False)
    colmap = dict(kv.split("=") for kv in a.map.split(","))
    prop = prop.rename(columns={v: k for k, v in colmap.items()})
    keep = ["building_value", "sqft", "year_built"] + (["foundation"] if "foundation" in prop else [])
    pk = pick(parcels, a.parcel_key)
    j = j.merge(prop[[a.prop_key] + keep].drop_duplicates(a.prop_key), left_on=pk, right_on=a.prop_key, how="inner")
    for c in ["building_value", "sqft", "year_built"]:
        j[c] = pd.to_numeric(j[c], errors="coerce")
    j = j.dropna(subset=["building_value", "sqft"]).query("building_value > 0 and sqft > 0")
    if len(j) > a.limit:
        j = j.sample(a.limit, random_state=0)
    print(f"{len(j)} parcels after attribute join; looking up elevation (USGS 3DEP)...")
    pts = list(zip(j.geometry.x, j.geometry.y))
    with ThreadPoolExecutor(8) as ex:
        j["elevation_ft"] = list(ex.map(elevation, pts))
    j = j.dropna(subset=["elevation_ft"])

    print("census income by tract...")
    acs = S.get(ACS, params={"get": "B19013_001E", "for": "tract:*", "in": "state:51 county:710"}, timeout=60).json()
    inc = pd.DataFrame(acs[1:], columns=acs[0])
    inc["GEOID"] = inc["state"] + inc["county"] + inc["tract"]
    inc["tract_median_income"] = pd.to_numeric(inc["B19013_001E"], errors="coerce").where(lambda s: s > 0)
    tr = gpd.read_file(io.BytesIO(S.get(TRACTS, timeout=300).content))
    tr = tr[tr["COUNTYFP"] == "710"].merge(inc[["GEOID", "tract_median_income"]], on="GEOID").to_crs(4326)
    j = gpd.sjoin(j.drop(columns=["index_right"], errors="ignore"), tr[["GEOID", "tract_median_income", "geometry"]],
                  how="left", predicate="within")
    j["tract_median_income"] = j["tract_median_income"].fillna(j["tract_median_income"].median())

    out = pd.DataFrame({
        "parcel_id": j[pk].values, "neighborhood": j["GEOID"].values,  # census tract stands in for neighborhood
        "elevation_ft": j["elevation_ft"].round(2).values, "fema_zone": j["fema_zone"].values,
        "bfe_ft": j["bfe_ft"].values, "building_value": j["building_value"].values, "sqft": j["sqft"].values,
        "year_built": j["year_built"].values,
        "foundation": j["foundation"].values if "foundation" in j else "unknown",
        "tract_median_income": j["tract_median_income"].values,
    })
    out.to_csv("data/parcels_real.csv", index=False)
    print(f"wrote data/parcels_real.csv rows={len(out)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("discover")
    b = sub.add_parser("build")
    b.add_argument("--property-layer", required=True)
    b.add_argument("--prop-key", required=True, help="field in the property layer that matches the parcel key")
    b.add_argument("--parcel-key", default="TAX_ACCT", help="TAX_ACCT or GPIN on the parcel layer")
    b.add_argument("--map", required=True, help="building_value=F1,sqft=F2,year_built=F3[,foundation=F4]")
    b.add_argument("--limit", type=int, default=3000)
    args = ap.parse_args()
    discover() if args.cmd == "discover" else build(args)
