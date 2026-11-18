"""Minimal ArcGIS REST client: layer listing, metadata, counts, and a random record sample."""
import random
import time

import requests

UA = "chesapeake-data-readiness/0.1 (portfolio analysis)"


class ArcGISError(RuntimeError):
    pass


class Client:
    def __init__(self, session=None, timeout=60, retries=3):
        self.s = session or requests.Session()
        self.s.headers["User-Agent"] = UA
        self.timeout, self.retries = timeout, retries

    def get(self, url, params=None):
        params = dict(params or {})
        params["f"] = "json"
        last = None
        for i in range(self.retries):
            try:
                if len(str(params)) > 1500:  # long objectIds lists: POST avoids URL-length limits
                    r = self.s.post(url, data=params, timeout=self.timeout)
                else:
                    r = self.s.get(url, params=params, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                if "error" in data:
                    raise ArcGISError(data["error"])
                return data
            except ArcGISError:
                raise  # a server-side refusal will not improve on retry
            except Exception as e:  # network or decode problem: back off and retry
                last = e
                time.sleep(0.5 * 2 ** i)
        raise last

    def layers(self, service_url):
        info = self.get(service_url)
        return [(l["id"], l["name"]) for l in info.get("layers", [])] + \
               [(t["id"], t["name"]) for t in info.get("tables", [])]

    def meta(self, layer_url):
        return self.get(layer_url)

    def count(self, layer_url):
        return int(self.get(layer_url + "/query", {"where": "1=1", "returnCountOnly": "true"})["count"])

    def sample(self, layer_url, n=2000, seed=42, chunk=200, geom_check=100):
        """Random sample of attribute records, plus the share of null geometries in a small subsample.

        A random sample (not the first N rows) avoids bias toward the oldest records.
        Returns (records, null_geometry_share or None, how) where `how` describes the sampling used.
        """
        q = layer_url + "/query"
        try:
            ids = self.get(q, {"where": "1=1", "returnIdsOnly": "true"})
            all_ids = ids.get("objectIds") or []
            chosen = random.Random(seed).sample(all_ids, min(n, len(all_ids)))
            records = []
            for i in range(0, len(chosen), chunk):
                part = chosen[i:i + chunk]
                d = self.get(q, {"objectIds": ",".join(map(str, part)), "outFields": "*", "returnGeometry": "false"})
                records += [f["attributes"] for f in d.get("features", [])]
            null_geom = None
            if chosen and geom_check:
                d = self.get(q, {"objectIds": ",".join(map(str, chosen[:geom_check])), "outFields": "OBJECTID",
                                 "returnGeometry": "true"})
                feats = d.get("features", [])
                if feats:
                    null_geom = sum(1 for f in feats if not f.get("geometry")) / len(feats)
            return records, null_geom, f"random sample of {len(records)} of {len(all_ids)} records"
        except ArcGISError:
            # Some layers refuse returnIdsOnly. Fall back to the first records (biased toward old rows).
            d = self.get(q, {"where": "1=1", "outFields": "*", "returnGeometry": "false"})
            recs = [f["attributes"] for f in d.get("features", [])][:n]
            return recs, None, f"FALLBACK: first {len(recs)} records only (biased toward oldest rows)"
