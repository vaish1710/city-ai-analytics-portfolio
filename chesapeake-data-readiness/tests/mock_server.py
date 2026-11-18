"""A tiny fake ArcGIS MapServer so the audit can be tested end to end with no internet."""
import json
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

NOW = datetime(2026, 11, 17, tzinfo=timezone.utc)


def ms(days_ago):
    return int((NOW - timedelta(days=days_ago)).timestamp() * 1000)


def build_layers():
    # Layer 0: tidy parcels. 50 records, unique PIN, aliases, description, edited 10 days ago.
    good = [{"OBJECTID": i, "PIN": f"P{i:04d}", "OWNER_CLASS": "R" if i % 2 else "C", "ACRES": 0.1 + i / 100}
            for i in range(1, 51)]
    # Layer 1: messy facilities. 40 records. FAC_ID duplicated (20 distinct), CITY constant,
    # NOTES empty in 30 of 40, no aliases, no description, edited 800 days ago.
    messy = [{"OBJECTID": i, "FAC_ID": f"F{(i - 1) % 20}", "CITY": "Chesapeake",
              "NOTES": ("see file" if i <= 10 else None), "NAME": f"Site {i}"} for i in range(1, 41)]
    # Layer 2: a table with no geometry, 30 records, no edit date published.
    table = [{"OBJECTID": i, "CASE_NUMBER": f"C{i:03d}", "STATUS": "open" if i % 3 else "closed"} for i in range(1, 31)]
    f = lambda n, t="esriFieldTypeString", alias=None: {"name": n, "type": t, "alias": alias or n}
    return {
        0: {"name": "Parcels (mock)", "geometryType": "esriGeometryPolygon", "description": "Mock parcels.",
            "editingInfo": {"lastEditDate": ms(10)}, "records": good, "geometry_null_ids": set(),
            "fields": [f("OBJECTID", "esriFieldTypeOID"), f("PIN", alias="Parcel Number"),
                       f("OWNER_CLASS", alias="Owner Class"), f("ACRES", "esriFieldTypeDouble", "Acres")]},
        1: {"name": "Facilities (mock)", "geometryType": "esriGeometryPoint", "description": "",
            "editingInfo": {"lastEditDate": ms(800)}, "records": messy, "geometry_null_ids": set(range(1, 11)),
            "fields": [f("OBJECTID", "esriFieldTypeOID"), f("FAC_ID"), f("CITY"), f("NOTES"), f("NAME")]},
        2: {"name": "Cases table (mock)", "geometryType": None, "description": "A table.",
            "records": table, "geometry_null_ids": set(),
            "fields": [f("OBJECTID", "esriFieldTypeOID"), f("CASE_NUMBER", alias="Case Number"), f("STATUS", alias="Status")]},
    }


LAYERS = build_layers()


class H(BaseHTTPRequestHandler):
    refuse_ids = False  # set True to test the fallback path

    def log_message(self, *a):
        pass

    def _params(self):
        u = urlparse(self.path)
        p = {k: v[0] for k, v in parse_qs(u.query).items()}
        if self.command == "POST":
            body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
            p.update({k: v[0] for k, v in parse_qs(body).items()})
        return u.path.rstrip("/"), p

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def _handle(self):
        path, p = self._params()
        parts = path.split("/")
        if parts[-1] == "MapServer":
            return self._send({"layers": [{"id": i, "name": L["name"]} for i, L in LAYERS.items() if L["geometryType"]],
                               "tables": [{"id": i, "name": L["name"]} for i, L in LAYERS.items() if not L["geometryType"]]})
        if parts[-1] == "query":
            L = LAYERS[int(parts[-2])]
            if p.get("returnCountOnly") == "true":
                return self._send({"count": len(L["records"])})
            if p.get("returnIdsOnly") == "true":
                if H.refuse_ids:
                    return self._send({"error": {"code": 400, "message": "returnIdsOnly not supported"}})
                return self._send({"objectIdFieldName": "OBJECTID", "objectIds": [r["OBJECTID"] for r in L["records"]]})
            ids = [int(x) for x in p["objectIds"].split(",")] if "objectIds" in p else [r["OBJECTID"] for r in L["records"]]
            feats = []
            for r in L["records"]:
                if r["OBJECTID"] in ids:
                    ft = {"attributes": r}
                    if p.get("returnGeometry") == "true" and L["geometryType"]:
                        ft["geometry"] = None if r["OBJECTID"] in L["geometry_null_ids"] else {"x": 1, "y": 2}
                    feats.append(ft)
            return self._send({"features": feats})
        L = LAYERS[int(parts[-1])]
        out = {k: v for k, v in L.items() if k not in ("records", "geometry_null_ids")}
        return self._send(out)

    do_GET = _handle
    do_POST = _handle


def start(port=0):
    srv = HTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/mapping/rest/services/OpenData/OpenData/MapServer"
