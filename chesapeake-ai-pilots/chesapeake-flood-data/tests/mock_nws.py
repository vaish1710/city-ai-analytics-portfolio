"""Fake api.weather.gov in the response shapes this project assumes. Records every request (path, query, headers)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse


def make_handler(state):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def _send(self, code, body):
            raw = json.dumps(body).encode()
            self.send_response(code); self.send_header("Content-Type", "application/geo+json"); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)

        def do_GET(self):
            u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
            state["requests"].append({"path": u.path, "query": q, "ua": self.headers.get("User-Agent")})
            host = f"http://127.0.0.1:{self.server.server_port}"
            if state.get("fail_next", 0) > 0:
                state["fail_next"] -= 1; return self._send(503, {"title": "unavailable"})
            if u.path.startswith("/points/"):
                return self._send(200, {"properties": {"gridId": "AKQ", "forecastGridData": host + "/gridpoints/AKQ/1,2", "observationStations": host + "/gridpoints/AKQ/1,2/stations"}})
            if u.path == "/gridpoints/AKQ/1,2":
                return self._send(200, {"properties": {"updateTime": state["update_time"],
                    "quantitativePrecipitation": {"uom": state.get("qpf_uom", "wmoUnit:mm"), "values": state["qpf"]},
                    "probabilityOfPrecipitation": {"uom": "wmoUnit:percent", "values": state["pop"]}}})
            if u.path == "/gridpoints/AKQ/1,2/stations":
                return self._send(200, {"features": [{"properties": {"stationIdentifier": s}} for s in state["stations"]]})
            if u.path.startswith("/stations/") and u.path.endswith("/observations"):
                return self._send(200, {"features": [{"properties": {"timestamp": t, "precipitationLastHour": ({"value": v, "unitCode": "wmoUnit:mm"} if v is not None else {"value": None, "unitCode": "wmoUnit:mm"})}} for t, v in state["obs"]]})
            return self._send(404, {"title": "not found"})
    return H


def start(**state):
    state.setdefault("requests", []); state.setdefault("stations", ["KAAA", "KBBB"]); state.setdefault("update_time", "2026-01-01T00:00:00+00:00")
    state.setdefault("qpf", []); state.setdefault("pop", []); state.setdefault("obs", [])
    srv = HTTPServer(("127.0.0.1", 0), make_handler(state))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}", state
