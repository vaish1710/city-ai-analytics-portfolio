"""Fake NOAA CO-OPS datagetter, serving a DataFrame, recording every request."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd

NO_DATA = {"error": {"message": "No data was found. This product may not be offered at this station at the requested time."}}


def make_handler(df, state):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            p = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            state["requests"].append(p)
            prod = p["product"]
            if state.get("force_error"):
                body = {"error": {"message": "Wrong Date: The end date should be greater than the begin date"}}
            elif prod in state.get("missing", ()):
                body = NO_DATA
            else:
                b, e = pd.Timestamp(p["begin_date"], tz="UTC"), pd.Timestamp(p["end_date"], tz="UTC")
                col = {"water_level": "observed", "predictions": "predicted", "wind": "wind_speed", "air_pressure": "pressure"}[prod]
                s = df[col].dropna().loc[b:e]
                if s.empty:
                    body = NO_DATA
                else:
                    t = [i.strftime("%Y-%m-%d %H:%M") for i in s.index]
                    if prod == "predictions":
                        body = {"predictions": [{"t": a, "v": f"{v:.3f}"} for a, v in zip(t, s)]}
                    elif prod == "wind":
                        body = {"metadata": {}, "data": [{"t": a, "s": f"{v:.2f}", "d": "90", "dr": "E", "g": "9.9", "f": "0,0"} for a, v in zip(t, s)]}
                    else:
                        body = {"metadata": {}, "data": [{"t": a, "v": f"{v:.3f}", "s": "0.01", "f": "0,0,0,0", "q": "v"} for a, v in zip(t, s)]}
            raw = json.dumps(body).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)
    return H


def start(df):
    state = {"requests": []}
    srv = HTTPServer(("127.0.0.1", 0), make_handler(df, state))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/api/prod/datagetter", state
