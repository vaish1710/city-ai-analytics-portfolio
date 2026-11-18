# Live Flood Nowcast: Elizabeth River water level, 1-6 hours ahead

An AI project for the **Enterprise AI & Analytics Program Manager** role (City of Chesapeake, IT).

## The problem
Chesapeake is spending state grant money to raise Battlefield Boulevard so people can reach medical facilities during floods
([VCN](https://vcnva.org/wp-content/uploads/2025/01/City-of-Chesapeake-RGGI-CFPF-2024.pdf)). Until roads like that are raised, a short-range
warning of when the water will cross a flood level helps crews decide when to close or monitor roads.

## What it does
NOAA publishes the **astronomical tide** ahead of time. What it cannot predict is the **non-tidal residual** (storm surge, wind, pressure).
This project forecasts that residual 1, 3 and 6 hours ahead from the latest observations and weather, adds it to the published tide, and judges the
result as a flood warning: how many floods it catches (POD), how many alerts were false (FAR), and the combined score (CSI).

- Station: **NOAA 8639348, Money Point, VA** (Elizabeth River, near Chesapeake; check its location against the roads you care about). Confirmed to exist as a NOAA PORTS station.
- Data: NOAA's public CO-OPS API (water level, tide predictions, and wind and pressure if the station has them).
- Methods compared at each horizon, all tuned identically: **published tide only**, **persistence** (residual stays as it is), **gradient-boosting model**, and a cautious **90th-percentile model**.
- The model learns the *change* from persistence. My first version predicted the raw residual and was worse than persistence at 1 hour; changing the target fixed it.

## Be clear about what this is not
- **Not new.** The Virginia Institute of Marine Science already publishes a 36-hour forecast of this same non-tidal water level at Money Point
  ([VIMS TideWatch](https://www.vims.edu/bayinfo/tidewatch/stations/mnpt/index.php)). The honest real-data test is to beat persistence and then compare against VIMS's archived forecasts. I make no claim to beat VIMS.
- **Not an official warning.** Use NWS/NOAA and city alerts.
- **Tide and surge only.** Chesapeake road flooding also comes from heavy rain, which this model does not see.

## Results so far: SYNTHETIC data only
Simulated tide plus wind-driven surge, 3 years, last 20% held out in time. Flood threshold = 99th percentile of training data (a stand-in). Lower RMSE is better; higher CSI is better.

| Horizon | RMSE persistence | RMSE model | CSI persistence | CSI model |
|---|---|---|---|---|
| 1 h | 0.054 ft | 0.044 ft | 0.880 | 0.918 |
| 3 h | 0.077 ft | 0.058 ft | 0.794 | 0.864 |
| 6 h | 0.118 ft | 0.087 ft | 0.707 | 0.797 |

This shows the pipeline learns a pattern that exists in the data. It says **nothing** about real Money Point skill. The simulator's surge is smoother and more
predictable than the real thing.

## Run it
```
pip install pandas numpy scikit-learn matplotlib requests joblib
python src/run_eval.py                                  # synthetic demo, works offline
python src/run_eval.py --source noaa --station 8639348 --begin 20240101 --end 20260101 --threshold-ft <OFFICIAL FLOOD LEVEL>
python src/live.py                                      # one forecast now (needs the real-data model above)
python src/live.py --watch 900                          # repeat every 15 minutes, appends to outputs/live_log.csv
```
**You must pass `--threshold-ft` from NOAA/NWS flood levels for the station, in the same datum** (default MLLW; use `--datum` if the station supports another).
I could not find the official levels, so none is built in. Without the flag a percentile stand-in is used and a warning is printed.

## What was tested (13 tests, run offline against a fake NOAA server)
- Features **never use the future**: corrupting every later observation leaves the features unchanged. A deliberately planted leak makes this test fail.
- Time splits are chronological with a gap; alert margins are tuned on validation data, never on the test block.
- Alert metrics (POD/FAR/CSI) checked against hand-worked examples.
- Long date ranges are split under NOAA's per-request limit with no rows lost or duplicated; request parameters and error handling checked; missing wind/pressure sensors handled.
- Live path and offline path give the same forecast; stale data refuses to forecast.

## Not verified (never run against live NOAA)
- Real responses: the wind and pressure field names ("s", "v") are from my recollection, and sensor availability at this station is unknown. Missing met data is handled, but check the first real download.
- The datum, time alignment of 6-minute observations (minute :00) with hourly predictions, and gaps in real data.
- Real surge dynamics, including tide-surge interaction, will be harder than the simulation.
- Official flood thresholds (see above).

## If you evaluate on real data
Report skill against persistence first. Require enough real events in the test block (the tool prints the count; a handful of events means any score is noise). Look at performance by season and storm type, and compare with VIMS's forecast archive.
