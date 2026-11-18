# Chesapeake flood data pipeline (NWS forecasts + NOAA tides -> storm table)

Feeds `road-flood-risk` with the one thing it cannot get anywhere else: **the forecast that existed before each storm**.
The NWS API gives current forecasts but, as far as I know, no archive of past ones, so this collector saves a snapshot every hour starting now.

    pip install pandas requests pytest
    # 1. keep this running (always-on machine; cron example below). NWS asks for an identifying contact.
    python src/collect.py --db data/chesapeake.db --contact you@example.com once
    #    hourly cron:   0 * * * *  cd /path/to/chesapeake-flood-data && python src/collect.py --db data/chesapeake.db --contact you@example.com once
    # 2. later, build the storm table (observed rain from the collector, or from an hourly CSV; tides from NOAA)
    python src/build_storm_table.py --db data/chesapeake.db --begin 2026-12-01 --end 2027-06-01 --out data/
    #    -> data/storms.csv in the exact columns road-flood-risk reads (storm,start,peak_fc,total_fc,tide_fc) and storms_full.csv
    python -m pytest tests -q      # 17 tests

## Status: built and tested against mock servers; NEVER RUN LIVE
- The build environment could not reach api.weather.gov or NOAA. The mock NWS server follows the response shapes I remember (listed at the top of `src/nws.py`), so the tests prove the code is consistent with my assumption, not with the real API. **Expect to fix small parser issues on the first live run.** Unit and shape mismatches raise clear errors instead of producing wrong numbers.
- `src/noaa.py` is the tide client from `flood-nowcast` (also untested live). Default tide station is Sewells Point (8638610), the one in your source list.
- Not built: **MRMS** radar rainfall (large gridded files; the next step for per-watershed rainfall) and **511 Virginia** (live conditions only, no history to train on).

## What it guarantees (each tested)
- Forecast features come only from a snapshot **fetched at least `--lead-hours` (default 6) before the storm began**, and not older than 36 h before that. Snapshots fetched later are ignored, even if they differ wildly.
- A snapshot must cover >= 80% of the storm window; otherwise the storm is excluded and counted (never filled with observed rain, which would leak the answer).
- Interval rain is prorated by overlap with the storm window. Hand-checked.
- Storm = run of hours with rain >= 0.10 in/h (or tide above `--tide-threshold-ft`), gaps under 12 h merged, total >= 0.5 in. These cutoffs are my defaults; set them with the city's own definition of a flood-relevant storm.

## Limits you should know
- **Timeline:** snapshots exist only from the day the collector starts. You need on the order of dozens of storms with forecasts before the road-flood model has anything to learn from, which likely means many months.
- **Observed rain:** many NWS stations report `precipitationLastHour` rarely or never. The collector says so when it gets none. Use `--rain-csv` with an hourly record (time = end of hour; `--rain-csv-tz` converts local time to UTC) from a source you trust, or pick a station that reports rain (`--station`; I did not verify station IDs).
- `tide_fc` is the predicted **astronomical** tide peak only. It contains no storm-surge forecast.
- QPF is NWS gridded model guidance at one point, not a measurement at a road segment.
- Still missing for a real road-flood model: **timestamped flood reports or road closures with locations** from Public Works, plus segment elevation/drainage data. This pipeline cannot produce those.
