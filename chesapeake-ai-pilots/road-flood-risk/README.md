# Road-segment flood risk: which segments should Public Works check first?

Ranks road segments before a storm (forecast rain, elevation, drainage distance, tide, past incidents), with a plain-language
reason for each flag. Tested on **whole storms the model never saw**, against the baselines a city would use today.

    pip install pandas numpy scikit-learn matplotlib pytest
    python src/run.py        # ~1 min; writes outputs/summary.csv, recall.png, example_storm_flags.csv
    python -m pytest tests -q

## Result on synthetic data (10 simulated cities per row; last 30% of storms held out; crews check the top 8 of 60 segments)
| History | Ceiling* | Hotspot history | Model (logistic) | Model (boosted trees) | Lowest elevation |
|---|---|---|---|---|---|
| 80 storms | 0.376 | 0.357 | 0.363 | 0.333 | 0.236 |
| 200 storms | 0.339 | 0.328 | 0.327 | 0.300 | 0.204 |

Cells are the mean share of reported incidents caught; city-to-city ranges are wide (see outputs/summary.csv).
*Ceiling = ranking by the simulator's true flood probability. Not achievable in practice.

**Honest reading:** in this simulator the "known hotspots" baseline is already within ~0.02 of the ceiling, so a model has
almost no room to beat it. I did not tune the simulator until the model won. What this shows is the method: always compare
to the hotspot baseline and a ceiling. The model's likely value is where history is thin (new roads, changed drainage) and in
explaining flags. The simulator has a permanent hidden "local" effect that history reveals fully; real cities may differ in
either direction. Real data is the only way to know.

## Design choices
- Label is a **reported** incident, not true flooding (some floods are never reported). The simulator reports 70% of them.
- "Past incidents" is counted from earlier storms only (tested by corrupting the present and future), and fed to the model as a rate. The raw count grows over time and made the model worse on later storms (train mean 1.2, test mean 2.7).
- Split by storm, chronologically, never by row.
- Flags show reasons from simple thresholds (heavy rain, low elevation, past incidents, tide). They are descriptions, not a model explanation.

## To run on real data (needs the city)
Timestamped, geocoded flooding/closure/311 records per storm; hourly rainfall (gauge or NWS); road-segment elevation and
drainage layers; tide gauge series (see the flood-nowcast project). Swap `make_data` for a loader with the same columns.

## Limits
Synthetic only. Forecast error is modeled as simple noise. No spatial correlation between neighboring segments. Pooled recall
is dominated by a few large storms. No cost model for crew time.

## Running on REAL data (loader built and tested; not yet run on real records)
    python src/run_real.py --storms storms.csv --segments segments.csv --incidents incidents.csv --mapping data/mapping.json
Copy `data/mapping.example.json` to `data/mapping.json` and put YOUR column names on the right. Three files: storms (id, start time,
forecast peak/total rain, forecast tide), segments (id, elevation, distance to drain, tidal flag), incidents (storm id, segment id; one row per reported flood).
- **Use forecast values available before each storm, not observed rainfall.** Observed rain makes results look far better than the tool could do.
- The run prints what it dropped, warns on too few storms/incidents, and gives a bootstrap interval for "model minus hotspot baseline". If the interval spans 0, there is no demonstrated gain.
- The loader is tested by exporting the synthetic data to files and checking it gives identical results through the real-data path.
