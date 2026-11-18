# Chesapeake AI pilots (synthetic-data prototypes)

Five small projects from the "City of Chesapeake data scientist" brief. Each has a synthetic data generator, a baseline to beat,
tests (mutation-checked), and a README that says what is and is not shown. **None has been run on real city data.**

| Project | Question | One-line result (synthetic) |
|---|---|---|
| road-flood-risk | Which road segments should crews check before a storm? | Hotspot history already ~ceiling; model only matches it |
| service-request-triage | Right team first time, with hand-off when unsure? | 0.74 keyword rules vs 0.81 model; 0.96 on the 79% it is confident about |
| intersection-safety | Which intersections deserve review? | Empirical Bayes ~ raw count at 3 yrs, better at 1 yr; crash-rate ranking worst |
| permit-completeness | Which applications are missing documents? | 0% wrong "missing" claims, but 68% sent to staff |
| chesapeake-flood-data | Save NWS forecasts + NOAA tides so storms have pre-storm forecasts | Tested on mock servers only; never run live; feeds road-flood-risk |

Run any project: `cd <project> && pip install -r ../requirements.txt && python src/run.py && python -m pytest tests -q`

## Real data
From the build environment, no real data source was reachable (only GitHub, whose repos I checked held code but no datasets). So every result above is synthetic.
Each project now has `src/run_real.py` plus `src/real_data.py` that load real exports using a column-mapping JSON, report everything they drop or find suspicious,
and are tested by round-tripping the synthetic data through files (identical results through both paths).

| Project | Real data needed | Likely public? |
|---|---|---|
| road-flood-risk | storm list with pre-storm forecasts; segment elevation/drainage; reported flood incidents per storm | rainfall/tide yes; incident reports need the city |
| service-request-triage | tickets: text, final team, time, location | needs the city (records request) |
| intersection-safety | crash records, volumes, intersection attributes | state crash and traffic-count data are the likely sources |
| permit-completeness | applications' file names, staff outcome | needs the city |
