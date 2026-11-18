# Intersection safety screening: which intersections deserve an engineering review?

Ranks intersections for review using **Empirical Bayes** (the Highway Safety Manual approach): blend each site's own crash
history with what a model expects for a site like it. Ranks on 2019-21 crashes only, then judges on 2022-23 crashes.

    pip install pandas numpy scikit-learn matplotlib pytest
    python src/run.py          # ~10 s; outputs/summary.csv, results_by_seed.csv, ranking_comparison.png
    python -m pytest tests -q  # 7 tests

## Results on synthetic data (300 intersections x 30 simulated cities; flag the top 10%)
| Ranking | Share of 2022-23 crashes at flagged sites (random = 0.10) |
|---|---|
| Empirical Bayes | 0.283 |
| Raw crash count | 0.280 |
| Model only (no history) | 0.222 |
| Crashes per vehicle | 0.193 |

- **Empirical Bayes vs raw count:** with 3 years of history the difference (+0.003, std error 0.002) is **not distinguishable from noise**. With only 1 year of history it is small but consistent (+0.012, std error 0.003; better in 26 of 30 cities). So its advantage grows as history gets shorter or sparser, which is what the theory predicts, but it is modest here.
- **Crashes per vehicle is the worst ranking.** A low-volume site with one crash gets a huge rate. This is the "account for exposure" trap: dividing by volume is not the same as modeling exposure.
- **Regression to the mean:** the 10% of sites with the most 2019-21 crashes averaged 4.60 crashes/yr then and 3.90 in 2022-23 with nothing done to them. A before/after study of a fix at such sites will overstate its benefit unless it accounts for this.

## How it works
1. Fit a safety performance function (Poisson regression on volume, signal, legs, speed) to expected crashes per year.
2. Estimate overdispersion k (method of moments, Var = mu + k*mu^2).
3. Expected crashes = w*predicted + (1-w)*observed, with w = 1/(1+k*mu). Hand-checked in tests.
4. Tests confirm the ranking cannot see test-year crashes.

## Real data needed
VDOT crash records (ideally severity-coded) per intersection by year, entering volumes (AADT) for each approach, and intersection
attributes (control type, legs, speed, lighting). Swap `make_sites` for a loader with the same columns. VDOT's crash dashboard is the
suggested public starting point (not checked here).

## Limits
- Synthetic: the simulator draws crashes from exactly the Poisson model the SPF assumes, so the SPF looks better than it will on real data.
- All crashes are treated as equal. Real screening should weight severity (fatal/serious injury) and consider crash type.
- Ranking is a screening step only. A flagged site needs an engineer's review; it is not evidence of a design fault.
- No spatial correlation, no changes in site over time, no reporting-threshold differences.

## Running on REAL data (loader built and tested; not yet run on real crashes)
    python src/run_real.py --crashes crashes.csv --sites sites.csv --mapping data/mapping.json
Copy `data/mapping.example.json` to `data/mapping.json`. Crash file: one row per crash (site id, date, optional severity). Sites file: site id, entering volume (AADT), signal, number of legs, speed limit.
- Optional `severity_keep` in the mapping filters to e.g. fatal/serious-injury crashes only.
- The loader warns about missing/incomplete years, unreadable dates, crashes at unknown sites, and sites with zero volume.
- Gives a bootstrap interval for "Empirical Bayes minus raw count"; if it spans 0 the data cannot tell them apart.
- Likely public sources (not verified from here): state crash records and state traffic-count (AADT) data, plus the city's signal inventory.
