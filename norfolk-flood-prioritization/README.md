# Norfolk: Which Homes Should the Flood Program Fix First?

## The problem (as of November 2026)
Norfolk's coastal flood defense project has seen its estimated cost rise from $2.6B to $6.1B,
and a councilman said the city can no longer afford even its 35% match
([WTKR](https://www.wtkr.com/news/in-the-community/norfolk/estimated-cost-of-norfolk-flood-project-more-than-doubles-to-6-1-billion)).
Reported completion dates differ by source (2037 vs 2040). A separate *nonstructural* program
aims to raise or flood-proof about 1,000 homes, starting with 62 in Willoughby/Ocean View,
backed by a $25M state grant. With limited money, **which homes go first?**

## Two modes
| | Synthetic demo | Real data |
|---|---|---|
| Command | `python src/run.py` | `python src/run.py --data data/parcels_real.csv` |
| Risk estimate | Gradient boosting trained on synthetic loss labels | **Physical depth-damage EAL** (`src/physical.py`), no ML |
| Why | Shows the method end to end | Real parcels have **no per-home loss labels**, so there is nothing to train on |

Both then run the same benefit-cost + equity allocation (`src/prioritize.py`) and write `outputs/`.
Income is never a risk input; it only weights priority and audits who gets funded.

## Getting the real data (run on your own computer)
The scripts could not be run against the live servers when written, because the build
environment had no access to them. Treat the first run as a test.
```
pip install requests pandas numpy scikit-learn matplotlib geopandas shapely pyproj
python src/fetch_real_data.py discover      # lists Norfolk GIS layers and their field names
python src/fetch_real_data.py build \
   --property-layer <URL of the Property_Information layer with assessed value> \
   --prop-key <field matching parcel TAX_ACCT or GPIN> \
   --map building_value=<F>,sqft=<F>,year_built=<F>[,foundation=<F>]
python src/run.py --data data/parcels_real.csv --budget 25000000
```
What is verified vs. not:
- **Verified:** Norfolk's parcel layer (`gisshare.norfolk.gov/.../OpenData/Parcels/MapServer/1`) has only IDs and
  geometry (`TAX_ACCT`, `GPIN`), max 2,000 records per request. Value, size and year built are **not** in it.
- **Not verified:** the FEMA flood-zone layer id used in `fetch_real_data.py`, and the field names of the
  Property_Information layers. `discover` shows the real ones; edit the script if they differ.
- Elevation comes from USGS 3DEP one point at a time, which is slow; the script caps parcels at `--limit 3000`.
  For the whole city, download a 3DEP DEM and sample it instead.

## What to fix before trusting any real result
1. **Placeholders in `src/physical.py`.** The depth-damage curve, first-floor heights, and the stage offsets
   for the 10-yr and 500-yr events are illustrative, not findings. In my test, these made very few homes pass the
   benefit-cost > 1 filter, so results are very sensitive to them. Use FEMA/HAZUS or USACE curves.
2. **Missing BFE.** Flood-zone parcels with no published base flood elevation get BFE = ground + 1 ft, which
   makes their risk independent of elevation (tested: correlation with elevation 0.02, vs -0.69 with a real BFE).
   `run.py` now prints how many parcels are affected. Treat those ranks as unreliable.
3. **Parcels outside A/V zones get zero loss**, including shaded-X areas. Tidal and rain flooding there is ignored.
4. **Frequent-event tail assumption.** The integral assumes zero damage at about the 5-year event.
5. **Reporting bias.** A peer-reviewed study of Norfolk 311 flood reports
   ([MDPI Water, 2025](https://www.mdpi.com/2073-4441/17/21/3178)) found reporting differed by tract demographics
   (for example, tracts with more Hispanic residents reported less, possibly due to language barriers). So 311 counts
   and insured claims under-state flooding in some neighborhoods. Do not use them as a loss label or a ranking input without
   correcting for this.
6. **Cost model** (`elevation_cost`) and the 85% loss-reduction figure are assumptions; replace with program bid data.
7. Ranks are decision support. A named city staff member should approve funding, and the method should be published.

## What the synthetic demo shows
Method check only. Nothing in `outputs/` describes real Norfolk. On synthetic data the model recovers the hidden truth
(r = 0.97), and adding an equity weight roughly doubles funding to the lowest-income quartile for about a 2% drop in
net benefit.

## Files
`src/generate_data.py` synthetic parcels · `src/model.py` ML risk model · `src/physical.py` depth-damage EAL ·
`src/prioritize.py` benefit-cost and allocation · `src/fetch_real_data.py` real-data download · `src/run.py` entry point.
