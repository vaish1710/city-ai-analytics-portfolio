# AI & analytics projects for city government (Chesapeake, VA focus)

Portfolio projects that apply data science and AI to city problems: flooding, traffic safety, 311 service requests, permits,
utility rates, and AI governance. Each folder has its own README, tests, and sample outputs.

## Read this first: what is and is not real
**Every result in this repository comes from synthetic or fictional data I generated.** None of the projects has been run on real
City of Chesapeake data, and the code that talks to live services (NOAA, NWS, city ArcGIS, FEMA) has only been tested against mock servers.
Each README says what its results do and do not show. Where a model failed to beat a simple baseline, the README says so.
The "real data" loaders (`run_real.py`) exist so that real exports can be dropped in; they are tested by round-tripping synthetic data.

## Projects
| Folder | What it does | Status |
|---|---|---|
| `chesapeake-ai-pilots/road-flood-risk` | Ranks road segments before a storm; tested on held-out storms | Synthetic; model only matches the hotspot baseline |
| `chesapeake-ai-pilots/service-request-triage` | Routes 311 tickets, hands uncertain ones to staff, flags duplicates | Synthetic; loader for real tickets |
| `chesapeake-ai-pilots/intersection-safety` | Empirical Bayes screening of intersections | Synthetic; loader for real crash data |
| `chesapeake-ai-pilots/permit-completeness` | Flags missing permit documents; routes uncertainty to staff | Synthetic; fictional checklist |
| `chesapeake-ai-pilots/chesapeake-flood-data` | Collects NWS forecast snapshots + NOAA tides into a storm table | Mock-server tests only; never run live |
| `flood-nowcast` | Short-horizon storm-surge/water-level nowcast from NOAA tide gauges | Synthetic results; VIMS already publishes forecasts for this area |
| `city-docs-assistant` | Question answering over city documents with citations, refusal and an evaluation harness | 5 fictional documents; live LLM not run |
| `chesapeake-data-readiness` | Audits a city open-data service for AI readiness | Mock-tested; never run live |
| `norfolk-flood-prioritization` | Home-elevation prioritization by benefit-cost and equity | Synthetic parcels |
| `utility-rate-impact` | Bill burden by income under rate scenarios | Synthetic households; placeholder rates |
| `ai-use-case-scorecard` | Weighted scoring workbook for AI use-case intake | Draft use cases and weights are illustrative |

## Running
Each Python project: `cd <folder> && pip install pandas numpy scikit-learn matplotlib pytest requests && python -m pytest tests -q`
(see the folder's README for the run script and any extra dependencies, e.g. `pypdf`, `openpyxl`, `anthropic`).

## Not legal, financial, or official guidance
Prototypes for demonstration. Nothing here is an official forecast, warning, rate recommendation, or permit decision.
