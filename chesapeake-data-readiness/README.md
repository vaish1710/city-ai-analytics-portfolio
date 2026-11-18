# Chesapeake City Data: AI-Readiness Audit

Built for the **Enterprise AI & Analytics Program Manager** role (City of Chesapeake, Department of Information Technology).

## Why this project
Before a city funds an AI or analytics project, someone has to answer: *is the data good enough?* The cover letter promises
"assessing data quality, and defining clear measures of success." This tool does that for every dataset on the city's public
ArcGIS server and turns it into a score that feeds the use-case scorecard's **Data readiness** column.

It is also a simple reliability monitor: every run is appended to `outputs/run_history.csv`, and the report lists record-count
changes since the last run. Run it weekly and you can see whether a dataset has gone stale or silently shrunk.

## The AI step: drafting data documentation, with guardrails
The audit most often flags "no description" and fields with no real alias. `--ai-docs` asks a language model to draft that
documentation for each dataset, under three rules a public-sector program can defend:
1. **Privacy.** The model sees schema and summary statistics only (field names, types, share empty, scores). Never a record value. A test enforces this.
2. **Grounding.** Every number in the draft must already appear in the facts given to the model. Anything else is flagged "ungrounded" in the file and the log.
3. **Human review.** Each draft is stamped "AI-DRAFT. NOT APPROVED" with reviewer, date and decision boxes, and logged in `outputs/ai_review_log.csv` as "pending human review".
   Field meanings are guesses until the data owner confirms them.

This is the pattern the role asks for: use AI where it saves staff time, measure it, and keep a person accountable.
```
pip install anthropic                    # plus set ANTHROPIC_API_KEY
python src/audit.py --ai-docs            # drafts go to outputs/ai_docs/
```
**Not yet run with a live model.** The tests use a stand-in model. The first live run may need prompt adjustments; keep a person reading every draft.
The grounding check catches invented numbers, not invented meanings or wrong facts stated in words.

## What it measures (per dataset, 0-100)
| Component | Weight* | How |
|---|---|---|
| Completeness | 35% | Share of non-system fields that are filled in |
| Key uniqueness | 20% | Uniqueness of the best identifier-like field |
| Documentation | 15% | Fields with a real alias, and a layer description |
| Freshness | 20% | 1.0 if edited today, falling to 0 at one year; 0.5 if no edit date is published |
| Spatial | 10% | Share of sampled features that have geometry |

\*The weights are my assumption, not a standard. They live in `src/readiness.py`; change them and record why.
The `readiness_1_to_5` column is the same score on a 1-5 scale, ready to paste into the scorecard.

## Run it (needs internet; on your own computer)
```
pip install requests pandas
python src/audit.py                      # every layer in Chesapeake's OpenData service
python src/audit.py --layers 8 15        # just city facilities and parcels
```
Outputs in `outputs/`: `REPORT.md` (read this first), `dataset_readiness.csv`, `field_profile.csv`, `run_history.csv`.
To monitor reliability, schedule the command weekly (Windows Task Scheduler or cron) and keep the outputs folder.

## Tests
```
python -m unittest discover -s tests -v
```
Ten tests run the real client against a local fake ArcGIS server, with expected scores worked out by hand
(94.5 / 45.9 / 82.5). They also cover the AI guardrails: no record values in prompts, invented numbers caught, banner and
review log written. I broke a weight and made the prompt leak a value on purpose; both times the tests failed as they should.

## What is verified and what is not
- **Verified (data.gov records, Oct 2026):** the City of Chesapeake publishes Parcels (layer 15) and City Facilities (layer 8) at
  `gis.cityofchesapeake.net/mapping/rest/services/OpenData/OpenData/MapServer`, last updated 8 Oct 2026.
- **Not verified:** I could not reach that server from the build environment, so the **audit has never run against the live city
  data**. The service's layer list, field names and edit dates are unknown to me. Expect to fix small things on the first run
  (for example a layer that refuses a query type: the client falls back to the first records and says so in the report).
- The job posting page blocked automated readers, so this is built from your cover letter and the job title, not the posting text.

## Limits to state out loud
- Sample-based: completeness, key uniqueness and geometry use a random sample of up to 2,000 records. Duplicates elsewhere in a large layer can be missed.
- A high score means the data is **tidy**, not **correct** or **unbiased**. It does not replace a fairness review (for example, 311 or claims data can under-count some neighborhoods).
- **License:** the Parcels record lists CC BY-NC-ND 4.0 (non-commercial, no derivatives). Publishing a derived dataset or
  analysis may be restricted. Check each dataset's license, and ask the city if unsure, before putting results on a public repository.

## How it connects to the role
- **Intake gate:** a use case cannot score above "Plan" in the scorecard until its data scores at least 3 of 5 here (a rule you can propose, not an existing policy).
- **Measures of success:** track the share of city datasets scoring 4+ over time as a program KPI.
- **Training:** the report is a plain-language teaching tool for departments about why documentation and unique IDs matter.
