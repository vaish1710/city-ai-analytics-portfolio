# Permit completeness assistant: flag missing documents before staff spend time on an application

Checks an application's uploaded files against a permit-type checklist and tells the applicant (or staff) what looks missing.
**Staff keep every decision**; the tool only drafts a message and routes uncertain cases to a person.

    pip install pytest
    python src/run.py          # outputs/results.csv, example_messages.txt
    python -m pytest tests -q  # 9 tests

## FICTIONAL CHECKLISTS
`data/checklists.json` is an invented example (4 permit types). It is **not** the City of Chesapeake's requirements. Replace it
with the real list from the city's permit pages; the code reads whatever is in that file. I could not verify the real list.

## How it decides
- Each filename is matched to document types by keywords. 0 matches = unknown, 1 = recognized, 2+ = ambiguous (e.g. "framing plan.pdf").
- A required document is **confirmed** only by a file that matches that type alone.
- If it isn't confirmed, it is **missing** only when no unknown or ambiguous file could be it. Otherwise it is **uncertain** and the application goes to staff.
- Statuses: `complete`, `incomplete` (something is certainly missing), `staff_review`.

## Results on 2,000 synthetic submissions (1,201 truly incomplete)
| | Assistant | Baseline (no uncertainty handling) |
|---|---|---|
| "Missing" claims that were wrong | **0%** | 37% |
| Said "complete" when it was incomplete (share of incomplete) | 1.5% | 1.8% |
| Sent to staff review | 68% | n/a |

- **The 0% wrong claims come with a cost:** 68% of submissions are routed to a person, because the simulator gives ~22% of uploaded files unrecognizable names (scans, "IMG_2231.pdf"). Real applicants may be better or worse; the share of unrecognizable names is the number to measure first.
- **The remaining 1.5% false "complete" is a limit of filename checking.** A photo named "site plan photos.pdf" is accepted as the site plan (there is a test documenting this). Catching it needs reading file contents, which this project does not do.
- The simulator's file-naming habits are my assumptions, so the percentages describe the design, not real applicants.

## Where AI would fit (not built)
A model that reads the first page of each PDF to classify document type would cut the unknown/ambiguous share and catch misleading names.
It should be evaluated the same way: wrong "missing" claims and false "complete" outcomes against staff-labelled applications.

## Real data needed
Past applications with their uploaded files (or file names) and staff's recorded reason when an application was returned as incomplete.

## Limits
Synthetic; fictional checklist; filename-only; no check of document quality, signatures, or whether a plan meets code. Not legal or official guidance.

## Running on REAL data (loader built and tested; not yet run on real applications)
    python src/run_real.py --submissions subs.csv --mapping data/mapping.json --rules data/checklists.json
**First replace `data/checklists.json` with the real requirements** (the run warns while it is still the fictional example).
One row per application: permit type, uploaded file names separated by `;`, answers the checklist conditions use, and optionally `staff_missing` (what staff recorded as missing).
- Even without staff labels it reports the share of unrecognized file names and the 15 most common unrecognized names: add keywords for those first.
- With staff labels it reports wrong "missing" claims and false "complete" outcomes against staff.
