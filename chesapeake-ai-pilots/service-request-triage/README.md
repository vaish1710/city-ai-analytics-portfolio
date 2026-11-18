# Service-request triage: route to the right team, hand uncertain tickets to a person, flag likely duplicates

    pip install pandas numpy scikit-learn pytest
    python src/run.py          # results + outputs/coverage_vs_accuracy.csv, example_routing.csv
    python -m pytest tests -q  # 8 tests

## What it does
1. **Routes** a free-text report (pothole, signal, missed trash, drainage, other) to a team with TF-IDF + logistic regression.
2. **Abstains**: below a confidence cutoff, the ticket goes to a person. The cutoff is chosen on validation data, never test data.
3. **Flags possible duplicates** (same predicted team, within 150 m and 72 h) for staff to review. It never merges anything.

## Results on synthetic tickets (4,000 tickets; train on oldest 60%, validate 15%, test on newest 25%)
| | |
|---|---|
| Keyword rules, correct first assignment | 0.74 |
| Model routing everything | 0.81 |
| Model only where confidence ≥ 0.5 (79% of tickets) | 0.96 |
| Model only where confidence ≥ 0.9 (66% of tickets) | 0.98 |
| Vague tickets ("something is wrong on my street") sent to a person | 100% (clear tickets: 54%) |

Full trade-off: `outputs/coverage_vs_accuracy.csv`. How many tickets can be auto-routed is a policy choice about how many wrong routings are acceptable.

## Things the results do not show (read before quoting numbers)
- **Tickets are templated by me.** Real tickets are messier; 0.81 / 0.96 will be lower on real data. Treat the *gap to the keyword baseline* and the *method*, not the absolute numbers, as the point.
- **The keyword baseline is my stand-in for current practice.** The city's real first-assignment accuracy is unknown; measuring it from reassignment history is step one of a real pilot.
- **The 95% accuracy target was not reached on validation** (label noise + vague tickets cap accuracy near 94%). The code reports this and falls back to the strictest cutoff, rather than hiding it. Picking a cutoff from the table above is the better use of the tool.
- **Duplicate flags: precision 0.39, recall 0.80** (219 pairs flagged for 106 true pairs). Adding text similarity lowered recall to 0.24, but that is a simulator artifact: repeat reports get independently generated text, so true duplicates share almost no words (median similarity 0.03). On real tickets, where people describe the same problem, text similarity may well help. This needs real data to decide.
- Labels have 5% noise and some tickets are unlearnable by design, so ~94-97% is roughly the ceiling here.

## Real data needed
Historical tickets with: free text, timestamp, location, initial team, and final team (the label should be the *final* team). Hold out the newest months for testing. Measure reassignment rate and time-to-assignment before and after.

## Limits
Synthetic. Five fictional teams. No attachments/photos, no non-English reports, no fairness check across neighborhoods (a real pilot should compare accuracy by area and language).

## Running on REAL data (loader built and tested; not yet run on real tickets)
    python src/run_real.py --tickets tickets.csv --mapping data/mapping.json
Copy `data/mapping.example.json` to `data/mapping.json`. Needed columns: ticket text, **final** team, created time, latitude, longitude; optional incident id for duplicates.
- Put your **current intake rules** in `keyword_rules` to get a real comparison with current practice. Otherwise the only baseline is the majority class.
- The label must be the team that finally handled the ticket, not the first assignment.
- Sorted by time before splitting (tested with a shuffled file). Outputs contain no ticket text.
- **Privacy:** tickets contain names, phones and addresses. Scrub before sharing results or using any outside service.
