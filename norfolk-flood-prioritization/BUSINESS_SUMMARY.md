# Flood Mitigation Prioritization: One-Page Summary and Charter

**Status:** working method on synthetic data. Real-data download script written but not yet run against live servers.
**Cities:** built first for Norfolk (its parcel layer was verified). Chesapeake is the target for the job application; its data layers are not yet confirmed.

## Decision this supports
With a fixed mitigation budget, **which homes should be elevated or flood-proofed first?**
Context: Norfolk plans to help about 1,000 homes with a $25M state grant, starting with 62
([WTKR](https://www.wtkr.com/news/in-the-community/norfolk/estimated-cost-of-norfolk-flood-project-more-than-doubles-to-6-1-billion)).
Chesapeake has about 12% of its area in a special flood hazard area
([VCN](https://vcnva.org/wp-content/uploads/2022/02/CFPF-Chesapeake.pdf)).

## Recommendation (method, not yet a finding)
Rank homes by **loss avoided per dollar**, then apply a modest equity weight and publish the ranking and method.
Compare against two simpler rules so the benefit is measurable.

## Evidence so far (SYNTHETIC data, $25M budget; method check only)
| Rule | Homes funded | Net benefit | Share to lowest-income quartile |
|---|---|---|---|
| Highest loss first (naive baseline) | 88 | $121M | 23% |
| Benefit-cost | 116 | $136M | 16% |
| Benefit-cost + equity weight | 114 | $133M | 30% |

On synthetic data, benefit-cost funds about 30% more homes than the naive rule for the same money, and the equity weight
nearly doubles funding to the lowest-income quartile for about a 2% drop in net benefit. **Whether this holds for real homes is
unknown.** It is the question the real data answers.

## Success measures (agree these before building on real data)
- Loss avoided per $1M spent, versus the naive baseline.
- Share of funded homes in the lowest-income quartile, versus the share of at-risk homes there.
- Share of top-ranked homes confirmed at-risk by field staff (aim to test on a sample of 20-30).
- Staff hours saved versus a manual ranking (ask Public Works for their current process time).

## What is needed
| Need | Owner | Status |
|---|---|---|
| Parcel geometry | City GIS | Norfolk verified; Chesapeake unconfirmed |
| Assessed value, size, foundation | Assessor / GIS | Layer and fields not yet identified |
| FEMA flood zones and base flood elevations | FEMA NFHL | Layer id not verified |
| Ground elevation | USGS 3DEP | Script uses point lookups; bulk DEM better |
| Depth-damage curve, elevation cost | Engineering / program bids | **Placeholders in code: replace** |

## Risks and responsible-AI controls
- **Missing base flood elevations** make some risk scores unreliable; the run prints how many parcels are affected.
- **Biased data.** 311 flood reports and insurance claims under-count some neighborhoods (a 2025 study of Norfolk 311 data found reporting differed by tract demographics: [MDPI](https://www.mdpi.com/2073-4441/17/21/3178)). They are not used as labels.
- **Income is never a model input**; it only sets an explicit, adjustable equity weight and audits outcomes.
- **Human decision.** The ranking is a recommendation. A named staff member approves funding and can override with a recorded reason.
- **Transparency.** Publish the method, assumptions and the ranked list's logic, so residents can see why a home was or was not chosen.
- **Privacy.** Uses parcel records; any claims data is aggregated to census-tract level.

## Plan
1. Run `fetch_real_data.py discover`, identify the property layer, fill the column mapping (about 1 day).
2. Replace placeholder curve and costs with engineering inputs (about 1 week, needs an engineer's review).
3. Run on real data, check a sample of results against field knowledge (about 1 week).
4. Present the one-page summary and ranked list to the program owner; log decisions.
