# Who Pays for Chesapeake's Water and Sewer Rate Increases?

## The issue
In July 2025 Chesapeake's council approved multi-year utility rate increases 8-1. The vote summary reports
**4.9% a year for combined water and sewer in 2026 and 2027, projected at 4.9% a year through 2032**, tied to about
$78 million of rate-funded infrastructure need. Residents said they felt "ambushed" and asked for more transparency
([vote summary](https://cpo.soar.com/articles/6510454/Chesapeake-City-Independent-City/Virginia/Chesapeake-council-approves-multiyear-utility-rate-increases-after-public-hearing)).
That summary is AI-generated and its dollar figures contradict each other ($3 a month vs. $3.10 every two months),
so only the percentage is used here. Check the meeting video for exact figures.

## Question
The same revenue can be raised in different ways. Who bears the cost under each?

| Option | What changes | Revenue |
|---|---|---|
| A. As approved | Every charge rises 4.9% a year | baseline |
| B. Lifeline blocks | Fixed charge and first usage block rise at half the pace; upper block solved to compensate | identical to A |
| C. Assistance credit | Same as A, plus a 25% credit for the lowest-income group, paid by a uniform surcharge on others | identical to A |

Run: `python src/analysis.py`. It asserts revenue neutrality and the compounding arithmetic (4.9% for 7 years = x1.398)
before writing `outputs/burden_results.csv` and `outputs/burden_by_income_2032.png`.

## Result on synthetic data (method check only, not Chesapeake)
Mean bill as a share of income in 2032, versus 2025:

| Income group | 2025 | A as approved | B lifeline | C credit |
|---|---|---|---|---|
| Q1 lowest | 0.98% | 1.37% | 1.30% | 1.02% |
| Q2 | 0.57% | 0.79% | 0.78% | 0.85% |
| Q3 | 0.41% | 0.57% | 0.57% | 0.61% |
| Q4 highest | 0.25% | 0.35% | 0.36% | 0.38% |

Reading: a flat percentage increase hits lower-income households hardest as a share of income. The lifeline-block
design barely helps, because usage differs little by income, so a usage-based tier is a weak way to target relief.
A direct credit is far more effective. About 12% of lowest-income households exceed 2% of income under A, versus 4% under C.
These conclusions depend on the placeholders below; they show what the tool can answer, not what is true in Chesapeake.

## Replace before using on real decisions
1. **Tariff.** `BASE_FIXED` and `BASE_PER_KGAL` are placeholders, calibrated to one unreliable staff figure. Use the
   city's actual rate schedule. If bills are much larger, burdens scale up proportionally.
2. **Households.** Incomes and usage are synthetic. Use ACS tract median income (table B19013) and the city's
   anonymized billing distribution.
3. **Credit targeting.** Option C assumes you can identify the lowest-income group. Real programs rely on enrollment,
   and take-up is usually well below 100%. Model it before promising results.
4. **Thresholds.** The 2% and 4.5% burden screening levels are commonly cited but should be verified, and some
   researchers criticize income-share thresholds. They are configurable in `analysis.py`.
5. **Scope.** Only water and sewer. Stormwater and trash were not in the source and are excluded.
