# Road flood risk: which roads should crews check first?

A held-out-storm study of whether a flood-risk model beats a "check the roads that flooded before" list, built around the City of Chesapeake, Virginia.

**All results use a simulated city.** No real Chesapeake road-level records or live tide/forecast data are used. See `road-flood-risk/README.md` for the findings table, the limits, and what a real pilot needs.

    cd road-flood-risk
    pip install numpy pandas scikit-learn scipy joblib matplotlib pytest
    python -m pytest -q tests      # 27 tests
    python src/study.py            # all scenarios -> outputs/study/
    # open demo/index.html in a browser for the interactive demo

Headline result: with 24 storms of history or fewer the model gains 3 to 12 recall points at 8 segments checked; with full history it ties the plain hotspot count or trails it by about one point.
