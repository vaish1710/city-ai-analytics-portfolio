"""Service-request triage: suggest the owning team, hand uncertain tickets to a person, flag likely duplicates.
Synthetic tickets only. Fictional departments. Labels carry 5% noise, like real final-assignment records."""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import linear_kernel

CATS = ["streets", "traffic", "waste", "stormwater", "parks_other"]
PHRASES = {
    "streets": ["huge pothole", "pothole that damaged my tire", "road is cracked and falling apart", "sinkhole forming in the road",
                "crumbling asphalt", "damaged pavement", "deep hole in the street"],
    "traffic": ["traffic light is out", "signal stuck on red", "stop sign knocked down", "signal flashing red all day",
                "crosswalk signal not working", "left turn arrow broken"],
    "waste": ["trash was not picked up", "missed garbage collection", "recycling bin not emptied", "bulk pickup missed",
              "garbage truck skipped our street", "trash can was never collected"],
    "stormwater": ["storm drain is clogged", "ditch is overflowing", "standing water that will not drain", "drain blocked with leaves",
                   "flooded street after the rain", "ditch behind my house is full"],
    "parks_other": ["tree limb fell in the park", "barking dog all night", "playground equipment broken", "overgrown lot",
                    "abandoned vehicle", "noise complaint next door"],
}
CONFUSABLE = [("water across the road", "streets", "stormwater"), ("branch blocking the road", "streets", "parks_other"),
              ("trash piled by the storm drain", "waste", "stormwater"), ("light out near the school crossing", "traffic", "parks_other")]
VAGUE = ["something is wrong on my street please fix", "need help with an issue near my house", "problem in my neighborhood",
         "please send someone to look at this"]
OPEN = ["", "Hi, ", "Please help: ", "Reporting a problem. ", "Again, "]
PLACE = ["on Main St", "at the corner of Elm and 5th", "near the school", "in front of my house", "by the shopping center", "on Cedar Rd", ""]
TAIL = ["", " It has been like this for days.", " Please fix soon.", " Dangerous for kids.", " Thanks."]


def typo(text, rng):
    words = text.split()
    cand = [i for i, w in enumerate(words) if len(w) > 4]
    if not cand:
        return text
    i = int(rng.choice(cand)); w = words[i]; j = int(rng.integers(1, len(w) - 2))
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def _text(cat, rng, vague=False):
    if vague:
        core = str(rng.choice(VAGUE))
    elif rng.random() < 0.12:
        c = CONFUSABLE[int(rng.integers(len(CONFUSABLE)))]
        core = c[0]
    else:
        core = str(rng.choice(PHRASES[cat]))
    t = f"{rng.choice(OPEN)}{core} {rng.choice(PLACE)}.{rng.choice(TAIL)}".replace("  ", " ").strip()
    return typo(t, rng) if rng.random() < 0.15 else t


def make_tickets(n=4000, seed=5, dup_frac=0.12, label_noise=0.05):
    rng = np.random.default_rng(seed)
    centers = np.c_[rng.uniform(36.60, 36.90, 25), rng.uniform(-76.50, -76.20, 25)]
    rows = []
    for i in range(n):
        t0 = rng.uniform(0, 365 * 24)
        if rows and rng.random() < dup_frac:                        # a repeat report of an earlier incident
            o = rows[int(rng.integers(len(rows)))]
            if o["vague"]:
                o = rows[0]
            cat, lat, lon, inc, ts = o["category"], o["lat"] + rng.normal(0, 2e-4), o["lon"] + rng.normal(0, 2e-4), o["incident"], o["hours"] + rng.uniform(0, 48)
            vague = False
        else:
            c = centers[int(rng.integers(25))]
            vague = rng.random() < 0.10
            cat = str(rng.choice(CATS)); lat, lon = c[0] + rng.normal(0, 8e-4), c[1] + rng.normal(0, 8e-4); inc, ts = i, t0
        text = _text(cat, rng, vague)
        if not vague and rng.random() < 0.0: pass
        label = str(rng.choice(CATS)) if (rng.random() < label_noise or vague) else cat
        rows.append({"hours": ts, "lat": lat, "lon": lon, "text": text, "category": cat, "label": label, "vague": vague, "incident": inc})
    df = pd.DataFrame(rows).sort_values("hours").reset_index(drop=True)
    df["ticket"] = df.index
    return df


def chrono_split(df, a=0.60, b=0.75):
    n = len(df); i, j = int(n * a), int(n * b)
    return df.iloc[:i], df.iloc[i:j], df.iloc[j:]


KEYWORDS = [("streets", ["pothole", "pavement", "asphalt", "road"]), ("traffic", ["signal", "traffic light", "stop sign", "crosswalk"]),
            ("waste", ["trash", "garbage", "recycling", "pickup"]), ("stormwater", ["drain", "ditch", "flood"])]


def keyword_route(text):
    """Stand-in for 'first matching keyword' intake rules. The city's real routing accuracy is unknown."""
    t = text.lower()
    for cat, words in KEYWORDS:
        if any(w in t for w in words):
            return cat
    return "parks_other"


def fit_router(train):
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    X = vec.fit_transform(train.text)
    clf = LogisticRegression(C=5.0, max_iter=2000).fit(X, train.label)
    return vec, clf


def predict(vec, clf, texts):
    p = clf.predict_proba(vec.transform(texts))
    return clf.classes_[p.argmax(1)], p.max(1)


def choose_threshold(pmax, correct, target=0.95, min_n=20, grid=None):
    """Smallest confidence cutoff whose auto-routed tickets (on validation data) reach the target accuracy.
    Returns (cutoff, reached). If no cutoff reaches the target, returns the strictest one with reached=False."""
    grid = np.linspace(0.30, 0.95, 27) if grid is None else grid
    for t in grid:
        m = pmax >= t
        if m.sum() >= min_n and correct[m].mean() >= target:
            return float(t), True
    return float(grid[-1]), False


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def candidate_pairs(df, cats, vec, max_m=150, max_h=72, min_cos=None):
    """Pairs of tickets that look like the same incident: close in space and time, same predicted team, similar text."""
    lat, lon, h = df.lat.values, df.lon.values, df.hours.values
    X = vec.transform(df.text)
    cos = linear_kernel(X)
    cats = np.asarray(cats)
    iu, ju = np.triu_indices(len(df), k=1)
    ok = (np.abs(h[iu] - h[ju]) <= max_h) & (cats[iu] == cats[ju])
    iu, ju = iu[ok], ju[ok]
    ok = haversine_m(lat[iu], lon[iu], lat[ju], lon[ju]) <= max_m
    iu, ju = iu[ok], ju[ok]
    if min_cos is not None:
        ok = cos[iu, ju] >= min_cos
        iu, ju = iu[ok], ju[ok]
    return set(zip(df.ticket.values[iu].tolist(), df.ticket.values[ju].tolist()))


def true_pairs(df):
    pairs = set()
    for _, g in df.groupby("incident"):
        t = sorted(g.ticket.tolist())
        pairs |= {(a, b) for k, a in enumerate(t) for b in t[k + 1:]}
    return pairs


def pr(found, truth):
    tp = len(found & truth)
    return (tp / len(found) if found else float("nan")), (tp / len(truth) if truth else float("nan"))
