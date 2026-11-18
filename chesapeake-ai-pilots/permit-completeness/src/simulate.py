"""Synthetic submissions with known truth, including messy, scanned, ambiguous and misleading file names."""
import numpy as np
from permit import required_docs

NAMES = {"site_plan": ["Site Plan.pdf", "site_plan_final.pdf", "plot plan v2.pdf", "Survey.pdf"],
         "floor_plan": ["Floor Plan.pdf", "floor-plan-v3.pdf"],
         "framing_detail": ["Deck Framing Detail.pdf", "joist layout.pdf"],
         "structural_drawings": ["Structural Drawings.pdf", "structural set.pdf"],
         "owner_authorization": ["Owner Authorization.pdf", "authorization letter.pdf"],
         "engineer_letter": ["Engineer Letter.pdf", "stamped engineer letter.pdf"],
         "elevation_certificate": ["Elevation Certificate.pdf", "elevation cert 2024.pdf"],
         "contractor_license": ["Contractor License.pdf", "license copy.pdf"],
         "product_specification": ["Shingle Product Specification.pdf", "spec sheet.pdf"]}
UNKNOWN = ["scan0012.pdf", "IMG_2231.pdf", "attachment.pdf", "doc_final.pdf", "sitepl_final.pdf", "plan1.pdf"]
AMBIG = {"framing_detail": ["framing plan.pdf"], "structural_drawings": ["framing plan.pdf"]}
EXTRA = ["photos_front_yard.pdf", "application_form.pdf", "photos back.pdf"]
MISLEADING = ["site plan photos.pdf"]            # a photo file whose name looks like a site plan


def make_submissions(rules, n=2000, seed=9, p_attach=0.75):
    rng = np.random.default_rng(seed)
    kinds = list(rules["permits"])
    out = []
    for _ in range(n):
        pt = kinds[int(rng.integers(len(kinds)))]
        fields = {"height_ft": float(rng.choice([4, 6, 10, 12])), "in_flood_zone": bool(rng.random() < 0.3), "uses_contractor": bool(rng.random() < 0.5)}
        req = required_docs(pt, fields, rules)
        files, truth_present = [], set()
        for t in req:
            if rng.random() > p_attach:
                continue
            truth_present.add(t)
            r = rng.random()
            if r < 0.60: name = NAMES[t][0]
            elif r < 0.78: name = NAMES[t][int(rng.integers(len(NAMES[t])))]
            elif r < 0.92: name = str(rng.choice(UNKNOWN))
            elif t in AMBIG: name = AMBIG[t][0]
            else: name = str(rng.choice(UNKNOWN))
            files.append(name)
        for _ in range(int(rng.integers(0, 3))):
            files.append(str(rng.choice(MISLEADING)) if rng.random() < 0.15 else str(rng.choice(EXTRA)))
        out.append({"permit_type": pt, "fields": fields, "files": files, "truly_missing": sorted(set(req) - truth_present)})
    return out
