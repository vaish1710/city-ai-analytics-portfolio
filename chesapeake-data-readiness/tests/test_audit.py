import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import mock_server  # noqa: E402
from arcgis import Client  # noqa: E402
from audit import audit, write_outputs  # noqa: E402


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv, cls.url = mock_server.start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        mock_server.H.refuse_ids = False

    def run_audit(self, **kw):
        res, fields = audit(self.url, now=mock_server.NOW, **kw)
        return res.set_index("layer_id"), fields

    def test_scores_match_hand_calculation(self):
        # Hand-computed from the mock data and the documented weights:
        #  L0: 100*(.35*1 + .20*1 + .15*(.5*(1/3)+.5) + .20*(1-10/365) + .10*1)        = 94.5
        #  L1: 100*(.35*.8125 + .20*.5 + .15*0 + .20*0 + .10*.75)                     = 45.9
        #  L2: 100*(.35*1 + .20*1 + .15*.5 + .20*.5 + .10*1)                          = 82.5
        res, _ = self.run_audit()
        self.assertAlmostEqual(res.loc[0, "score"], 94.5, places=1)
        self.assertAlmostEqual(res.loc[1, "score"], 45.9, places=1)
        self.assertAlmostEqual(res.loc[2, "score"], 82.5, places=1)
        self.assertEqual(list(res["record_count"]), [50, 40, 30])

    def test_messy_layer_is_flagged_for_the_right_reasons(self):
        res, _ = self.run_audit()
        f = res.loc[1, "flags"]
        for expected in ("only 50.0% unique", "not edited for 800 days", "NOTES", "CITY", "25% of sampled features have no geometry"):
            self.assertIn(expected, f)
        self.assertNotIn("not edited", res.loc[0, "flags"])

    def test_system_fields_are_not_profiled(self):
        _, fields = self.run_audit()
        self.assertNotIn("OBJECTID", set(fields["field"]))

    def test_fallback_when_server_refuses_id_queries(self):
        mock_server.H.refuse_ids = True
        res, _ = self.run_audit()
        self.assertIn("FALLBACK", res.loc[0, "sampling"])
        self.assertGreater(res.loc[0, "score"], 80)  # geometry unknown -> credited 1.0, still scored

    def test_sample_smaller_than_layer_is_random_and_sized(self):
        recs, _, how = Client().sample(self.url + "/0", n=20, seed=1)
        self.assertEqual(len(recs), 20)
        self.assertIn("random sample of 20 of 50", how)
        recs2, _, _ = Client().sample(self.url + "/0", n=20, seed=1)
        self.assertEqual([r["OBJECTID"] for r in recs], [r["OBJECTID"] for r in recs2])  # reproducible

    def test_one_failing_layer_does_not_stop_the_audit(self):
        class Flaky(Client):
            def meta(self, url):
                if url.endswith("/1"):
                    raise RuntimeError("boom")
                return super().meta(url)
        res, _ = audit(self.url, client=Flaky(), now=mock_server.NOW)
        res = res.set_index("layer_id")
        self.assertIn("ERROR: boom", res.loc[1, "flags"])
        self.assertAlmostEqual(res.loc[0, "score"], 94.5, places=1)

    def test_history_detects_record_count_change(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            res, fields = audit(self.url, now=mock_server.NOW)
            self.assertEqual(write_outputs(res, fields, out, "2026-11-17 10:00:00"), [])
            mock_server.LAYERS[0]["records"].append({"OBJECTID": 51, "PIN": "P0051", "OWNER_CLASS": "R", "ACRES": 0.6})
            try:
                res2, fields2 = audit(self.url, now=mock_server.NOW)
                changes = write_outputs(res2, fields2, out, "2026-11-18 10:00:00")
            finally:
                mock_server.LAYERS[0]["records"].pop()
            self.assertEqual(len(changes), 1)
            self.assertIn("+1", changes[0])
            self.assertTrue((out / "REPORT.md").read_text().startswith("# Data AI-readiness report"))
            import pandas as pd
            self.assertEqual(len(pd.read_csv(out / "run_history.csv")), 6)  # 3 layers x 2 runs


if __name__ == "__main__":
    unittest.main()


class AIDocsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv, cls.url = mock_server.start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        mock_server.H.refuse_ids = False
        self.keep = {}
        audit(self.url, now=mock_server.NOW, keep=self.keep)

    def test_prompt_contains_schema_but_never_record_values(self):
        from ai_docs import build_prompt
        name, prof, result, n = self.keep[0]
        prompt = build_prompt(name, prof, result, n)
        self.assertIn("PIN", prompt)                         # schema is shared
        for secret in ("P0007", "P0033", "Site 5", "C012"):  # values from the mock records must not appear
            self.assertNotIn(secret, prompt)
        # check across ALL layers too
        for lid, (nm, pf, rs, cnt) in self.keep.items():
            p = build_prompt(nm, pf, rs, cnt)
            for rec in mock_server.LAYERS[lid]["records"]:
                for v in rec.values():
                    if isinstance(v, str) and len(v) > 4 and v not in ("Chesapeake", "see file", "open", "closed"):
                        self.assertNotIn(v, p)

    def test_grounded_draft_passes_and_invented_number_is_flagged(self):
        from ai_docs import build_prompt, ungrounded_numbers
        name, prof, result, n = self.keep[1]
        prompt = build_prompt(name, prof, result, n)
        good = "1. Fix the duplicate FAC_ID values.\n2. Fill NOTES (75% empty).\nThe layer has 40 records and scores 45.9."
        self.assertEqual(ungrounded_numbers(good, prompt), [])
        bad = good + " About 99% of sites are schools and 312 were added in 2019."
        self.assertEqual(ungrounded_numbers(bad, prompt), ["2019", "312", "99"])

    def test_draft_layer_writes_banner_log_and_flags_hallucination(self):
        from ai_docs import draft_layer
        name, prof, result, n = self.keep[1]
        with tempfile.TemporaryDirectory() as d:
            status, bad = draft_layer(1, name, prof, result, n, lambda p: "Summary. 777 sites exist.", Path(d))
            self.assertEqual(bad, ["777"])
            self.assertIn("CHECK", status)
            text = next((Path(d) / "ai_docs").glob("*.md")).read_text()
            self.assertIn("AI-DRAFT. NOT APPROVED", text)
            self.assertIn("777", text.split("Guardrail")[1])
            log = (Path(d) / "ai_review_log.csv").read_text()
            self.assertIn("pending human review", log)
            status2, bad2 = draft_layer(0, *self.keep[0][:1], self.keep[0][1], self.keep[0][2], self.keep[0][3],
                                        lambda p: "Parcels. 50 records.", Path(d))
            self.assertEqual(bad2, [])
            self.assertEqual(status2, "pending human review")
