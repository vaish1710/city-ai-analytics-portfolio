import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import pytest
from index import BM25Index, chunk_pages, load_docs, tokens
from assistant import answer, check, numbers, NOT_FOUND
from evaluate import evaluate, summarize

ROOT = Path(__file__).resolve().parent.parent


def idx_small():
    ch = [{"doc": "d1", "page": 1, "chunk": 0, "text": "flood grant funds"},
          {"doc": "d2", "page": 1, "chunk": 1, "text": "budget tax rate"},
          {"doc": "d3", "page": 1, "chunk": 2, "text": "flood warning siren"}]
    return BM25Index(ch)


def sample_index():
    return BM25Index(chunk_pages(load_docs(ROOT / "data" / "sample_docs")))


def test_bm25_hand_computed():
    hits = idx_small().search("flood grant", k=3)
    assert [h["doc"] for h in hits][:2] == ["d1", "d3"]
    assert hits[0]["score"] == pytest.approx(1.451, abs=0.01)
    assert hits[1]["score"] == pytest.approx(0.470, abs=0.01)
    assert hits[2]["score"] == 0


def test_chunks_keep_page_and_never_cross_pages():
    pages = [{"doc": "a", "page": 1, "text": " ".join(f"w{i}" for i in range(200))},
             {"doc": "a", "page": 2, "text": "second page words"}]
    ch = chunk_pages(pages, size=90, overlap=20)
    assert {c["page"] for c in ch} == {1, 2}
    assert all(len(c["text"].split()) <= 90 for c in ch)
    assert all("second" not in c["text"] for c in ch if c["page"] == 1)
    covered = set(" ".join(c["text"] for c in ch if c["page"] == 1).split())
    assert covered == {f"w{i}" for i in range(200)}      # nothing dropped


def test_load_docs_pages():
    pages = load_docs(ROOT / "data" / "sample_docs")
    assert {p["doc"] for p in pages} >= {"budget_summary.txt", "flood_resilience_plan.txt"}
    assert all(p["page"] >= 1 for p in pages)


def test_off_topic_refused_without_calling_model():
    called = []
    r = answer("quantum spaceship launch codes", sample_index(), llm=lambda p: called.append(p) or "x [S1]")
    assert r["refused"] and not called and r["answer"] == NOT_FOUND


def test_grounded_answer_passes():
    r = answer("What is the real estate tax rate?", sample_index(), llm=lambda p: "The rate is $0.95 per $100 [S1].")
    assert not r["refused"] and r["flags"] == []


def test_hallucinated_number_flagged():
    r = answer("What is the real estate tax rate?", sample_index(), llm=lambda p: "The rate is $1.25 per $100 [S1].")
    assert any("numbers not in cited" in f for f in r["flags"])


def test_invalid_and_missing_citation_flagged():
    i = sample_index()
    assert any("invalid citation" in f for f in answer("What is the real estate tax rate?", i, llm=lambda p: "It is $0.95 [S9].")["flags"])
    assert "no valid citation" in answer("What is the real estate tax rate?", i, llm=lambda p: "It is $0.95.")["flags"]


def test_number_from_uncited_source_is_flagged():
    hits = [{"text": "alpha 5"}, {"text": "beta 7"}]
    assert check("Value is 7 [S1]", hits) == ["numbers not in cited sources: 7"]
    assert check("Value is 7 [S2]", hits) == []


def test_citation_tag_digits_not_counted_as_numbers():
    assert numbers("Rate 4.5 [S12]") == {"4.5"}


def test_model_not_found_is_refusal():
    r = answer("What is the real estate tax rate?", sample_index(), llm=lambda p: "NOT FOUND")
    assert r["refused"] and r["reason"] == "model said NOT FOUND"


def test_prompt_marks_sources_untrusted_and_has_tags():
    seen = []
    answer("What is the real estate tax rate?", sample_index(), llm=lambda p: seen.append(p) or "$0.95 [S1]")
    assert "untrusted" in seen[0] and "[S1]" in seen[0] and "NOT FOUND" in seen[0]


def test_eval_regression_guard():
    qs = json.loads((ROOT / "data" / "eval_questions.json").read_text())
    s = summarize(evaluate(sample_index(), qs))
    assert s["retrieval_hit_rate"] >= 0.9
    assert s["citation_clean_rate"] == 1.0


def test_eval_scores_a_perfect_and_a_bad_model():
    qs = json.loads((ROOT / "data" / "eval_questions.json").read_text())
    gold = {q["q"]: " ".join(q["facts"]) for q in qs if q["answerable"]}
    def oracle(prompt):
        q = prompt.split("QUESTION: ")[1].strip()
        return (gold[q] + " [S1]") if q in gold else "NOT FOUND"
    s = summarize(evaluate(sample_index(), qs, llm=oracle))
    assert s["answer_accuracy"] >= 0.9 and s["wrong_answers_to_unanswerable"] == 0
    s2 = summarize(evaluate(sample_index(), qs, llm=lambda p: "made up 999 [S1]"))
    assert s2["answer_accuracy"] == 0 and s2["wrong_answers_to_unanswerable"] >= 1
