"""Measure the assistant on a gold question set: does it find, answer correctly, cite properly, and refuse when it should?

    python src/evaluate.py                       # no-AI extractive baseline on the sample corpus
    python src/evaluate.py --llm anthropic       # the real model (needs `pip install anthropic` and ANTHROPIC_API_KEY)
    python src/evaluate.py --docs path/to/pdfs --questions path/to/your_gold.json
Gold file: list of {"q", "answerable", "gold_doc", "facts": [strings that must appear in a correct answer]}.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from assistant import anthropic_llm, answer
from index import BM25Index, chunk_pages, load_docs

ROOT = Path(__file__).resolve().parent.parent


def evaluate(index, questions, llm=None, k=4):
    rows = []
    for item in questions:
        out = answer(item["q"], index, llm=llm, k=k)
        docs = [h["doc"] for h in out["hits"]]
        row = {"question": item["q"], "answerable": item["answerable"], "refused": out["refused"],
               "flags": "; ".join(out["flags"]), "answer": out["answer"]}
        if item["answerable"]:
            row["retrieval_hit"] = item["gold_doc"] in docs[:k]
            row["correct"] = (not out["refused"]) and all(f.lower() in out["answer"].lower() for f in item["facts"])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df):
    a, u = df[df.answerable], df[~df.answerable]
    answered = df[~df.refused]
    return {
        "answerable_questions": len(a), "unanswerable_questions": len(u),
        "retrieval_hit_rate": a["retrieval_hit"].astype(bool).mean(),
        "answer_accuracy": a["correct"].astype(bool).mean(),
        "false_refusal_rate": a["refused"].mean(),
        "correct_refusal_rate": u["refused"].mean() if len(u) else float("nan"),
        "wrong_answers_to_unanswerable": int((~u["refused"]).sum()),
        "citation_clean_rate": (answered["flags"] == "").mean() if len(answered) else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=str(ROOT / "data" / "sample_docs"))
    ap.add_argument("--questions", default=str(ROOT / "data" / "eval_questions.json"))
    ap.add_argument("--llm", choices=["none", "anthropic"], default="none")
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    a = ap.parse_args()
    index = BM25Index(chunk_pages(load_docs(a.docs)))
    qs = json.loads(Path(a.questions).read_text())
    llm = anthropic_llm(a.model) if a.llm == "anthropic" else None
    df = evaluate(index, qs, llm=llm, k=a.k)
    s = summarize(df)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    name = "anthropic" if llm else "extractive_baseline"
    df.to_csv(out / f"eval_{name}.csv", index=False)
    print(f"System: {name}  ({len(df)} questions; small sample, treat as a smoke test)")
    for k_, v in s.items():
        print(f"  {k_:<32}{v:.2f}" if isinstance(v, float) else f"  {k_:<32}{v}")
    ok = df["correct"].fillna(False).astype(bool)
    bad = df[(df.answerable & ~ok) | (~df.answerable & ~df.refused)]
    if len(bad):
        print("\nFailures:")
        for _, r in bad.iterrows():
            print(f"  - {r['question']}\n      -> {r['answer']}")


if __name__ == "__main__":
    main()
