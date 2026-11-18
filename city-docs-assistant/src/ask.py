"""Ask one question.   python src/ask.py "How long is the Harbor Road project?" [--llm anthropic] [--docs folder]"""
import argparse
from pathlib import Path

from assistant import anthropic_llm, answer
from index import BM25Index, chunk_pages, load_docs

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("question")
ap.add_argument("--docs", default=str(ROOT / "data" / "sample_docs"))
ap.add_argument("--llm", choices=["none", "anthropic"], default="none")
ap.add_argument("--model", default="claude-sonnet-5-5")
a = ap.parse_args()
idx = BM25Index(chunk_pages(load_docs(a.docs)))
r = answer(a.question, idx, llm=anthropic_llm(a.model) if a.llm == "anthropic" else None)
print(r["answer"])
if r["flags"]:
    print("CHECK BEFORE USING:", "; ".join(r["flags"]))
if not r["refused"]:
    print("Sources considered:")
    for i, h in enumerate(r["hits"], 1):
        print(f"  [S{i}] {h['doc']}, page {h['page']}")
