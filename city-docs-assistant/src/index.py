"""Load documents, split into page-aware chunks, and search them with BM25 (no model download, fully deterministic)."""
import math
import re
from pathlib import Path

STOP = set("a an the of to in on for and or is are was were be by with at as it its this that from what how when who which "
           "does do did will would per about s".split())
PAGE = re.compile(r"\[page (\d+)\]", re.I)


def tokens(text: str):
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()) if t not in STOP and (len(t) > 1 or t.isdigit())]


def load_docs(folder):
    """Return a list of {doc, page, text}. .txt/.md files mark pages with '[page N]'; PDFs use real pages (needs pypdf)."""
    pages = []
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() in (".txt", ".md"):
            parts = PAGE.split(p.read_text(encoding="utf-8"))
            head = parts[0].strip()
            if len(parts) == 1:
                pages.append({"doc": p.name, "page": 1, "text": head})
                continue
            for i in range(1, len(parts), 2):
                text = parts[i + 1].strip()
                if i == 1 and head:
                    text = head + "\n" + text      # keep the title with the first page
                pages.append({"doc": p.name, "page": int(parts[i]), "text": text})
        elif p.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            for n, pg in enumerate(PdfReader(str(p)).pages, start=1):
                pages.append({"doc": p.name, "page": n, "text": (pg.extract_text() or "").strip()})
    return [x for x in pages if x["text"]]


def chunk_pages(pages, size=90, overlap=20):
    """Chunks never cross a page boundary, so every citation names one page."""
    chunks = []
    for pg in pages:
        words = pg["text"].split()
        step = max(1, size - overlap)
        for start in range(0, max(1, len(words)), step):
            piece = words[start:start + size]
            if not piece:
                break
            chunks.append({"doc": pg["doc"], "page": pg["page"], "chunk": len(chunks), "text": " ".join(piece)})
            if start + size >= len(words):
                break
    return chunks


class BM25Index:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.toks = [tokens(c["text"]) for c in chunks]
        self.n = len(chunks)
        self.avgdl = sum(len(t) for t in self.toks) / self.n if self.n else 0.0
        self.df = {}
        for t in self.toks:
            for w in set(t):
                self.df[w] = self.df.get(w, 0) + 1

    def idf(self, w):
        d = self.df.get(w, 0)
        return math.log(1 + (self.n - d + 0.5) / (d + 0.5))

    def score(self, q_terms, i):
        t = self.toks[i]
        dl = len(t)
        s = 0.0
        for w in q_terms:
            tf = t.count(w)
            if tf:
                s += self.idf(w) * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def search(self, query, k=4):
        q = sorted(set(tokens(query)))
        if not q:
            return []
        ranked = sorted(((self.score(q, i), i) for i in range(self.n)), reverse=True)[:k]
        out = []
        for s, i in ranked:
            cov = len(set(q) & set(self.toks[i])) / len(q)
            out.append({**self.chunks[i], "score": s, "coverage": cov})
        return out
