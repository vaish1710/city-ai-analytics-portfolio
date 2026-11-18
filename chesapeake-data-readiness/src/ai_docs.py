"""AI-assisted data documentation with built-in guardrails (human review required).

The audit flags datasets with no description and fields with no real alias. This module asks an LLM to DRAFT
the missing documentation, under rules a public-sector program can defend:

  1. PRIVACY   The model sees schema and summary statistics only: field names, types, null/distinct counts,
               scores. It never sees a record value.
  2. GROUNDING Every number in the draft must already appear in the prompt. Anything else is reported as
               "ungrounded" and the draft is marked for closer review.
  3. HUMAN     Each draft carries an AI-DRAFT banner and a logged "pending human review" status. Meanings of
               fields are guesses until the data owner confirms them.

`llm` is any function str -> str, so the logic is testable offline. `anthropic_llm()` is the real one
(needs `pip install anthropic` and ANTHROPIC_API_KEY). It has not been run live in the build environment.
"""
import csv
import re
from datetime import datetime, timezone
from pathlib import Path

NUM = re.compile(r"\d+(?:[.,]\d+)*")
LIST_MARKER = re.compile(r"^\s*(?:[-*]\s+)?\d+[.)]\s+", re.M)
BANNER = ("> **AI-DRAFT. NOT APPROVED.** Written by a language model from schema and summary statistics only. "
          "Field meanings are guesses until the data owner confirms them.\n> Reviewer: ____________  Date: ________  "
          "Decision: [ ] approve  [ ] edit  [ ] reject\n")


def build_prompt(layer_name, prof, result, record_count) -> str:
    """Facts only. No record values ever enter this string."""
    lines = [f"Dataset: {layer_name}", f"Records: {record_count}",
             f"Readiness score: {result['score']} out of 100 ({result['readiness_1_to_5']} on a 1-5 scale)",
             "Audit flags: " + (result["flags"] or "none"), "Fields (name | type | share empty | distinct values in sample):"]
    for _, r in prof.iterrows():
        lines.append(f"- {r['field']} | {r['type']} | {r['null_share']:.0%} | {r['distinct']}")
    facts = "\n".join(lines)
    return (
        "You are helping a city IT department document an open dataset for staff who are not data specialists.\n"
        "Use ONLY the facts below. Do not invent numbers, dates, owners or sources.\n"
        "Write: (1) a two-sentence plain-language summary of what the dataset appears to contain; "
        "(2) a bullet list with one line per field saying what it likely means, ending each guess with "
        "[NEEDS OWNER CONFIRMATION]; (3) exactly 3 recommended fixes to raise the readiness score, "
        "each tied to an audit flag.\n\nFACTS\n" + facts + "\n")


def _numbers(text: str) -> set:
    text = LIST_MARKER.sub("", text)
    return {n.replace(",", "").rstrip(".") for n in NUM.findall(text)}


def ungrounded_numbers(draft: str, prompt: str) -> list:
    """Numbers in the draft that do not appear anywhere in the prompt."""
    return sorted(_numbers(draft) - _numbers(prompt))


def anthropic_llm(model="claude-sonnet-5-5", max_tokens=1200):
    import anthropic  # imported lazily so the rest of the project needs no extra install
    client = anthropic.Anthropic()

    def call(prompt: str) -> str:
        msg = client.messages.create(model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return call


def draft_layer(layer_id, layer_name, prof, result, record_count, llm, out_dir: Path, model_label="llm"):
    prompt = build_prompt(layer_name, prof, result, record_count)
    draft = llm(prompt)
    bad = ungrounded_numbers(draft, prompt)
    status = "pending human review" + (" (CHECK: ungrounded numbers)" if bad else "")
    docs = Path(out_dir) / "ai_docs"
    docs.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9]+", "_", layer_name).strip("_")
    (docs / f"{layer_id}_{safe}.md").write_text(
        f"# {layer_name}: draft documentation\n\n{BANNER}\n"
        + (f"> **Guardrail:** the draft contains numbers not found in the audit facts: {', '.join(bad)}. Verify or delete them.\n\n" if bad else "")
        + draft.strip() + "\n")
    log = Path(out_dir) / "ai_review_log.csv"
    new = not log.exists()
    with log.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc", "layer_id", "layer", "model", "ungrounded_numbers", "status", "reviewer", "decision"])
        w.writerow([datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), layer_id, layer_name, model_label,
                    ";".join(bad), status, "", ""])
    return status, bad
