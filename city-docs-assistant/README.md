# Ask-the-City-Documents Assistant (with evaluation)

Answers staff/resident questions from budget books, ordinances, notices and minutes, **with page-level citations**, and
refuses when the documents don't say. The point of the project is the **evaluation and guardrails**, not the chat box.

## Run
    pip install pandas pytest pypdf            # anthropic only for the live model
    python src/ask.py "How did the Board vote on the state grant?"
    python src/evaluate.py                     # no-AI baseline on the sample corpus
    ANTHROPIC_API_KEY=... python src/evaluate.py --llm anthropic
    python src/evaluate.py --docs my_pdfs/ --questions my_gold.json     # real documents
    python -m pytest tests -q                  # 13 tests

## How it works
1. `index.py` splits documents into page-bound chunks (a citation always names one page) and ranks with BM25 (own implementation, hand-checked in tests).
2. `assistant.py` refuses *without calling the model* if the best chunk covers < 50% of the question's terms.
3. The model sees only numbered sources and must cite `[S#]`, or reply `NOT FOUND`.
4. Automatic checks flag: invalid/missing citations, and any number in the answer that is not in the cited source.
5. `evaluate.py` scores a gold question set: retrieval hit, answer accuracy (required facts present), false-refusal rate, correct-refusal rate on unanswerable questions, citation-clean rate.

## Results so far (honest)
Corpus: 5 **fictional** "Example County" documents I wrote; 16 questions (12 answerable, 4 not). Tiny, so a smoke test only.

| System | Retrieval hit | Answer accuracy | False refusals | Correct refusals |
|---|---|---|---|---|
| No-AI extractive baseline | 1.00 | 0.67 | 0.08 | 0.50 (2 of 4 unanswerable questions got a wrong answer) |
| Live LLM | **not run** (no API access from where this was built) | | | |

The baseline fails exactly where expected: questions needing two facts, and near-miss questions ("speed limit on Harbor Road"
retrieves the Harbor Road project text and answers with it). Whether an LLM fixes this is the thing to measure, not assume.

## Limits
- Not run on real Chesapeake documents or with the live model; `anthropic_llm` is untested live.
- Keyword retrieval misses paraphrases; a hybrid/embedding retriever is the next step, measured on the same gold set.
- "Answer accuracy" is substring matching on required facts: strict, and can mark correct paraphrases wrong.
- The number check cannot catch wrong *words*, wrong units, or misread tables.
- **Prompt injection** inside a document is not stopped by these checks. Documents are marked untrusted in the prompt, which reduces but does not remove the risk.
- Outputs are drafts for staff to verify; this is not legal or official guidance.
